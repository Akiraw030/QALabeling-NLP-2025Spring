#!/usr/bin/env python3
"""
Hierarchical TPE Optimization
First optimizes fold weights for each model, then optimizes model weights using the fold-optimized predictions.
"""

import os
import gc
import json
import html
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DebertaV2TokenizerFast
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Target columns
TARGET_COLS = [
    "question_asker_intent_understanding", "question_body_critical", "question_conversational",
    "question_expect_short_answer", "question_fact_seeking", "question_has_commonly_accepted_answer",
    "question_interestingness_others", "question_interestingness_self", "question_multi_intent",
    "question_not_really_a_question", "question_opinion_seeking", "question_type_choice",
    "question_type_compare", "question_type_consequence", "question_type_definition",
    "question_type_entity", "question_type_instructions", "question_type_procedure",
    "question_type_reason_explanation", "question_type_spelling", "question_well_written",
    "answer_helpful", "answer_level_of_information", "answer_plausible", "answer_relevance",
    "answer_satisfaction", "answer_type_instructions", "answer_type_procedure",
    "answer_type_reason_explanation", "answer_well_written",
]

MODEL_PATHS = {
    'deberta_v3': {
        'model': 'microsoft/deberta-v3-base',
        'tokenizer': 'microsoft/deberta-v3-base',
    },
    'modernbert': {
        'model': 'answerdotai/ModernBERT-base',
        'tokenizer': 'answerdotai/ModernBERT-base',
    },
    'electra': {
        'model': 'google/electra-base-discriminator',
        'tokenizer': 'google/electra-base-discriminator',
    },
    'xlnet': {
        'model': 'xlnet-base-cased',
        'tokenizer': 'xlnet-base-cased',
    },
    'llama': {
        'model': 'meta-llama/Llama-3.2-1B',
        'tokenizer': 'meta-llama/Llama-3.2-1B',
    },
    'qwen': {
        'model': 'Qwen/Qwen3-0.6B',
        'tokenizer': 'Qwen/Qwen3-0.6B',
    },
}

@dataclass
class ModelSpec:
    """Specification for each model in the ensemble."""
    name: str
    base_model: str
    tokenizer_path: Optional[str] = None
    pooling_strategy: str = "arch1_6groups"
    max_len: int = 512
    batch_size: int = 8
    pass_token_type_ids: bool = True
    trust_remote_code: bool = False

@dataclass
class Config:
    """Global configuration."""
    train_path: str = "./data/train.csv"
    test_path: str = "./data/test.csv"
    sample_frac: float = 1.0
    num_workers: int = 4
    tpe_trials: int = 50
    seed: int = 42
    weights_dir: str = "./model"
    output_dir: str = "./tpe_results"

# ============================================================================
# Data Processing
# ============================================================================

def modern_preprocess(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    return " ".join(text.split())


class QuestDataset(Dataset):
    """Dataset for Q&A pairs."""
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.questions = [
            modern_preprocess(t) + " " + modern_preprocess(b)
            for t, b in zip(df["question_title"].values, df["question_body"].values)
        ]
        self.answers = [modern_preprocess(a) for a in df["answer"].values]

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.questions[idx]
        answer = self.answers[idx]
        inputs = self.tokenizer(
            question, answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in inputs:
            item["token_type_ids"] = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        return item


# ============================================================================
# Model Architecture
# ============================================================================

class QuestModel(nn.Module):
    """6-head QA model with flexible pooling strategies."""
    def __init__(self, model_name: str, num_targets: int, pooling_strategy: str = "arch1_6groups",
                 dropout_rate: float = 0.1, trust_remote_code: bool = False):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if pooling_strategy == "cls_all":
            self.config.update({"output_hidden_states": True})
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=trust_remote_code)
        hidden_size = self.config.hidden_size

        # 6-group task indices
        self.idx_g1 = [3, 4, 5, 16, 17]
        self.idx_g2 = [0, 1, 6, 7, 20]
        self.idx_g3 = [2, 10]
        self.idx_g4 = [8, 9, 11, 12, 13, 14, 15, 18, 19]
        self.idx_g5 = [26, 27]
        self.idx_g6 = [21, 22, 23, 24, 25, 28, 29]

        if self.pooling_strategy == "arch1_6groups":
            self.head_g1 = self._make_head(hidden_size * 3, len(self.idx_g1), dropout_rate)
            self.head_g2 = self._make_head(hidden_size * 3, len(self.idx_g2), dropout_rate)
            self.head_g3 = self._make_head(hidden_size * 3, len(self.idx_g3), dropout_rate)
            self.head_g4 = self._make_head(hidden_size * 3, len(self.idx_g4), dropout_rate)
            self.head_g5 = self._make_head(hidden_size * 3, len(self.idx_g5), dropout_rate)
            self.head_g6 = self._make_head(hidden_size * 3, len(self.idx_g6), dropout_rate)

    def _make_head(self, input_dim: int, output_dim: int, dropout_rate: float):
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, output_dim),
        )

    def _masked_mean_pooling(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        return torch.sum(hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _get_pooling_features(self, last_hidden_state, attention_mask, token_type_ids):
        cls_token = last_hidden_state[:, 0, :]
        global_avg = self._masked_mean_pooling(last_hidden_state, attention_mask)
        if token_type_ids is None:
            return cls_token, global_avg, global_avg, global_avg
        q_mask = attention_mask * (1 - token_type_ids)
        a_mask = attention_mask * token_type_ids
        q_avg = self._masked_mean_pooling(last_hidden_state, q_mask)
        a_avg = self._masked_mean_pooling(last_hidden_state, a_mask)
        return cls_token, global_avg, q_avg, a_avg

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        backbone_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**backbone_kwargs)

        cls_token, global_avg, q_avg, a_avg = self._get_pooling_features(
            outputs.last_hidden_state, attention_mask, token_type_ids
        )
        feat_q = torch.cat([cls_token, global_avg, q_avg], dim=1)
        feat_a = torch.cat([cls_token, global_avg, a_avg], dim=1)
        
        out_g1 = self.head_g1(feat_q)
        out_g2 = self.head_g2(feat_q)
        out_g3 = self.head_g3(feat_q)
        out_g4 = self.head_g4(feat_q)
        out_g5 = self.head_g5(feat_a)
        out_g6 = self.head_g6(feat_a)

        output = torch.zeros(input_ids.size(0), 30, dtype=out_g1.dtype, device=input_ids.device)
        output[:, self.idx_g1] = out_g1
        output[:, self.idx_g2] = out_g2
        output[:, self.idx_g3] = out_g3
        output[:, self.idx_g4] = out_g4
        output[:, self.idx_g5] = out_g5
        output[:, self.idx_g6] = out_g6
        return output


# ============================================================================
# Utility Functions
# ============================================================================

def mean_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean Spearman correlation across all targets."""
    scores = []
    for i in range(y_true.shape[1]):
        score, _ = spearmanr(y_true[:, i], y_pred[:, i])
        scores.append(0.0 if np.isnan(score) else score)
    return float(np.mean(scores))


def build_loader(df: pd.DataFrame, spec: ModelSpec, cfg: Config) -> Tuple[DataLoader, any]:
    """Build DataLoader for a given model spec."""
    tokenizer_name = spec.tokenizer_path or spec.base_model

    if "deberta" in spec.base_model.lower():
        tokenizer = DebertaV2TokenizerFast.from_pretrained(
            tokenizer_name,
            trust_remote_code=spec.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=spec.trust_remote_code,
        )

    dataset = QuestDataset(df, tokenizer, max_len=spec.max_len)
    loader = DataLoader(
        dataset, batch_size=spec.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return loader, tokenizer


def get_weight_paths(model_dir: str, prefix: str) -> List[str]:
    """Get sorted list of .pth files matching the prefix."""
    if not os.path.exists(model_dir):
        return []
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth") and f.startswith(prefix)]
    paths = [os.path.join(model_dir, f) for f in files]
    return sorted(paths)


def tpe_weight_search(preds: np.ndarray, y_true: np.ndarray, max_evals: int, label: str = "model") -> np.ndarray:
    """Use TPE to find optimal weights for blending predictions."""
    n = preds.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float32)

    space = {f"w_{i}": hp.uniform(f"w_{i}", 0.0, 1.0) for i in range(n)}

    def objective(params: Dict[str, float]) -> Dict[str, float]:
        weights = np.array([params[f"w_{i}"] for i in range(n)], dtype=np.float64)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / weights.sum()
        blended = np.tensordot(weights, preds, axes=((0), (0)))
        score = mean_spearman(y_true, blended)
        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    best_params = fmin(
        fn=objective, space=space, algo=tpe.suggest,
        max_evals=max_evals, trials=trials, verbose=0
    )
    
    raw = np.array([best_params[f"w_{i}"] for i in range(n)], dtype=np.float64)
    raw = np.clip(raw, 1e-6, None)
    weights = raw / raw.sum()
    
    print(f"\n{label.upper()} WEIGHTS:")
    for i, w in enumerate(weights):
        print(f"  {label}_{i}: {w:.4f}")
    
    return weights.astype(np.float32)


def blend_preds(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Blend predictions using weights."""
    return np.tensordot(weights, preds, axes=((0), (0)))


def load_fold_predictions(fold_paths: List[str], loader: DataLoader, spec: ModelSpec) -> Tuple[np.ndarray, List[str]]:
    """Load predictions from multiple fold weight files."""
    fold_preds = []
    fold_names = []
    
    for weight_path in sorted(fold_paths):
        model = QuestModel(
            model_name=spec.base_model,
            num_targets=len(TARGET_COLS),
            pooling_strategy=spec.pooling_strategy,
            trust_remote_code=spec.trust_remote_code,
        )
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Loading {os.path.basename(weight_path)}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None and spec.pass_token_type_ids:
                    token_type_ids = token_type_ids.to(device)
                else:
                    token_type_ids = None
                outputs = model(input_ids, attention_mask, token_type_ids)
                preds.append(outputs.sigmoid().cpu().numpy())
        
        fold_pred = np.concatenate(preds)
        fold_preds.append(fold_pred)
        fold_names.append(os.path.basename(weight_path).replace('.pth', ''))
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    return np.stack(fold_preds), fold_names


# ============================================================================
# Stage 1: Fold-level Optimization
# ============================================================================

def optimize_model_folds(model_name: str, spec: ModelSpec, train_loader: DataLoader, 
                         test_loader: DataLoader, y_true: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Optimize fold weights for a single model."""
    print(f"\n{'='*60}")
    print(f"FOLD OPTIMIZATION: {model_name}")
    print(f"{'='*60}")
    
    # Find fold weight files
    fold_prefix = f"{model_name}_fold"
    fold_files = get_weight_paths(cfg.weights_dir, fold_prefix)
    
    if not fold_files:
        print(f"WARNING: No fold files found for {model_name}, skipping...")
        return None, None, None
    
    print(f"Found {len(fold_files)} fold files")
    
    # Load fold predictions
    print("Loading fold predictions from training set...")
    fold_train_preds, fold_names = load_fold_predictions(fold_files, train_loader, spec)
    
    print("Loading fold predictions from test set...")
    fold_test_preds, _ = load_fold_predictions(fold_files, test_loader, spec)
    
    # Individual fold scores
    print("\nIndividual fold scores:")
    for i, fold_name in enumerate(fold_names):
        score = mean_spearman(y_true, fold_train_preds[i])
        print(f"  {fold_name}: {score:.4f}")
    
    # Optimize fold weights
    print(f"\nRunning TPE optimization for {len(fold_files)} folds...")
    fold_weights = tpe_weight_search(
        fold_train_preds, y_true,
        max_evals=cfg.tpe_trials,
        label=f"{model_name}_fold"
    )
    
    # Blend predictions
    blended_train = blend_preds(fold_train_preds, fold_weights)
    blended_test = blend_preds(fold_test_preds, fold_weights)
    
    fold_score = mean_spearman(y_true, blended_train)
    print(f"\nFold-optimized score: {fold_score:.4f}")
    
    # Save fold weights
    fold_result = {
        "model": model_name,
        "score": float(fold_score),
        "fold_weights": {name: float(w) for name, w in zip(fold_names, fold_weights)},
    }
    
    return blended_train, blended_test, fold_result


# ============================================================================
# Stage 2: Model-level Optimization
# ============================================================================

def optimize_model_ensemble(model_train_preds: List[np.ndarray], model_test_preds: List[np.ndarray],
                            model_names: List[str], y_true: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Optimize weights across different models using their fold-optimized predictions."""
    print(f"\n{'='*60}")
    print(f"MODEL ENSEMBLE OPTIMIZATION")
    print(f"{'='*60}")
    
    model_train_preds_np = np.stack(model_train_preds)
    model_test_preds_np = np.stack(model_test_preds)
    
    print(f"\nOptimizing across {len(model_names)} models:")
    for i, name in enumerate(model_names):
        score = mean_spearman(y_true, model_train_preds[i])
        print(f"  {name}: {score:.4f}")
    
    # Optimize model weights
    print(f"\nRunning TPE optimization for model ensemble...")
    model_weights = tpe_weight_search(
        model_train_preds_np, y_true,
        max_evals=cfg.tpe_trials,
        label="model"
    )
    
    # Blend predictions
    final_train = blend_preds(model_train_preds_np, model_weights)
    final_test = blend_preds(model_test_preds_np, model_weights)
    
    final_score = mean_spearman(y_true, final_train)
    print(f"\nFinal ensemble score: {final_score:.4f}")
    
    model_result = {
        "final_score": float(final_score),
        "model_weights": {name: float(w) for name, w in zip(model_names, model_weights)},
    }
    
    return final_train, final_test, model_result


# ============================================================================
# Main Hierarchical Optimization
# ============================================================================

def main(args):
    cfg = Config(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_frac=args.sample_frac,
        tpe_trials=args.tpe_trials,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
    )
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("HIERARCHICAL TPE OPTIMIZATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Train: {cfg.train_path}")
    print(f"  Test: {cfg.test_path}")
    print(f"  Weights dir: {cfg.weights_dir}")
    print(f"  Output dir: {cfg.output_dir}")
    print(f"  Sample fraction: {cfg.sample_frac}")
    print(f"  TPE trials: {cfg.tpe_trials}")
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)
    
    if cfg.sample_frac < 1.0:
        train_df = train_df.sample(frac=cfg.sample_frac, random_state=cfg.seed).reset_index(drop=True)
        print(f"Using {cfg.sample_frac:.1%} of training data: {len(train_df)} samples")
    else:
        print(f"Using full training data: {len(train_df)} samples")
    
    y_true = train_df[TARGET_COLS].values
    
    # Define models to process
    MODEL_SPECS = [
        ModelSpec(
            name="deberta_v3",
            base_model=MODEL_PATHS['deberta_v3']['model'],
            tokenizer_path=MODEL_PATHS['deberta_v3']['tokenizer'],
            batch_size=8,
            pass_token_type_ids=True,
        ),
        ModelSpec(
            name="modernbert",
            base_model=MODEL_PATHS['modernbert']['model'],
            tokenizer_path=MODEL_PATHS['modernbert']['tokenizer'],
            batch_size=8,
            pass_token_type_ids=False,
        ),
        ModelSpec(
            name="electra",
            base_model=MODEL_PATHS['electra']['model'],
            tokenizer_path=MODEL_PATHS['electra']['tokenizer'],
            batch_size=8,
            pass_token_type_ids=True,
        ),
        ModelSpec(
            name="xlnet",
            base_model=MODEL_PATHS['xlnet']['model'],
            tokenizer_path=MODEL_PATHS['xlnet']['tokenizer'],
            batch_size=8,
            pass_token_type_ids=True,
        ),
        ModelSpec(
            name="llama",
            base_model=MODEL_PATHS['llama']['model'],
            tokenizer_path=MODEL_PATHS['llama']['tokenizer'],
            batch_size=4,
            pass_token_type_ids=False,
            trust_remote_code=True,
        ),
        ModelSpec(
            name="qwen",
            base_model=MODEL_PATHS['qwen']['model'],
            tokenizer_path=MODEL_PATHS['qwen']['tokenizer'],
            batch_size=4,
            pass_token_type_ids=False,
            trust_remote_code=True,
        ),
    ]
    
    # ========================================================================
    # STAGE 1: Optimize folds for each model
    # ========================================================================
    print("\n" + "#"*60)
    print("# STAGE 1: FOLD-LEVEL OPTIMIZATION")
    print("#"*60)
    
    model_train_preds = []
    model_test_preds = []
    model_names = []
    fold_results = {}
    
    for spec in MODEL_SPECS:
        try:
            # Build loaders
            train_loader, _ = build_loader(train_df, spec, cfg)
            test_loader, _ = build_loader(test_df, spec, cfg)
            
            # Optimize folds
            blended_train, blended_test, fold_result = optimize_model_folds(
                spec.name, spec, train_loader, test_loader, y_true, cfg
            )
            
            if blended_train is not None:
                model_train_preds.append(blended_train)
                model_test_preds.append(blended_test)
                model_names.append(spec.name)
                fold_results[spec.name] = fold_result
                
                # Save individual fold weights
                fold_output_path = os.path.join(cfg.output_dir, f"fold_weights_{spec.name}.json")
                with open(fold_output_path, "w", encoding="utf-8") as f:
                    json.dump(fold_result, f, indent=2)
                print(f"✓ Saved fold weights to: {fold_output_path}")
        
        except Exception as e:
            print(f"ERROR processing {spec.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(model_train_preds) == 0:
        print("ERROR: No models were successfully processed!")
        return
    
    # ========================================================================
    # STAGE 2: Optimize model ensemble weights
    # ========================================================================
    print("\n" + "#"*60)
    print("# STAGE 2: MODEL-LEVEL OPTIMIZATION")
    print("#"*60)
    
    final_train, final_test, model_result = optimize_model_ensemble(
        model_train_preds, model_test_preds, model_names, y_true, cfg
    )
    
    # ========================================================================
    # Save final results
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save model weights
    model_output_path = os.path.join(cfg.output_dir, "model_weights.json")
    model_result["config"] = {
        "tpe_trials": cfg.tpe_trials,
        "sample_frac": cfg.sample_frac,
    }
    with open(model_output_path, "w", encoding="utf-8") as f:
        json.dump(model_result, f, indent=2)
    print(f"✓ Saved model weights to: {model_output_path}")
    
    # Save complete hierarchical results
    hierarchical_result = {
        "final_score": model_result["final_score"],
        "model_weights": model_result["model_weights"],
        "fold_weights": fold_results,
        "config": {
            "tpe_trials": cfg.tpe_trials,
            "sample_frac": cfg.sample_frac,
            "weights_dir": cfg.weights_dir,
        }
    }
    
    hierarchical_output_path = os.path.join(cfg.output_dir, "hierarchical_weights.json")
    with open(hierarchical_output_path, "w", encoding="utf-8") as f:
        json.dump(hierarchical_result, f, indent=2)
    print(f"✓ Saved hierarchical weights to: {hierarchical_output_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nFinal Training Spearman Score: {model_result['final_score']:.4f}")
    print("\nModel Weights:")
    for name, weight in model_result["model_weights"].items():
        print(f"  {name}: {weight:.4f}")
    print("\nFold-optimized Scores:")
    for name, result in fold_results.items():
        print(f"  {name}: {result['score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical TPE optimization: optimize folds first, then models"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="./data/train.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="./data/test.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Directory containing fold .pth weight files (e.g., ./model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tpe_results",
        help="Output directory for all weight files"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0)"
    )
    parser.add_argument(
        "--tpe-trials",
        type=int,
        default=50,
        help="Number of TPE optimization trials"
    )
    
    args = parser.parse_args()
    main(args)
