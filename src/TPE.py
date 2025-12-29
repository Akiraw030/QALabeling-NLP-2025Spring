#!/usr/bin/env python3
"""
TPE-Optimized Ensemble Weight Calculation
Computes optimal weights for blending transformer model predictions using Tree-structured Parzen Estimator (TPE).
Can run on local machine with internet connection.
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

# ============================================================================
# Configuration - Normal HuggingFace model names (will download from internet)
# ============================================================================

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
    optimize_fold_weights: bool = False
    use_voter_postprocessing: bool = False
    voter_dev_threshold: float = 0.01
    output_weights_path: str = "tpe_weights.json"
    weights_dir: Optional[str] = None  # Path to directory with .pth weight files

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

    # Use fast DeBERTa tokenizer for DeBERTa models
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


class VotersRounder:
    """Snap predictions to nearest observed training values with a deviation guard."""

    def __init__(self, train_vals: np.ndarray, dev_threshold: float = 0.01):
        clean_vals = train_vals[~np.isnan(train_vals)]
        self.unique_vals = np.sort(np.unique(clean_vals))
        self.dev_threshold = dev_threshold

    def predict(self, preds: np.ndarray) -> np.ndarray:
        preds = np.nan_to_num(preds, nan=0.5)
        idx = np.abs(preds[:, None] - self.unique_vals[None, :]).argmin(axis=1)
        snapped = self.unique_vals[idx]
        if np.std(snapped) < self.dev_threshold:
            return preds
        return snapped


def apply_voter_postprocessing(preds: np.ndarray, train_df: pd.DataFrame, dev_threshold: float) -> np.ndarray:
    """Apply VotersRounder per target using training column distributions."""
    rounded = preds.copy()
    for i, col in enumerate(TARGET_COLS):
        voter = VotersRounder(train_df[col].values, dev_threshold=dev_threshold)
        rounded[:, i] = voter.predict(preds[:, i])
    return rounded


# ============================================================================
# Inference Functions
# ============================================================================

def get_weight_paths(model_dir: str, prefix: str) -> List[str]:
    """Get sorted list of .pth files matching the prefix."""
    if not os.path.exists(model_dir):
        return []
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth") and f.startswith(prefix)]
    paths = [os.path.join(model_dir, f) for f in files]
    return sorted(paths)


def inference_single_model(loader: DataLoader, spec: ModelSpec, weight_paths: Optional[List[str]] = None) -> np.ndarray:
    """Run inference with a single model from HuggingFace, optionally loading local weights.
    
    If weight_paths provided, averages predictions from all weight checkpoints.
    Otherwise, uses pretrained model directly.
    """
    if weight_paths:
        # Average predictions across multiple weight checkpoints
        all_preds = []
        for weight_path in weight_paths:
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
                for batch in tqdm(loader, desc=f"{spec.name}:{os.path.basename(weight_path)}", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None and spec.pass_token_type_ids:
                        token_type_ids = token_type_ids.to(device)
                    else:
                        token_type_ids = None
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    preds.append(outputs.sigmoid().cpu().numpy())
            
            all_preds.append(np.concatenate(preds))
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Average across checkpoints
        return np.mean(np.stack(all_preds), axis=0)
    else:
        # Use pretrained model directly
        model = QuestModel(
            model_name=spec.base_model,
            num_targets=len(TARGET_COLS),
            pooling_strategy=spec.pooling_strategy,
            trust_remote_code=spec.trust_remote_code,
        )
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{spec.name}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None and spec.pass_token_type_ids:
                    token_type_ids = token_type_ids.to(device)
                else:
                    token_type_ids = None
                outputs = model(input_ids, attention_mask, token_type_ids)
                preds.append(outputs.sigmoid().cpu().numpy())

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return np.concatenate(preds)


# ============================================================================
# TPE Weight Optimization
# ============================================================================

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


# ============================================================================
# Main Execution
# ============================================================================

def main(args):
    cfg = Config(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_frac=args.sample_frac,
        tpe_trials=args.tpe_trials,
        output_weights_path=args.output,
    )

    print(f"\nConfig:")
    print(f"  Train: {cfg.train_path}")
    print(f"  Test: {cfg.test_path}")
    print(f"  Sample fraction: {cfg.sample_frac}")
    print(f"  TPE trials: {cfg.tpe_trials}")
    print(f"  Output weights: {cfg.output_weights_path}")

    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    if cfg.sample_frac < 1.0:
        train_df = train_df.sample(frac=cfg.sample_frac, random_state=cfg.seed).reset_index(drop=True)
        print(f"Using {cfg.sample_frac:.1%} of training data: {len(train_df)} samples")
    else:
        print(f"Using full training data: {len(train_df)} samples")

    print(f"Test data: {len(test_df)} samples")
    y_true = train_df[TARGET_COLS].values

    # Create model specs
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

    # Collect predictions from all models
    print(f"\nCollecting predictions from {len(MODEL_SPECS)} models...")
    model_train_preds = []
    model_test_preds = []
    
    # Prepare weight paths if weights_dir specified
    weights_dir = getattr(args, 'weights_dir', None) or cfg.weights_dir
    if weights_dir:
        print(f"Will load weights from: {weights_dir}")

    for spec in MODEL_SPECS:
        print(f"\n{'='*60}")
        print(f"Processing: {spec.name}")
        print(f"{'='*60}")

        try:
            loader, _ = build_loader(train_df, spec, cfg)
            test_loader, _ = build_loader(test_df, spec, cfg)
            
            # Find weight files if weights_dir specified
            weight_files = None
            if weights_dir:
                weight_prefix = f"{spec.name}_fold"
                weight_files = get_weight_paths(weights_dir, weight_prefix)
                if weight_files:
                    print(f"Found {len(weight_files)} weight files for {spec.name}")
                else:
                    print(f"No weight files found for {spec.name}, using pretrained model")

            print(f"Running inference on train set...")
            train_pred = inference_single_model(loader, spec, weight_files)

            print(f"Running inference on test set...")
            test_pred = inference_single_model(test_loader, spec, weight_files)

            score = mean_spearman(y_true, train_pred)
            print(f"{spec.name} train score: {score:.4f}")

            model_train_preds.append(train_pred)
            model_test_preds.append(test_pred)

        except Exception as e:
            print(f"ERROR processing {spec.name}: {e}")
            print(f"Skipping {spec.name}...")
            continue

    if len(model_train_preds) == 0:
        print("ERROR: No models were successfully processed!")
        return

    model_train_preds_np = np.stack(model_train_preds)
    model_test_preds_np = np.stack(model_test_preds)

    print(f"\nCollected predictions from {len(model_train_preds)} models")
    print(f"Train predictions shape: {model_train_preds_np.shape}")
    print(f"Test predictions shape: {model_test_preds_np.shape}")

    # TPE optimization
    print(f"\nRunning TPE optimization across {len(MODEL_SPECS)} models...")
    print(f"TPE trials: {cfg.tpe_trials}")

    model_weights = tpe_weight_search(
        model_train_preds_np, y_true,
        max_evals=cfg.tpe_trials,
        label="model"
    )

    # Final predictions
    final_train = blend_preds(model_train_preds_np, model_weights)
    final_score = mean_spearman(y_true, final_train)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Training Spearman Score: {final_score:.4f}")
    print("="*60)

    # Save weights
    weight_payload = {
        "final_score": float(final_score),
        "model_weights": {spec.name: float(w) for spec, w in zip(MODEL_SPECS[:len(model_train_preds)], model_weights)},
        "config": {
            "tpe_trials": cfg.tpe_trials,
            "sample_frac": cfg.sample_frac,
        }
    }

    with open(cfg.output_weights_path, "w", encoding="utf-8") as f:
        json.dump(weight_payload, f, indent=2)

    print(f"\n✓ Weights saved to: {cfg.output_weights_path}")
    print("\nWeight summary:")
    print(json.dumps(weight_payload["model_weights"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate optimal ensemble weights using TPE optimization"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="train.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="test.csv",
        help="Path to test CSV file"
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
    parser.add_argument(
        "--output",
        type=str,
        default="tpe_weights.json",
        help="Output path for calculated weights JSON"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Optional: directory containing .pth weight files (e.g., ./model). If not specified, uses pretrained models."
    )

    args = parser.parse_args()
    main(args)
