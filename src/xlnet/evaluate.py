import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import html

# ------------------- 配置 (Configuration) -------------------
class CFG:
    # 模型權重資料夾 (請確保這裡放的是 Regression 版本的 .pth)
    model_dir = "model/v19_xlnet" 
    
    # 訓練資料路徑 (用來評估)
    data_path = "./data/train.csv"
    
    base_model = "xlnet-base-cased" 
    pooling_strategy = 'arch1_6groups' 
    max_len = 512
    batch_size = 16
    num_workers = 4
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 原始 30 個目標欄位
TARGET_COLS = [
    'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
    'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
    'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
    'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
    'question_type_compare', 'question_type_consequence', 'question_type_definition',
    'question_type_entity', 'question_type_instructions', 'question_type_procedure',
    'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
    'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',
    'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure',
    'answer_type_reason_explanation', 'answer_well_written'
]

# ------------------- 資料處理 -------------------
def modern_preprocess(text):
    if pd.isna(text): return ""
    text = str(text)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text

class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.questions = [
            modern_preprocess(t) + " " + modern_preprocess(b) 
            for t, b in zip(df['question_title'].values, df['question_body'].values)
        ]
        self.answers = [modern_preprocess(a) for a in df['answer'].values]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        inputs = self.tokenizer(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        item = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }
        
        if 'token_type_ids' in inputs:
            item['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            
        return item

# ------------------- 模型定義 (Regression Version) -------------------
class QuestModel(nn.Module):
    def __init__(self, model_name, num_targets, pooling_strategy='arch1_6groups', dropout_rate=0.1):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name)
        
        if pooling_strategy == 'cls_all':
            self.config.update({'output_hidden_states': True})
            
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        # 定義 6-Head 分群索引
        self.idx_g1 = [3, 4, 5, 16, 17]          
        self.idx_g2 = [0, 1, 6, 7, 20]           
        self.idx_g3 = [2, 10]                    
        self.idx_g4 = [8, 9, 11, 12, 13, 14, 15, 18, 19] 
        self.idx_g5 = [26, 27]                   
        self.idx_g6 = [21, 22, 23, 24, 25, 28, 29] 
        
        if self.pooling_strategy == 'arch1_6groups':
            # 回歸版本：輸出維度直接等於該組的目標數量
            self.head_g1 = self._make_head(hidden_size * 3, len(self.idx_g1), dropout_rate)
            self.head_g2 = self._make_head(hidden_size * 3, len(self.idx_g2), dropout_rate)
            self.head_g3 = self._make_head(hidden_size * 3, len(self.idx_g3), dropout_rate)
            self.head_g4 = self._make_head(hidden_size * 3, len(self.idx_g4), dropout_rate)
            self.head_g5 = self._make_head(hidden_size * 3, len(self.idx_g5), dropout_rate)
            self.head_g6 = self._make_head(hidden_size * 3, len(self.idx_g6), dropout_rate)
            
        elif self.pooling_strategy == 'cls_all':
            # Concatenate CLS tokens from all hidden layers
            num_hidden_states = self.config.num_hidden_layers + 1
            cls_concat_dim = hidden_size * num_hidden_states
            
            self.head_g1 = self._make_head(cls_concat_dim, len(self.idx_g1), dropout_rate)
            self.head_g2 = self._make_head(cls_concat_dim, len(self.idx_g2), dropout_rate)
            self.head_g3 = self._make_head(cls_concat_dim, len(self.idx_g3), dropout_rate)
            self.head_g4 = self._make_head(cls_concat_dim, len(self.idx_g4), dropout_rate)
            self.head_g5 = self._make_head(cls_concat_dim, len(self.idx_g5), dropout_rate)
            self.head_g6 = self._make_head(cls_concat_dim, len(self.idx_g6), dropout_rate)
            
        elif self.pooling_strategy == 'mlp_only':
            # MLP that reduces sequence length dimension to 1
            self.mlp = nn.Sequential(
                nn.Linear(512, hidden_size // 2),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, 1)
            )
            self.head_g1 = self._make_head(hidden_size, len(self.idx_g1), dropout_rate)
            self.head_g2 = self._make_head(hidden_size, len(self.idx_g2), dropout_rate)
            self.head_g3 = self._make_head(hidden_size, len(self.idx_g3), dropout_rate)
            self.head_g4 = self._make_head(hidden_size, len(self.idx_g4), dropout_rate)
            self.head_g5 = self._make_head(hidden_size, len(self.idx_g5), dropout_rate)
            self.head_g6 = self._make_head(hidden_size, len(self.idx_g6), dropout_rate)

    def _make_head(self, input_dim, output_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, output_dim)
        )

    def _masked_mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_pooling_features(self, last_hidden_state, attention_mask, token_type_ids):
        cls_token = last_hidden_state[:, 0, :]
        global_avg = self._masked_mean_pooling(last_hidden_state, attention_mask)
        
        if token_type_ids is None:
            q_avg = global_avg; a_avg = global_avg
        else:
            q_mask = attention_mask * (1 - token_type_ids)
            q_avg = self._masked_mean_pooling(last_hidden_state, q_mask)
            a_mask = attention_mask * token_type_ids
            a_avg = self._masked_mean_pooling(last_hidden_state, a_mask)
            
        return cls_token, global_avg, q_avg, a_avg
    
    def _pool_cls_concat(self, all_hidden_states):
        """Concatenate CLS tokens from all hidden layers"""
        cls_embeddings = [layer[:, 0, :] for layer in all_hidden_states]
        return torch.cat(cls_embeddings, dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # XLNet supports token_type_ids
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        
        if self.pooling_strategy == 'arch1_6groups':
            cls, glob, q, a = self._get_pooling_features(last_hidden_state, attention_mask, token_type_ids)
            
            feat_pure_q = torch.cat([cls, glob, q], dim=1)
            feat_pure_a = torch.cat([cls, glob, a], dim=1)
            
            out_g1 = self.head_g1(feat_pure_q)
            out_g2 = self.head_g2(feat_pure_q)
            out_g3 = self.head_g3(feat_pure_q)
            out_g4 = self.head_g4(feat_pure_q)
            out_g5 = self.head_g5(feat_pure_a)
            out_g6 = self.head_g6(feat_pure_a)
            
            batch_size = input_ids.size(0)
            output = torch.zeros(batch_size, 30, dtype=out_g1.dtype, device=input_ids.device)
            
            output[:, self.idx_g1] = out_g1
            output[:, self.idx_g2] = out_g2
            output[:, self.idx_g3] = out_g3
            output[:, self.idx_g4] = out_g4
            output[:, self.idx_g5] = out_g5
            output[:, self.idx_g6] = out_g6
            
            return output
            
        elif self.pooling_strategy == 'cls_all':
            # Concatenate CLS tokens from all hidden layers
            a_feat = self._pool_cls_concat(outputs.hidden_states)
            
            out_g1 = self.head_g1(a_feat)
            out_g2 = self.head_g2(a_feat)
            out_g3 = self.head_g3(a_feat)
            out_g4 = self.head_g4(a_feat)
            out_g5 = self.head_g5(a_feat)
            out_g6 = self.head_g6(a_feat)
            
            batch_size = input_ids.size(0)
            output = torch.zeros(batch_size, 30, dtype=out_g1.dtype, device=input_ids.device)
            
            output[:, self.idx_g1] = out_g1
            output[:, self.idx_g2] = out_g2
            output[:, self.idx_g3] = out_g3
            output[:, self.idx_g4] = out_g4
            output[:, self.idx_g5] = out_g5
            output[:, self.idx_g6] = out_g6
            
            return output
            
        elif self.pooling_strategy == 'mlp_only':
            batch_size, seq_len, hidden = last_hidden_state.shape
            transposed = last_hidden_state.transpose(1, 2)
            reduced = self.mlp(transposed)
            a_feat = reduced.squeeze(-1)
            
            out_g1 = self.head_g1(a_feat)
            out_g2 = self.head_g2(a_feat)
            out_g3 = self.head_g3(a_feat)
            out_g4 = self.head_g4(a_feat)
            out_g5 = self.head_g5(a_feat)
            out_g6 = self.head_g6(a_feat)
            
            output = torch.zeros(batch_size, 30, dtype=out_g1.dtype, device=input_ids.device)
            
            output[:, self.idx_g1] = out_g1
            output[:, self.idx_g2] = out_g2
            output[:, self.idx_g3] = out_g3
            output[:, self.idx_g4] = out_g4
            output[:, self.idx_g5] = out_g5
            output[:, self.idx_g6] = out_g6
            
            return output
            
        return None

# ------------------- 推論與評估邏輯 -------------------
def inference_fn(test_loader, model, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            y_preds = model(input_ids, attention_mask, token_type_ids)
            # Regression Output: Sigmoid -> Probability
            preds.append(y_preds.sigmoid().cpu().numpy())
            
    return np.concatenate(preds)

def evaluate_metrics(true_df, pred_df, target_cols):
    print("\n" + "="*60)
    print(f"{'Target Column':<40} | {'Spearman':<12}")
    print("-" * 60)
    
    scores = []
    
    for col in target_cols:
        y_true = true_df[col].values
        y_pred = pred_df[col].values
        
        # 只計算原始 Spearman
        score, _ = spearmanr(y_true, y_pred)
        scores.append(score)
        
        print(f"{col:<40} | {score:.4f}")
        
    print("-" * 60)
    print(f"{'AVERAGE':<40} | {np.mean(scores):.4f}")
    print("="*60 + "\n")

# ------------------- Main -------------------
if __name__ == '__main__':
    if not os.path.exists(CFG.data_path):
        print(f"Error: Data file not found at {CFG.data_path}")
        sys.exit(1)
        
    print(f"Loading data from {CFG.data_path}...")
    df = pd.read_csv(CFG.data_path)
    print(f"Data shape: {df.shape}")
    
    # 1. Prepare DataLoader
    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
    dataset = QuestDataset(df, tokenizer, max_len=CFG.max_len)
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    
    # 2. Find Models
    if not os.path.exists(CFG.model_dir):
        print(f"Error: Model directory {CFG.model_dir} does not exist.")
        sys.exit(1)
        
    weight_paths = [os.path.join(CFG.model_dir, f) for f in os.listdir(CFG.model_dir) if f.endswith('.pth')]
    if not weight_paths:
        print(f"No .pth models found in {CFG.model_dir}")
        sys.exit(1)
    print(f"Found {len(weight_paths)} models: {[os.path.basename(p) for p in weight_paths]}")
    
    # 3. Inference loop
    model_preds = []
    
    for weight_path in weight_paths:
        print(f"\nProcessing {os.path.basename(weight_path)}...")
        
        # 回歸模型不需要 target_encoder，直接傳入 num_targets=30
        model = QuestModel(
            CFG.base_model, 
            num_targets=len(TARGET_COLS),
            pooling_strategy=CFG.pooling_strategy
        )
        model.load_state_dict(torch.load(weight_path, map_location=CFG.device))
        model.to(CFG.device)
        
        preds = inference_fn(loader, model, CFG.device)
        model_preds.append(preds)
        
        del model
        torch.cuda.empty_cache()
        
    # 4. Average Predictions
    avg_preds = np.mean(model_preds, axis=0)
    
    # 5. Evaluation
    pred_df = pd.DataFrame(avg_preds, columns=TARGET_COLS)
    evaluate_metrics(df, pred_df, TARGET_COLS)