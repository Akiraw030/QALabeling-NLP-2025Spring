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

# ------------------- Configuration -------------------
class CFG:
    # 模型權重資料夾
    model_dir = "./model" 
    
    # 訓練資料路徑 (用來評估)
    data_path = "./data/train.csv"
    
    base_model = "microsoft/deberta-v3-base" 
    pooling_strategy = 'arch1_6groups' 
    max_len = 512
    batch_size = 16
    num_workers = 4 # 根據筆電 CPU 核心數調整
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

# 配合 6-Head Grouping 的排序
GROUP_ORDER_INDICES = [
    3, 4, 5, 16, 17,          # G1
    0, 1, 6, 7, 20,           # G2
    2, 10,                    # G3
    8, 9, 11, 12, 13, 14, 15, 18, 19, # G4
    26, 27,                   # G5
    21, 22, 23, 24, 25, 28, 29 # G6
]
SORTED_TARGET_COLS = [TARGET_COLS[i] for i in GROUP_ORDER_INDICES]

# ------------------- 核心模組 -------------------
def modern_preprocess(text):
    if pd.isna(text): return ""
    text = str(text)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text

class BinaryTargetEncoder:
    def __init__(self, target_cols=SORTED_TARGET_COLS):
        self.target_cols = target_cols
        self.unique_values = {} 
        self.thresholds = {}    
        self.output_slices = {} 
        self.total_output_dim = 0

    def fit(self, df):
        print(f"Fitting Binary Encoder on {len(df)} samples...")
        current_idx = 0
        for col in self.target_cols:
            uniques = sorted(df[col].unique())
            self.unique_values[col] = uniques
            
            if len(uniques) > 1:
                thresh = uniques[:-1]
            else:
                thresh = [uniques[0]] 
                
            self.thresholds[col] = thresh
            n_dims = len(thresh)
            self.output_slices[col] = slice(current_idx, current_idx + n_dims)
            current_idx += n_dims
            
        self.total_output_dim = current_idx
        print(f"Total Binary Output Dimension: {self.total_output_dim}")

    def inverse_transform(self, binary_preds):
        batch_size = binary_preds.shape[0]
        output = np.zeros((batch_size, len(self.target_cols)), dtype=np.float32)
        
        for i, col in enumerate(self.target_cols):
            slc = self.output_slices[col]
            col_preds = binary_preds[:, slc]
            output[:, i] = col_preds.mean(axis=1)
            
        return output

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

# ------------------- 模型定義 (Binary Version) -------------------
class QuestModel(nn.Module):
    def __init__(self, model_name, target_encoder, pooling_strategy='arch1_6groups', dropout_rate=0.1):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        all_cols = target_encoder.target_cols 
        g1_cols = all_cols[0:5]
        g2_cols = all_cols[5:10]
        g3_cols = all_cols[10:12]
        g4_cols = all_cols[12:21]
        g5_cols = all_cols[21:23]
        g6_cols = all_cols[23:30]
        
        self.group_dims = {}
        for g_name, cols in zip(['g1','g2','g3','g4','g5','g6'], [g1_cols, g2_cols, g3_cols, g4_cols, g5_cols, g6_cols]):
            dim = 0
            for c in cols:
                slc = target_encoder.output_slices[c]
                dim += (slc.stop - slc.start)
            self.group_dims[g_name] = dim
            
        self.head_g1 = self._make_head(hidden_size * 3, self.group_dims['g1'], dropout_rate)
        self.head_g2 = self._make_head(hidden_size * 3, self.group_dims['g2'], dropout_rate)
        self.head_g3 = self._make_head(hidden_size * 3, self.group_dims['g3'], dropout_rate)
        self.head_g4 = self._make_head(hidden_size * 3, self.group_dims['g4'], dropout_rate)
        self.head_g5 = self._make_head(hidden_size * 3, self.group_dims['g5'], dropout_rate)
        self.head_g6 = self._make_head(hidden_size * 3, self.group_dims['g6'], dropout_rate)

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

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        
        cls, glob, q, a = self._get_pooling_features(last_hidden_state, attention_mask, token_type_ids)
        feat_pure_q = torch.cat([cls, glob, q], dim=1)
        feat_pure_a = torch.cat([cls, glob, a], dim=1)
        
        out_g1 = self.head_g1(feat_pure_q)
        out_g2 = self.head_g2(feat_pure_q)
        out_g3 = self.head_g3(feat_pure_q)
        out_g4 = self.head_g4(feat_pure_q)
        out_g5 = self.head_g5(feat_pure_a)
        out_g6 = self.head_g6(feat_pure_a)
        
        output = torch.cat([out_g1, out_g2, out_g3, out_g4, out_g5, out_g6], dim=1)
        return output

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
            preds.append(y_preds.sigmoid().cpu().numpy())
            
    return np.concatenate(preds)

def evaluate_metrics(true_df, pred_df, target_cols):
    print("\n" + "="*50)
    print(f"{'Target Column':<40} | {'Spearman':<10}")
    print("-" * 55)
    
    spearman_scores = []
    
    for col in target_cols:
        y_true = true_df[col].values
        y_pred = pred_df[col].values
        
        # 只計算 Spearman
        s_score, _ = spearmanr(y_true, y_pred)
        spearman_scores.append(s_score)
        
        print(f"{col:<40} | {s_score:.4f}")
        
    print("-" * 55)
    print(f"{'AVERAGE SPEARMAN':<40} | {np.mean(spearman_scores):.4f}")
    print("="*50 + "\n")

# ------------------- Main -------------------
if __name__ == '__main__':
    if not os.path.exists(CFG.data_path):
        print(f"Error: Data file not found at {CFG.data_path}")
        sys.exit(1)
        
    print(f"Loading data from {CFG.data_path}...")
    df = pd.read_csv(CFG.data_path)
    print(f"Data shape: {df.shape}")
    
    # 1. Initialize Encoder
    target_encoder = BinaryTargetEncoder(target_cols=SORTED_TARGET_COLS)
    target_encoder.fit(df)
    
    # 2. Prepare DataLoader
    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
    dataset = QuestDataset(df, tokenizer, max_len=CFG.max_len)
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    
    # 3. Find Models
    weight_paths = [os.path.join(CFG.model_dir, f) for f in os.listdir(CFG.model_dir) if f.endswith('.pth')]
    if not weight_paths:
        print(f"No .pth models found in {CFG.model_dir}")
        sys.exit(1)
    print(f"Found {len(weight_paths)} models: {[os.path.basename(p) for p in weight_paths]}")
    
    # 4. Inference loop
    model_preds = []
    
    for weight_path in weight_paths:
        print(f"\nProcessing {os.path.basename(weight_path)}...")
        model = QuestModel(
            CFG.base_model, 
            target_encoder=target_encoder,
            pooling_strategy=CFG.pooling_strategy
        )
        model.load_state_dict(torch.load(weight_path, map_location=CFG.device))
        model.to(CFG.device)
        
        binary_preds = inference_fn(loader, model, CFG.device)
        
        # 還原分數
        decoded_preds = target_encoder.inverse_transform(binary_preds)
        model_preds.append(decoded_preds)
        
        del model
        torch.cuda.empty_cache()
        
    # 5. Average Predictions
    avg_preds = np.mean(model_preds, axis=0)
    
    # 6. Evaluation
    pred_df = pd.DataFrame(avg_preds, columns=SORTED_TARGET_COLS)
    evaluate_metrics(df, pred_df, TARGET_COLS)