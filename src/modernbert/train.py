import os
# Disable torch.compile to avoid C compiler requirement for ModernBERT
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
import gc
import numpy as np
import pandas as pd
import torch
import torch
torch._dynamo.config.suppress_errors = True  # Disable torch.compile() error suppression
import torch.nn as nn

# Disable torch.compile() globally to avoid C compiler requirements
if hasattr(torch, '_compile_threshold'):
    torch._compile_threshold = float('inf')
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm

from data import DebertaDataset, TARGET_COLS
from model import QuestModel

POS_WEIGHT_VALUES = [
    0.9, 1, 1.5, 0.8, 0.8, 0.8, 0.96, 1.1, 1.1, 3, 
    1, 1.1, 2, 3, 3, 2, 1, 2, 1, 2, 
    0.9, 0.75, 0.9, 0.75, 0.75, 0.7, 1, 2.5, 1, 0.75
]

# --- Configuration ---
class CFG:
    model_name = 'answerdotai/ModernBERT-base'  # ModernBERT for better efficiency
    
    pooling_strategy = 'arch1_6groups' 
    
    # 【關鍵開關】
    # True  -> 執行完整的 5-Fold GroupKFold 訓練 (適合最終提交)
    # False -> 只執行一次 Train/Valid 切分 (適合快速 Ablation Study)
    use_kfold = True
    
    max_len = 512
    batch_size = 2        # Reduced for RTX 4060
    accum_steps = 2       # Reduced accumulation steps
    epochs = 10            # Fewer epochs for efficiency
    lr = 1e-5             # Lower LR for stability
    head_lr = 5e-5        # Proportionally lower head LR
    weight_decay = 0.01
    max_grad_norm = 1.0   
    seed = 42
    n_fold = 5            # 只在 use_kfold=True 時生效
    val_size = 0.2        # 只在 use_kfold=False 時生效
    num_workers = 2       # Reduced for memory efficiency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_cols = TARGET_COLS
    num_targets = len(TARGET_COLS)

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Helper: Optimized Rounder ---
class OptimizedRounder:
    def __init__(self):
        self.coef_ = [0.05, 0.95] # 預設初始值

    def _loss_func(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = coef[0]
            elif pred > coef[1]: X_p[i] = coef[1]
        
        try:
            ll = spearmanr(y, X_p).correlation
            if np.isnan(ll): return 0
            return -ll
        except:
            return 0

    def fit(self, X, y):
        from scipy.optimize import minimize
        x0 = [0.05, 0.95]
        opt = minimize(self._loss_func, x0, args=(X, y), method='nelder-mead', tol=1e-6)
        self.coef_ = opt.x

    def predict(self, X, coef):
        X_p = np.copy(X)
        X_p = np.nan_to_num(X_p, nan=0.5)
        
        low, high = coef[0], coef[1]
        X_p = np.clip(X_p, low, high)
        
        if np.unique(X_p).size == 1:
            eps = 1e-6
            max_idx = np.argmax(X)
            X_p[max_idx] += eps
            
        return X_p

# --- Training Helper Functions ---
def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = spearmanr(y_true[:, i], y_pred[:, i]).correlation
            if np.isnan(score):
                scores.append(0.0)
            else:
                scores.append(score)
        except:
            scores.append(0.0)
    return np.mean(scores)

def train_fn(train_loader, model, optimizer, scheduler, epoch, scaler):
    model.train()
    losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
    
    pos_weight = torch.tensor(POS_WEIGHT_VALUES, dtype=torch.float32).to(CFG.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        labels = batch['labels'].to(CFG.device)
        
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(CFG.device)
        
        with torch.amp.autocast('cuda'):
            y_preds = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(y_preds, labels)
            
        loss = loss / CFG.accum_steps
        scaler.scale(loss).backward()
        
        if (step + 1) % CFG.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        losses.append(loss.item() * CFG.accum_steps)
        pbar.set_postfix({'loss': np.mean(losses)})
        
    return np.mean(losses)

def valid_fn(valid_loader, model):
    model.eval()
    preds = []
    valid_labels = []
    pbar = tqdm(valid_loader, desc="Valid")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            labels = batch['labels'].to(CFG.device)
            
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(CFG.device)
            
            with torch.amp.autocast('cuda'):
                y_preds = model(input_ids, attention_mask, token_type_ids)
            
            preds.append(y_preds.sigmoid().cpu().numpy())
            valid_labels.append(labels.cpu().numpy())
            
    return np.concatenate(valid_labels), np.concatenate(preds)

if __name__ == '__main__':
    seed_everything(CFG.seed)    
    print(f"Using Strategy: {CFG.pooling_strategy}")
    print(f"Running Mode: {'K-Fold Cross Validation' if CFG.use_kfold else 'Single Train/Valid Split'}")
    
    if not os.path.exists('./model'):
        os.makedirs('./model')
    
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
    else:
        # Fallback check
        if os.path.exists('train.csv'):
            train = pd.read_csv('train.csv')
        else:
            print("Error: train.csv not found.")
            exit()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    
    # ------------------------------------------------------------------
    # 數據切分邏輯 (Splitting Logic)
    # ------------------------------------------------------------------
    splits = []
    
    if CFG.use_kfold:
        # K-Fold 模式：產生 5 組 (train_idx, val_idx)
        gkf = GroupKFold(n_splits=CFG.n_fold)
        splits = list(gkf.split(train, train[CFG.target_cols], train['question_body']))
    else:
        # Single Run 模式：產生 1 組 (train_idx, val_idx)
        # 為了相容下面的迴圈，我們手動取得索引並包成 list
        all_indices = np.arange(len(train))
        train_idx, val_idx = train_test_split(all_indices, test_size=CFG.val_size, random_state=CFG.seed)
        splits = [(train_idx, val_idx)] # 包成列表，讓迴圈跑一次

    # ------------------------------------------------------------------
    # 訓練迴圈 (相容 K-Fold 與 Single Run)
    # ------------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*30} Fold {fold} {'='*30}")
        
        train_df = train.iloc[train_idx]
        valid_df = train.iloc[val_idx]
        
        train_dataset = DebertaDataset(train_df, tokenizer, max_len=CFG.max_len, is_train=True)
        valid_dataset = DebertaDataset(valid_df, tokenizer, max_len=CFG.max_len, is_train=True)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, 
                                  num_workers=CFG.num_workers, pin_memory=True)
        
        model = QuestModel(
            CFG.model_name, 
            num_targets=CFG.num_targets,
            pooling_strategy=CFG.pooling_strategy
        )
        model.to(CFG.device)
        
        # 分層學習率 (Layer-wise Learning Rate)
        optimizer_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters()], 'lr': CFG.lr},
            {'params': [p for n, p in model.named_parameters() if "backbone" not in n], 'lr': CFG.head_lr}
        ]
        
        optimizer = AdamW(optimizer_parameters, weight_decay=CFG.weight_decay)
        
        num_train_steps = int(len(train_df) / CFG.batch_size / CFG.accum_steps * CFG.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, num_train_steps // 20), num_training_steps=num_train_steps
        )
        
        scaler = GradScaler('cuda')
        best_score = -1
        
        for epoch in range(CFG.epochs):
            avg_loss = train_fn(train_loader, model, optimizer, scheduler, epoch, scaler)
            valid_labels, valid_preds = valid_fn(valid_loader, model)
            
            # 1. Raw Spearman Score
            score = get_score(valid_labels, valid_preds)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Raw Score: {score:.4f}")
            
            # 2. Optimized Score (Post-processing)
            opt_preds = np.copy(valid_preds)
            for i in range(CFG.num_targets):
                opt = OptimizedRounder()
                opt.fit(valid_preds[:, i], valid_labels[:, i])
                opt_preds[:, i] = opt.predict(valid_preds[:, i], opt.coef_)
                
            opt_score = get_score(valid_labels, opt_preds)
            print(f"Epoch {epoch+1} - Optimized Score: {opt_score:.4f} (+{opt_score - score:.4f})")
            
            if score > best_score:
                best_score = score
                
                if CFG.use_kfold:
                    save_name = f"modernbert_fold{fold}_best.pth"
                else:
                    save_name = f"modernbert_single_run_best.pth"
                    
                torch.save(model.state_dict(), f"./model/{save_name}")
                print(f"--> Saved Best Model: {best_score:.4f} ({save_name})")
        
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()