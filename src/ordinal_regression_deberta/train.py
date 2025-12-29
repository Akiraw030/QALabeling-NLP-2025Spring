import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm

from data import DebertaDataset, BinaryTargetEncoder, SORTED_TARGET_COLS
from model import QuestModel

# --- Configuration ---
class CFG:
    model_name = 'microsoft/deberta-v3-base'
    pooling_strategy = 'arch1_6groups' 
    use_kfold = False     # True: 5-Fold, False: Single Run
    
    max_len = 512
    batch_size = 4        
    accum_steps = 4       
    epochs = 5            
    lr = 2e-5             
    head_lr = 1e-4        
    weight_decay = 0.01
    max_grad_norm = 1.0   
    seed = 42
    n_fold = 5            
    val_size = 0.2        
    num_workers = 2       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use reordered target list
    target_cols = SORTED_TARGET_COLS

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Training Helper Functions ---
def get_score(y_true, y_pred):
    """
    y_true: (Batch, 30) - original scores
    y_pred: (Batch, 30) - predictions restored to score space
    """
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = spearmanr(y_true[:, i], y_pred[:, i]).correlation
            if np.isnan(score): scores.append(0.0)
            else: scores.append(score)
        except:
            scores.append(0.0)
    return np.mean(scores)

def train_fn(train_loader, model, optimizer, scheduler, epoch, scaler):
    model.train()
    losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
    
    criterion = nn.BCEWithLogitsLoss()
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        labels = batch['labels'].to(CFG.device) # Flattened label vector
        
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(CFG.device)
        
        with autocast():
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

def valid_fn(valid_loader, model, encoder):
    model.eval()
    binary_preds = [] 
    
    pbar = tqdm(valid_loader, desc="Valid")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(CFG.device)
            
            with autocast():
                y_preds = model(input_ids, attention_mask, token_type_ids)
            
            # Convert logits to probabilities
            binary_preds.append(y_preds.sigmoid().cpu().numpy())
            
    binary_preds = np.concatenate(binary_preds)
    
    # Restore to 30 scores
    decoded_preds = encoder.inverse_transform(binary_preds)
    
    return decoded_preds

# --- Main Execution ---
if __name__ == '__main__':
    seed_everything(CFG.seed)    
    print(f"Using Strategy: {CFG.pooling_strategy} (Binary Encoded)")
    print(f"Running Mode: {'K-Fold' if CFG.use_kfold else 'Single Run'}")
    
    if not os.path.exists('./model'):
        os.makedirs('./model')
    
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
    else:
        if os.path.exists('train.csv'):
            train = pd.read_csv('train.csv')
        else:
            print("Error: train.csv not found.")
            exit()

    # 1. Fit Binary Encoder
    print("Fitting Binary Target Encoder...")
    # Use sorted columns so slice order stays consistent
    target_encoder = BinaryTargetEncoder(target_cols=CFG.target_cols)
    target_encoder.fit(train)
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    
    # Splitting Logic
    splits = []
    if CFG.use_kfold:
        gkf = GroupKFold(n_splits=CFG.n_fold)
        splits = list(gkf.split(train, train[CFG.target_cols], train['question_body']))
    else:
        all_indices = np.arange(len(train))
        train_idx, val_idx = train_test_split(all_indices, test_size=CFG.val_size, random_state=CFG.seed)
        splits = [(train_idx, val_idx)]

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*30} Fold {fold} {'='*30}")
        
        train_df = train.iloc[train_idx]
        valid_df = train.iloc[val_idx]
        
        # Collect true 30-d labels for validation
        valid_true_labels = valid_df[CFG.target_cols].values
        
        # Dataset uses encoder for binary targets
        train_dataset = DebertaDataset(train_df, tokenizer, target_encoder, max_len=CFG.max_len, is_train=True)
        valid_dataset = DebertaDataset(valid_df, tokenizer, target_encoder, max_len=CFG.max_len, is_train=True)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, 
                                  num_workers=CFG.num_workers, pin_memory=True)
        
        # Model consumes the encoder slices
        model = QuestModel(
            CFG.model_name, 
            target_encoder=target_encoder,
            pooling_strategy=CFG.pooling_strategy
        )
        model.to(CFG.device)
        
        # Layer-wise LR
        optimizer_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters()], 'lr': CFG.lr},
            {'params': [p for n, p in model.named_parameters() if "backbone" not in n], 'lr': CFG.head_lr}
        ]
        
        optimizer = AdamW(optimizer_parameters, weight_decay=CFG.weight_decay)
        
        num_train_steps = int(len(train_df) / CFG.batch_size / CFG.accum_steps * CFG.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_train_steps * 0.1, num_training_steps=num_train_steps
        )
        
        scaler = GradScaler()
        best_score = -1
        
        for epoch in range(CFG.epochs):
            avg_loss = train_fn(train_loader, model, optimizer, scheduler, epoch, scaler)
            
            # Validation restores to 30-dim scores
            valid_preds = valid_fn(valid_loader, model, target_encoder)
            
            score = get_score(valid_true_labels, valid_preds)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Spearman Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                save_name = f"deberta_v3_fold{fold}_best.pth" if CFG.use_kfold else "deberta_v3_single_run_best.pth"
                torch.save(model.state_dict(), f"./model/{save_name}")
                print(f"--> Saved Best Model: {best_score:.4f}")
        
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()