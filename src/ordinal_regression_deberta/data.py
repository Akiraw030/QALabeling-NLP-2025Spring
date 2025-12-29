import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import html

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

# Define reordered indices to match 6-head grouping
GROUP_ORDER_INDICES = [
    3, 4, 5, 16, 17,          
    0, 1, 6, 7, 20,           
    2, 10,                    
    8, 9, 11, 12, 13, 14, 15, 18, 19, 
    26, 27,                  
    21, 22, 23, 24, 25, 28, 29 
]
SORTED_TARGET_COLS = [TARGET_COLS[i] for i in GROUP_ORDER_INDICES]

def modern_preprocess(text):
    if pd.isna(text): return ""
    text = str(text)
    text = html.unescape(text)
    text = " ".join(text.split()) 
    return text

class BinaryTargetEncoder:
    """
    Encode continuous scores (e.g., 0.33) into binary threshold vectors (e.g., [1, 1, 0]),
    and decode predicted probabilities back to continuous scores.
    """
    def __init__(self, target_cols=SORTED_TARGET_COLS):
        self.target_cols = target_cols
        self.unique_values = {} 
        self.thresholds = {}    
        self.output_slices = {} 
        self.total_output_dim = 0

    def fit(self, df):
        """Scan data to build encoding rules"""
        current_idx = 0
        for col in self.target_cols:
            uniques = sorted(df[col].unique())
            self.unique_values[col] = uniques
            
            # Thresholds are all unique values except the max
            if len(uniques) > 1:
                thresh = uniques[:-1]
            else:
                thresh = [uniques[0]] 
                
            self.thresholds[col] = thresh
            n_dims = len(thresh)
            self.output_slices[col] = slice(current_idx, current_idx + n_dims)
            current_idx += n_dims
            
        self.total_output_dim = current_idx
        print(f"Binary Encoder Fitted. Total Output Dimension: {self.total_output_dim}")

    def transform(self, targets):
        """(Batch, 30) -> (Batch, Total_Dim)"""
        batch_size = targets.shape[0]
        output = np.zeros((batch_size, self.total_output_dim), dtype=np.float32)
        
        for i, col in enumerate(self.target_cols):
            vals = targets[:, i]
            slc = self.output_slices[col]
            threshs = self.thresholds[col]
            
            for j, t in enumerate(threshs):
                # Core logic: if value > threshold, set bit to 1
                output[:, slc.start + j] = (vals > t + 1e-5).astype(np.float32)
        return output

    def inverse_transform(self, binary_preds):
        """(Batch, Total_Dim) -> (Batch, 30)"""
        batch_size = binary_preds.shape[0]
        output = np.zeros((batch_size, len(self.target_cols)), dtype=np.float32)
        
        for i, col in enumerate(self.target_cols):
            slc = self.output_slices[col]
            col_preds = binary_preds[:, slc]
            # Decode by mean (expected value approximation)
            output[:, i] = col_preds.mean(axis=1)
        return output

class DebertaDataset(Dataset):
    def __init__(self, df, tokenizer, target_encoder=None, max_len=512, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.target_encoder = target_encoder
        
        self.questions = [
            modern_preprocess(t) + " " + modern_preprocess(b) 
            for t, b in zip(df['question_title'].values, df['question_body'].values)
        ]
        self.answers = [modern_preprocess(a) for a in df['answer'].values]
        
        if self.is_train:
            # Use reordered columns
            raw_targets = df[SORTED_TARGET_COLS].values
            if self.target_encoder is not None:
                self.targets = self.target_encoder.transform(raw_targets)
            else:
                self.targets = raw_targets
            
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
            
        if self.is_train:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
            
        return item