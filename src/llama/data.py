import pandas as pd
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

def modern_preprocess(text):
    """
    Modern text preprocessing:
    1. Convert to string
    2. HTML unescape
    3. Normalize whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = " ".join(text.split()) 
    return text

class DebertaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        
        self.questions = [
            modern_preprocess(t) + " " + modern_preprocess(b) for t, b in zip(df['question_title'].values, df['question_body'].values)
        ]
        self.answers = [modern_preprocess(a) for a in df['answer'].values]
        
        if self.is_train:
            self.targets = df[TARGET_COLS].values
            
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

        # Some tokenizers (e.g., Qwen embeddings) do not provide token_type_ids.
        if 'token_type_ids' in inputs:
            item['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            
        if self.is_train:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
            
        return item