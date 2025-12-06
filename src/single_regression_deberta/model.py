import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class QuestModel(nn.Module):
    def __init__(self, model_name, num_targets, pooling_strategy='arch1', dropout_rate=0.1):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name)
        
        if pooling_strategy == 'arch2':
            self.config.update({'output_hidden_states': True})
            
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        # -----------------------------------------------------------------------
        # 定義 6-Head 分群索引 (QA Splitted Strategy)
        # -----------------------------------------------------------------------
        # Question Groups (G1-G4)
        # [Group 1]: Fact, Instructions, Procedure
        self.idx_g1 = [3, 4, 5, 16, 17]          
        # [Group 2]: Quality, Intent, Interest
        self.idx_g2 = [0, 1, 6, 7, 20]           
        # [Group 3]: Conversational, Opinion
        self.idx_g3 = [2, 10]                    
        # [Group 4]: Type, Multi-intent, Reason, Spelling
        self.idx_g4 = [8, 9, 11, 12, 13, 14, 15, 18, 19] 
        
        # Answer Groups (G5-G6)
        # [Group 5]: Answer Instructions/Procedure
        self.idx_g5 = [26, 27]                   
        # [Group 6]: Answer Quality, Helpful, Reason
        self.idx_g6 = [21, 22, 23, 24, 25, 28, 29] 
        
        # -----------------------------------------------------------------------
        # 定義 Heads
        # -----------------------------------------------------------------------
        if self.pooling_strategy == 'mean':
            self.fc = nn.Linear(hidden_size, num_targets)
            self._init_weights(self.fc)
            
        elif self.pooling_strategy == 'arch1':
            # 舊版 2-Head 邏輯
            self.q_head = self._make_head(hidden_size * 3, 21, dropout_rate)
            self.a_head = self._make_head(hidden_size * 3, 9, dropout_rate)
            
        elif self.pooling_strategy == 'arch1_6groups':
            # --- 6-Head 策略 (QA Splitted) ---
            # 所有的 Head 輸入都是 3 * hidden (因為只用純 Q 特徵 或 純 A 特徵)
            
            # Question Heads
            self.head_g1 = self._make_head(hidden_size * 3, len(self.idx_g1), dropout_rate)
            self.head_g2 = self._make_head(hidden_size * 3, len(self.idx_g2), dropout_rate)
            self.head_g3 = self._make_head(hidden_size * 3, len(self.idx_g3), dropout_rate)
            self.head_g4 = self._make_head(hidden_size * 3, len(self.idx_g4), dropout_rate)
            
            # Answer Heads
            self.head_g5 = self._make_head(hidden_size * 3, len(self.idx_g5), dropout_rate)
            self.head_g6 = self._make_head(hidden_size * 3, len(self.idx_g6), dropout_rate)

        elif self.pooling_strategy == 'arch2':
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * 4, num_targets)
            )
            self._init_weights(self.fc[1])

    def _make_head(self, input_dim, output_dim, dropout_rate):
        """建立標準的 MLP Head"""
        head = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, output_dim)
        )
        self._init_weights(head[0])
        self._init_weights(head[3])
        return head

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _masked_mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_pooling_features(self, last_hidden_state, attention_mask, token_type_ids):
        """提取基礎特徵：CLS, Global, Q_Avg, A_Avg"""
        cls_token = last_hidden_state[:, 0, :]
        global_avg = self._masked_mean_pooling(last_hidden_state, attention_mask)
        
        if token_type_ids is None:
            q_avg = global_avg
            a_avg = global_avg
        else:
            q_mask = attention_mask * (1 - token_type_ids)
            q_avg = self._masked_mean_pooling(last_hidden_state, q_mask)
            a_mask = attention_mask * token_type_ids
            a_avg = self._masked_mean_pooling(last_hidden_state, a_mask)
            
        return cls_token, global_avg, q_avg, a_avg

    def _pool_arch2(self, all_hidden_states):
        last_4_layers = all_hidden_states[-4:]
        cls_embeddings = [layer[:, 0, :] for layer in last_4_layers]
        return torch.cat(cls_embeddings, dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        
        if self.pooling_strategy == 'mean':
            feature = self._masked_mean_pooling(last_hidden_state, attention_mask)
            output = self.fc(feature)
            
        elif self.pooling_strategy == 'arch1':
            # 舊的 2-Head 邏輯
            cls, glob, q, a = self._get_pooling_features(last_hidden_state, attention_mask, token_type_ids)
            q_feat = torch.cat([cls, glob, q], dim=1)
            a_feat = torch.cat([cls, glob, a], dim=1)
            output = torch.cat([self.q_head(q_feat), self.a_head(a_feat)], dim=1)
            
        elif self.pooling_strategy == 'arch1_6groups':
            # --- 新的 6-Head 邏輯 ---
            cls, glob, q, a = self._get_pooling_features(last_hidden_state, attention_mask, token_type_ids)
            
            # 1. 準備特徵 (3x Hidden)
            # Question 特徵：[CLS, Global, Q_Avg] (完全不含 A)
            feat_pure_q = torch.cat([cls, glob, q], dim=1)
            # Answer 特徵：[CLS, Global, A_Avg] (完全不含 Q)
            feat_pure_a = torch.cat([cls, glob, a], dim=1)
            
            # 2. 通過各個 Head
            # Q Groups 使用 Q 特徵
            out_g1 = self.head_g1(feat_pure_q)
            out_g2 = self.head_g2(feat_pure_q)
            out_g3 = self.head_g3(feat_pure_q)
            out_g4 = self.head_g4(feat_pure_q)
            
            # A Groups 使用 A 特徵
            out_g5 = self.head_g5(feat_pure_a)
            out_g6 = self.head_g6(feat_pure_a)
            
            # 3. 輸出重組 (Re-assemble)
            batch_size = input_ids.size(0)
            
            # 【關鍵】確保 dtype 一致性 (for mixed precision)
            output = torch.zeros(batch_size, 30, dtype=out_g1.dtype, device=input_ids.device)
            
            # 依照索引填回去
            output[:, self.idx_g1] = out_g1
            output[:, self.idx_g2] = out_g2
            output[:, self.idx_g3] = out_g3
            output[:, self.idx_g4] = out_g4
            output[:, self.idx_g5] = out_g5
            output[:, self.idx_g6] = out_g6
            
        elif self.pooling_strategy == 'arch2':
            feature = self._pool_arch2(outputs.hidden_states)
            output = self.fc(feature)
            
        return output