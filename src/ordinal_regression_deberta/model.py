import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class QuestModel(nn.Module):
    def __init__(self, model_name, target_encoder, pooling_strategy='arch1_6groups', dropout_rate=0.1):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name)
        
        if pooling_strategy == 'arch2':
            self.config.update({'output_hidden_states': True})
            
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        # 1. 計算每個 Group 需要的輸出維度
        all_cols = target_encoder.target_cols 
        
        # 根據 SORTED_TARGET_COLS 的順序切分
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
            
        # 2. 定義 Heads
        if self.pooling_strategy == 'arch1_6groups':
            self.head_g1 = self._make_head(hidden_size * 3, self.group_dims['g1'], dropout_rate)
            self.head_g2 = self._make_head(hidden_size * 3, self.group_dims['g2'], dropout_rate)
            self.head_g3 = self._make_head(hidden_size * 3, self.group_dims['g3'], dropout_rate)
            self.head_g4 = self._make_head(hidden_size * 3, self.group_dims['g4'], dropout_rate)
            self.head_g5 = self._make_head(hidden_size * 3, self.group_dims['g5'], dropout_rate)
            self.head_g6 = self._make_head(hidden_size * 3, self.group_dims['g6'], dropout_rate)

    def _make_head(self, input_dim, output_dim, dropout_rate):
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
            
            # 直接 Concat (Dataset 已經按照這個順序排好 Label 了)
            output = torch.cat([out_g1, out_g2, out_g3, out_g4, out_g5, out_g6], dim=1)
            return output
        
        return None