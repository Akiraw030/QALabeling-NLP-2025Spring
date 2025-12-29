import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class QuestModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_targets,
        pooling_strategy='arch1',
        dropout_rate=0.1,
        freeze_backbone=True,
        trust_remote_code=False
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        if pooling_strategy == 'arch2':
            self.config.update({'output_hidden_states': True})
            
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16  # Use bfloat16 (no GradScaler needed)
        )

        # Optionally freeze the embedding/backbone to train heads only.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Enable gradient checkpointing to save memory during full finetuning
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, 'enable_input_require_grads'):
                self.backbone.enable_input_require_grads()
        hidden_size = self.config.hidden_size
        
        # -----------------------------------------------------------------------
        # Define 6-Head group indices (QA split strategy)
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
        # Define heads
        # -----------------------------------------------------------------------
        if self.pooling_strategy == 'mean':
            self.fc = nn.Linear(hidden_size, num_targets)
            self._init_weights(self.fc)
            
        elif self.pooling_strategy == 'arch1':
            # Legacy 2-Head logic
            self.q_head = self._make_head(hidden_size * 3, 21, dropout_rate)
            self.a_head = self._make_head(hidden_size * 3, 9, dropout_rate)
            
        elif self.pooling_strategy == 'arch1_6groups':
            # --- 6-Head strategy (QA split) ---
            # All heads use 3 * hidden inputs (pure Q or pure A features)
            
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
        """Build a standard MLP head"""
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

    def _last_token_pool(self, last_hidden_state, attention_mask):
        """Extract last non-padding token (works for both left/right padding)"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    
    def _masked_mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_pooling_features(self, last_hidden_state, attention_mask, token_type_ids):
        """Extract base features: Llama uses mean pooling for better representation"""
        # Use mean pooling as primary representation
        global_avg = self._masked_mean_pooling(last_hidden_state, attention_mask)
        last_token = self._last_token_pool(last_hidden_state, attention_mask)
        
        # Llama doesn't have token_type_ids, use shared features for Q/A
        if token_type_ids is None:
            # Use mean pooling for both Q and A representations
            q_repr = global_avg
            a_repr = global_avg
        else:
            # Fallback for models with token_type_ids
            q_mask = attention_mask * (1 - token_type_ids)
            q_repr = self._masked_mean_pooling(last_hidden_state, q_mask)
            a_mask = attention_mask * token_type_ids
            a_repr = self._masked_mean_pooling(last_hidden_state, a_mask)
            
        return global_avg, last_token, q_repr, a_repr

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
            # Legacy 2-Head logic
            glob, last_tok, q, a = self._get_pooling_features(last_hidden_state, attention_mask, token_type_ids)
            q_feat = torch.cat([glob, last_tok, q], dim=1)
            a_feat = torch.cat([glob, last_tok, a], dim=1)
            output = torch.cat([self.q_head(q_feat), self.a_head(a_feat)], dim=1)
            
        elif self.pooling_strategy == 'arch1_6groups':
            # --- 6-Head with Llama mean pooling ---
            glob, last_tok, q, a = self._get_pooling_features(last_hidden_state, attention_mask, token_type_ids)
            
            # Llama has no token_type_ids, use shared features (mean + context)
            feat_shared = torch.cat([glob, last_tok, q], dim=1)
            
            # 2. Pass through each head
            # Q groups use Q features
            out_g1 = self.head_g1(feat_shared)
            out_g2 = self.head_g2(feat_shared)
            out_g3 = self.head_g3(feat_shared)
            out_g4 = self.head_g4(feat_shared)
            
            # A groups use A features
            out_g5 = self.head_g5(feat_shared)
            out_g6 = self.head_g6(feat_shared)
            
            # 3. Reassemble output
            batch_size = input_ids.size(0)
            
            # Ensure dtype consistency (mixed precision)
            output = torch.zeros(batch_size, 30, dtype=out_g1.dtype, device=input_ids.device)
            
            # Assign by indices
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