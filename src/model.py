import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class QuestModel(nn.Module):
    def __init__(self, model_name, num_targets, pooling_strategy='arch1', dropout_rate=0.1):
        """
        Args:
            model_name (str): HuggingFace model name (e.g., 'microsoft/deberta-v3-base')
            num_targets (int): Number of outputs (30)
            pooling_strategy (str): Choose between:
                - 'mean': Original simple mean pooling.
                - 'arch1': The top strategy in the image (5 specific features concatenated).
                - 'arch2': The bottom strategy in the image (last 4 layers CLS concatenated).
            dropout_rate (float): Dropout probability for the regressor head.
        """
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.config = AutoConfig.from_pretrained(model_name)
        
        # IMPORTANT for Arch2: We need hidden states from all layers
        if pooling_strategy == 'arch2':
            self.config.update({'output_hidden_states': True})
            
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        # Define Classification Heads based on strategy
        if self.pooling_strategy == 'mean':
            self.fc = nn.Linear(hidden_size, num_targets)
            self._init_weights(self.fc)
            
        elif self.pooling_strategy == 'arch1':
            # Strategy: Concat 5 features (CLS, Last, GlobalAvg, Q_Avg, A_Avg)
            # Input dim = 5 * hidden_size. 
            # The image shows an intermediate linear layer.
            self.intermediate_layer = nn.Sequential(
                nn.Linear(hidden_size * 5, hidden_size),
                nn.Tanh(), # Standard activation between linear layers in BERT heads
                nn.Dropout(dropout_rate)
            )
            self.fc = nn.Linear(hidden_size, num_targets)
            self._init_weights(self.intermediate_layer[0])
            self._init_weights(self.fc)
            
        elif self.pooling_strategy == 'arch2':
            # Strategy: Concat [CLS] from last 4 layers
            # Input dim = 4 * hidden_size
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * 4, num_targets)
            )
            self._init_weights(self.fc[1])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    # ================= Helper: Masked Mean Pooling =================
    def _masked_mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # ================= Strategy Implementation: Arch 1 =================
    def _pool_arch1(self, last_hidden_state, attention_mask, token_type_ids):
        batch_size = last_hidden_state.size(0)
        
        # 1. First token ([CLS])
        cls_token = last_hidden_state[:, 0, :]
        
        # 2. Last token ([SEP])
        # Find the index of the last non-padding token for each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1
        # Select those tokens
        last_token = last_hidden_state[torch.arange(batch_size), last_token_indices, :]
        
        # 3. Global Average Pooling
        global_avg = self._masked_mean_pooling(last_hidden_state, attention_mask)
        
        # Need token_type_ids for Q vs A averaging
        if token_type_ids is None:
            # Fallback if not provided, though performance will degrade
            q_avg = global_avg
            a_avg = global_avg
        else:
            # 4. Question Average Pooling (where token_type_ids is 0)
            # Create mask that is 1 only where it's real data AND it's part of sentence 0
            q_mask = attention_mask * (1 - token_type_ids)
            q_avg = self._masked_mean_pooling(last_hidden_state, q_mask)

            # 5. Answer Average Pooling (where token_type_ids is 1)
            a_mask = attention_mask * token_type_ids
            a_avg = self._masked_mean_pooling(last_hidden_state, a_mask)
            
        # Concatenate all 5 features along dimension 1
        pooled_output = torch.cat([cls_token, last_token, global_avg, q_avg, a_avg], dim=1)
        return pooled_output

    # ================= Strategy Implementation: Arch 2 =================
    def _pool_arch2(self, all_hidden_states):
        # Take the last 4 layers
        last_4_layers = all_hidden_states[-4:]
        
        # Extract [CLS] token (index 0) from each of these layers
        # Each layer shape: (batch, seq_len, hidden) -> [CLS] shape: (batch, hidden)
        cls_embeddings = [layer_output[:, 0, :] for layer_output in last_4_layers]
        
        # Concatenate along dimension 1
        # Final shape: (batch, 4 * hidden_size)
        pooled_output = torch.cat(cls_embeddings, dim=1)
        return pooled_output

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Note: We accept token_type_ids now!
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        
        if self.pooling_strategy == 'mean':
            feature = self._masked_mean_pooling(last_hidden_state, attention_mask)
            output = self.fc(feature)
            
        elif self.pooling_strategy == 'arch1':
            feature = self._pool_arch1(last_hidden_state, attention_mask, token_type_ids)
            # Arch1 has an intermediate layer before the final head
            projected_feature = self.intermediate_layer(feature)
            output = self.fc(projected_feature)
            
        elif self.pooling_strategy == 'arch2':
            # For arch2, we need all hidden states tuple
            feature = self._pool_arch2(outputs.hidden_states)
            output = self.fc(feature)
            
        return output