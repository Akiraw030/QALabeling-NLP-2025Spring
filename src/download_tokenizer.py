import os
from transformers import AutoTokenizer, AutoConfig

model_name = "microsoft/deberta-v3-base"
output_dir = "./deberta-v3-base-tokenizer"

# 1. Download Tokenizer and Config
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# 2. Save them to a local folder
tokenizer.save_pretrained(output_dir)
config.save_pretrained(output_dir)

print(f"Files saved to {output_dir}")