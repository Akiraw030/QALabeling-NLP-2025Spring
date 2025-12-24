#!/usr/bin/env python3
"""
Download script for google/electra-base-discriminator
Downloads the model and tokenizer for offline use in Kaggle
"""

import os
from transformers import AutoModel, AutoTokenizer

# Model name
model_name = "google/electra-base-discriminator"

# Download directory
download_dir = "./electra-base-discriminator"

print(f"Downloading {model_name}...")
print(f"Saving to: {download_dir}")

# Create directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Download model
print("\n[1/2] Downloading model...")
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(download_dir)
print(f"✓ Model saved to {download_dir}")

# Download tokenizer
print("\n[2/2] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(download_dir)
print(f"✓ Tokenizer saved to {download_dir}")

print("\n" + "="*60)
print("✓ Download complete!")
print(f"Now you can upload '{download_dir}' to Kaggle")
print("and load it with: AutoModel.from_pretrained('./electra-base-discriminator')")
print("="*60)
