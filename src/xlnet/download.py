from transformers import AutoModel, AutoTokenizer
import os

# Configure model name
model_name = "xlnet-base-cased"

# Save path relative to project root
save_dir = "../../xlnet-base-tokenizer"

# Create the output directory
os.makedirs(save_dir, exist_ok=True)

print(f"Downloading {model_name} model and tokenizer...")

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
print(f"✓ Tokenizer saved to {save_dir}")

# Download and save model
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_dir)
print(f"✓ Model saved to {save_dir}")

print("\nDownload complete!")
print(f"Model artifacts stored at: {os.path.abspath(save_dir)}")
print("\nFor Kaggle, upload this directory as a dataset and load with:")
print("AutoTokenizer.from_pretrained('/kaggle/input/xlnet-base-cased', local_files_only=True)")
