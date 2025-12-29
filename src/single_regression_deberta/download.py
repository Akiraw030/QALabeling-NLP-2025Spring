"""Download the full DeBERTa v3 base model (weights + tokenizer)."""

import os
from transformers import AutoTokenizer, AutoConfig, AutoModel


def download_deberta_v3(model_name: str = "microsoft/deberta-v3-base", output_dir: str = "./deberta-v3-base-full") -> None:
	os.makedirs(output_dir, exist_ok=True)

	print(f"Downloading tokenizer + config for {model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	config = AutoConfig.from_pretrained(model_name)

	print(f"Downloading model weights for {model_name}...")
	model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)

	print(f"Saving files to {output_dir}...")
	tokenizer.save_pretrained(output_dir)
	config.save_pretrained(output_dir)
	model.save_pretrained(output_dir)
	print("Done.")


if __name__ == "__main__":
	download_deberta_v3()