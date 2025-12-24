from transformers import AutoModel, AutoTokenizer
import os

# 設定模型名稱
model_name = "xlnet-base-cased"

# 設定保存路徑（專案根目錄下）
save_dir = "../../xlnet-base-tokenizer"

# 創建保存目錄
os.makedirs(save_dir, exist_ok=True)

print(f"正在下載 {model_name} 模型和分詞器...")

# 下載並保存分詞器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
print(f"✓ 分詞器已保存到 {save_dir}")

# 下載並保存模型
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_dir)
print(f"✓ 模型已保存到 {save_dir}")

print("\n下載完成！")
print(f"模型文件保存在: {os.path.abspath(save_dir)}")
print("\n在 Kaggle 中使用時，請將此目錄上傳為數據集，然後使用:")
print(f"AutoTokenizer.from_pretrained('/kaggle/input/xlnet-base-cased', local_files_only=True)")
