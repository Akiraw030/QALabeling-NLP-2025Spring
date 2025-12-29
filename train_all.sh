echo "Training ELECTRA model..."
python ./src/electra/train.py

echo "Training ModernBERT model..."
python ./src/modernbert/train.py

echo "Training Qwen model..."
python ./src/qwen/train.py

echo "Training Single Regression DeBERTa model..."
python ./src/single_regression_deberta/train.py

echo "Training XLNet model..."
python ./src/xlnet/train.py

echo "Training Llama model..."
python ./src/llama/train.py

echo "All training completed!"