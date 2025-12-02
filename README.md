## How to run

```
pip install -r requirements.txt
put csv training data file inside ./data
python ./src/train.py
put .pth inside ./model to kaggle model input
python ./util/download_tokenizer.py
put deberta-v3-base-tokenizer to kaggle dataset input
submit kaggle.ipynb under the competition 
```