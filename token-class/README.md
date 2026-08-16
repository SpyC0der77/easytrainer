# token-class

Assign a label to every token. Swap the example in `preprocess.py` and `config.json` for any token-classification dataset.

The bundled example tags emotion spans as BIO (`B-Joy`, `I-Sadness`, `O`) on GoEmotions.

```
.
├── config.json    # model, data, and training settings
├── preprocess.py  # load data and build per-token labels
├── train.py       # train and save the best checkpoint
├── infer.py       # run the saved model on validation examples
├── README.md
```

Edit `config.json`, then from this folder:

```
python train.py
python infer.py
```
