# easytrainer

Need to train a model? Find your task here, change the parameters, and run it. It's that simple.

`progress.py` at the repo root is shared by every task. It keeps Hugging Face / tqdm logs readable on Kaggle and in notebooks when the output width changes.

**Tasks:**
- [`token-class`](token-class/) — label each token (example: emotion spans)

Each task is in its own folder, each following this structure:

- `config.json`: Contains all configuration parameters for your project, such as model selection, training hyperparameters, data paths, and other customizable options.
- `preprocess.py`: For all data preparation steps, including dataset manipulation, cleaning, feature engineering, and schema migration. Prepare your data for training in this file.
- `train.py`: Dedicated solely to model training. It imports its input data from `preprocess.py` and should not contain any data preparation, evaluation, or inference code. Loads necessary settings from `config.json` to configure, train, and save your model.
- `evaluate.py`: Scores a saved model on the validation split. Holds the task metrics so `train.py` can reuse them during training, and can be run on its own after training.
- `infer.py`: Handles inference and prediction. Load trained models and generate predictions in this file, also taking required configurations from `config.json`.
- `README.md`: Explains the tasks purpose, folder organization, and how to use the code.

**Folder Structure:**
```
.
├── config.json    # All configuration parameters and settings
├── preprocess.py  # Data loading and preprocessing
├── train.py       # Model training (uses preprocess.py, loads config)
├── evaluate.py    # Validation metrics (loads the saved model)
├── infer.py       # Inference routines (loads config)
├── README.md      # Documentation
```

By keeping each file focused on its own role and storing all configuration in a central `config.json`, the codebase stays clean, flexible, and easy to understand.
