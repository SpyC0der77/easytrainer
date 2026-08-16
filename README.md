# easytrainer

Need to train a model? Find your task here, change the parameters, and run it. It's that simple.

**Tasks:**
- [`token-class`](token-class/) — label each token (example: emotion spans)

Each task is in its own folder, each following this structure:

- `config.json`: Contains all configuration parameters for your project, such as model selection, training hyperparameters, data paths, and other customizable options.
- `preprocess.py`: For all data preparation steps, including dataset manipulation, cleaning, feature engineering, and schema migration. Prepare your data for training in this file.
- `train.py`: Dedicated solely to model training. It imports its input data from `preprocess.py` and should not contain any data preparation or inference code. Loads necessary settings from `config.json` to configure, train, and save your model.
- `infer.py`: Handles inference and prediction. Load trained models and generate predictions in this file, also taking required configurations from `config.json`.
- `README.md`: Explains the tasks purpose, folder organization, and how to use the code.

**Recommended Structure:**
```
.
├── config.json    # All configuration parameters and settings
├── preprocess.py  # Data loading and preprocessing
├── train.py       # Model training (uses preprocess.py, loads config)
├── infer.py       # Inference routines (loads config)
├── README.md      # Documentation
```

By keeping each file focused on its own role and storing all configuration in a central `config.json`, the codebase stays clean, flexible, and easy to understand.