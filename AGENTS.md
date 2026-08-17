Read about this project in the README.md. Other instructions you should follow that are not in the README are as follows:

All preprocessing (dataset manipulation, schema migrationo, etc.) should go in a preprocess.py file. The train.py should ONLY do the training and should import the dataset from preprocess.py. Evaluation metrics and scoring a saved model should go in an evaluate.py file. Inference should go in a infer.py file.

This is a quick-start kit, not production software. Prefer short, readable scripts over defensive error handling, logging filters, key-alias shims, or version-compat wrappers. Leave expected Hugging Face warnings alone unless they break the happy-path train/eval loop. 