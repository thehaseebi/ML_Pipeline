import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_model(file_path: str) -> RandomForestClassifier:
    """Train a Random Forest model."""
    try:
        df = pd.read_csv(file_path)
        X = df.drop('label', axis=1).values
        y = df['label'].values

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        logger.debug("model trained successfully with data from %s", file_path)
        return model
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Error training model: %s", e)
        raise

def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("model saved successfully to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise

def main():
    try:
        os.makedirs("models", exist_ok=True)
        model = train_model('data/processed/train_tfidf.csv')
        save_model(model, 'models/random_forest_model.pkl')
    except Exception as e:
        logger.error("Error in main function: %s", e)
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
