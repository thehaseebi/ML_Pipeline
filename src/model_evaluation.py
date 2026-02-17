import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded successfully from {file_path}. Shape: {df.shape}')
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parser the csv file: %s', e)
    except Exception as e:
        logger.error('Unexpected error occured while loading data: %s', e)
        raise

def evaluate_model(random_forest_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = random_forest_model.predict(X_test)
        y_proba = random_forest_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        logger.debug('Model evaluation completed. Accuracy: %.4f, Precision: %.4f, Recall: %.4f, ROC AUC: %.4f', 
                     accuracy, precision, recall, roc_auc)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
    except Exception as e:
        logger.error('Unexpected error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a json file."""
    try:
        os.makedirs("reports", exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug('Evaluation metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error while saving metrics: %s', e)
        raise

def main():
    """Main function to run the model evaluation."""
    try:
        test_data = load_data('data/processed/test_tfidf.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        with open('models/random_forest_model.pkl', 'rb') as f:
            random_forest_model = pickle.load(f)

        metrics = evaluate_model(random_forest_model, X_test, y_test)
        save_metrics(metrics, 'reports/evaluation_metrics.json')
    except Exception as e:
        logger.error('Unexpected error in main function: %s', e)
        print('Unexpected error in main function:', e)


if __name__ == "__main__":
    main()