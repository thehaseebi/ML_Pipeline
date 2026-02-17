import pandas as pd
import os
from sklearn.model_selection import train_test_split 
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('DataIngestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data."""
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
        logger.debug('Data preprocessing completed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test dataset."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving data: %s', e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df = load_data(file_path=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=76)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Unexpected error in the data ingestion process: %s', e)
        print(f'Error: {e}')

if __name__ == "__main__":
    main()
