import os
import logging 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('DataPreprocessing')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text: str) -> str:
    """Transform text by removing punctuation, stopwords and applying stemming."""
    try:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        porter_stemmer = PorterStemmer()
        stemmed_tokens = [porter_stemmer.stem(word) for word in tokens]
        transformed_text = ' '.join(stemmed_tokens)
        return transformed_text
    except Exception as e:
        logger.error('Error during text transformation: %s', e)
        raise

def preprocess_data(df, text_column='text', target_column='target'):
    """Preprocess the df by transforming text and encoding target columns."""
    try:
        logger.debug('Starting data preprocessing.')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded successfully.')
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed.')
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """Main function to load raw, preprocess it, and save the processed data."""
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully.')

        train_preprocessed_data = preprocess_data(train_data, text_column, target_column)
        test_preprocessed_data = preprocess_data(test_data, text_column, target_column)

        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_preprocessed_data.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)

        logger.debug('Preprocessed data saved successfully to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('Empty data error: %s', e)
    except Exception as e:
        logger.error('Unexpected error in main function: %s', e)
        print("Failed to complete data transformation process: %s",e)

if __name__ == "__main__":    
    main()