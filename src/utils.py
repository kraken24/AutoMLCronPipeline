from typing import Dict
import yaml
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 as pg

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

import logging
logger = logging.getLogger()


@dataclass(frozen=True)
class TrainingParameters:
    target_col: str = 'economic_deal'
    model_directory: str = './trained_models/'
    model_name: str = 'trained_classifier.joblib'
    accuracy_threshold: float = 0.95
    validation_set_size: float = 0.20
    schema: str = 'schema'
    table: str = 'table'
    time_interval_in_days: int = 20

    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return TrainingParameters(**params)


def query_training_data(db_params) -> pd.DataFrame:
    """Downloads data from the last 20 days from a specified PostgreSQL table.
    The required parameters are defined in db_params dataclass.

    Args:
        db_params (dict): Database parameters such as schema, table etc.

    Returns:
        DataFrame: Pandas DataFrame containing the data from the last 20 days.
    """
    try:
        conn = pg.connect("dbname=test user=postgres password=secret")
        query = f"""
            SELECT *
            FROM {db_params.schema}.{db_params.table}
            WHERE date BETWEEN CURDATE() - INTERVAL {db_params.time_interval_in_days} DAY AND CURDATE()
            """

        return pd.read_sql_query(query, conn)

    except Exception as e:
        print(f'Error while getting data from databank: {e}')
    finally:
        if conn is not None:
            conn.close()


def load_sample_data(samples: int = 10000) -> pd.DataFrame:
    """Function to load sample training data. Normally this function
    would be replaced with query_training-data function which dynamically
    loads the latest data from the database.

    Args:
        samples (int, optional): Size of training data. Defaults to 10000.

    Returns:
        pd.DataFrame: Training dataframe
    """
    logger.info('Generating sample training data.')
    housing_data = pd.DataFrame({
        'age_of_house': np.random.randint(0, 100, samples),
        'house_id': [
            f'house_ID_{i}' for i in np.random.randint(0, 100, samples)],
        'rented_house': np.random.choice(['yes', 'no'], samples),
        'house_area_m2': np.random.normal(50, 100, samples),
        'num_bedrooms': np.random.randint(1, 6, samples),
        'num_bathrooms': np.random.randint(1, 4, samples),
        'price': np.random.uniform(200000, 1200000, samples),
        'location_category': np.random.choice(
            ['Urban', 'Suburban', 'Rural'], samples),
        'near_school': np.random.choice([True, False], samples),
        'crime_rate': np.random.uniform(0, 10, samples),
        'economic_deal': np.random.choice(['yes', 'no'], samples)
    })
    return housing_data


def get_class_weights(y_train: np.ndarray) -> Dict:
    """Calculates class weights for the given training labels.

    This function is useful for handling class imbalance in classification
    tasks. It computes the weights for each class to balance the dataset.

    Args:
        y_train (np.ndarray): An array of training labels.

    Returns:
        Dict: A dictionary where keys are classes and values are the computed
              class weights. This helps in balancing the classes during
              model training.

    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights))

    return class_weights


def preprocess_data(training_data: pd.DataFrame) -> pd.DataFrame:
    """This function has been limited in its preprocessing capabilities.
    It could be extended to include imputation, normalisation and other
    preprocessing steps depending on the problem. Here it only sets the
    correct data types and eliminates redundant columns.

    Args:
        training_data (pd.DataFrame): Training data

    Returns:
        pd.DataFrame: Preprocessed training data
    """
    logger.info('Preprocessing training data.')

    integer_columns = ['age_of_house', 'num_bedrooms', 'num_bathrooms']
    float_columns = ['house_area_m2', 'crime_rate', 'price']
    category_columns = ['rented_house', 'location_category', 'near_school']
    target_column = ['economic_deal']

    # Remove unnecessary columns
    not_required_cols = (
        set(training_data.columns) - set(integer_columns) -
        set(float_columns) - set(category_columns) -
        set(target_column)
    )
    training_data.drop(columns=list(not_required_cols), inplace=True)

    # Set Data Types
    training_data[integer_columns] = training_data[
        integer_columns].astype('int')
    training_data[float_columns] = training_data[
        float_columns].astype('float64')
    training_data[category_columns] = training_data[
        category_columns].astype('category')
    training_data[target_column] = training_data[
        target_column].astype('category')

    return training_data


def print_classification_report(
    classifier: CatBoostClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> None:
    """Prints the classification report and confusion matrix for the
    given validation data and classifier.

    Args:
        classifier (CatBoostClassifier): The trained classifier to evaluate.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Actual validation labels.

    Returns:
        None: This function does not return anything. It prints the
            classification report and displays a confusion matrix heatmap.
    """
    y_pred = classifier.predict(X_val)
    # y_proba = classifier.predict_proba(X_val)[:, 1]

    # Print formatted report
    print("Binary Classification Report:")
    print("-----------------------------")
    print(classification_report(y_val, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    return None
