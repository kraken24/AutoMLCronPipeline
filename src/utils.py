from typing import Dict
import yaml
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
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

    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return TrainingParameters(**params)


def load_sample_data(samples: int = 10000) -> pd.DataFrame:
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
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights))

    return class_weights


def preprocess_data(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    # Data Imputation
    # Data Normalisation
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
    y_pred = classifier.predict(X_val)
    y_proba = classifier.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_pred))
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    # f1 = f1_score(y_val, y_pred, labels=classifier.classes_)
    # precision = precision_score(y_val, y_pred, labels=classifier.classes_)
    recall = recall_score(y_val, y_pred, labels=classifier.classes_)
    roc_auc = roc_auc_score(y_val, y_proba, labels=classifier.classes_)

    # Print formatted report
    print("Binary Classification Report:")
    print("-----------------------------")
    print(classification_report(y_val, y_pred))
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    return None
