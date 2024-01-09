import yaml
import numpy as np
import pandas as pd


def load_config():
    raise NotImplementedError


def load_data(samples: int = 10000) -> pd.DataFrame:
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
    })

    return housing_data


def update_data_class_from_yaml(
    file_path: str,
    data_class: 'dataclass'
) -> None:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    for key, value in data.items():
        if hasattr(data_class, key):
            setattr(data_class, key, value)


def print_classification_report(classifier, X_val, y_val):
    y_pred = classifier.predict(X_val)
    y_proba = classifier.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)

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
    plt.show()

    return None
