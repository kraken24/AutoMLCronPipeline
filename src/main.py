import os
import argparse
from typing import Dict, Tuple, List, Optional, Type, Any
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuna import create_study, Trial
from optuna.samplers import RandomSampler
from optuna.pruners import MedianPruner
from optuna.integration import CatBoostPruningCallback
import joblib

from utils import (
    TrainingParameters,
    load_sample_data,
    get_class_weights,
    preprocess_data,
    print_classification_report
)
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='logs/message.log',
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    return logger


def objective_fn(
    trial: Trial,
    train_pool: Pool,
    val_pool: Pool = None,
    class_weights: Optional[Dict[Any, Any]] = None
) -> float:
    trial_params = {
        'iterations': trial.suggest_int(
            'iterations', 500, 1000, 50),
        'depth': trial.suggest_int(
            'depth', 6, 11, 1),
        'learning_rate': trial.suggest_discrete_uniform(
            'learning_rate', 0.01, 0.2, 0.01)
    }

    fix_params = {
        'eval_metric': 'Accuracy',
        'class_weights': class_weights,
        'verbose': 0,
    }

    params = {**fix_params, **trial_params}
    pruning_callback = CatBoostPruningCallback(trial, params['eval_metric'])
    classifier = CatBoostClassifier(**trial_params)

    val_pool = val_pool if val_pool else train_pool
    classifier.fit(train_pool, eval_set=val_pool, callbacks=[pruning_callback])

    # manual pruning check
    pruning_callback.check_pruned()

    accuracy = accuracy_score(
        y_true=val_pool.get_label(),
        y_pred=classifier.predict(val_pool)
    )

    return accuracy


def find_optimal_hyperparameters(
    train_pool: Pool,
    tune_parameters: bool,
    val_pool: Pool = None,
    class_weights: Optional[Dict[Any, Any]] = None
) -> Dict:
    if not tune_parameters:
        default_params = {
            'iterations': 20,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3,
            'eval_metric': 'Accuracy',
            'class_weights': class_weights,
            'verbose': 0,
        }
        return default_params
    logger.info('Starting hyperparameter tuning process.')
    study = create_study(
        direction='maximize',
        sampler=RandomSampler(),
        pruner=MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective_fn(
            trial,
            train_pool,
            val_pool),
        n_trials=5
    )
    optimal_parameters = {**study.best_params}
    print(type(study.best_params))

    return optimal_parameters


def train(
    args: argparse.Namespace,
    training_data: pd.DataFrame,
    training_params: TrainingParameters
) -> CatBoostClassifier:
    logger.info('Starting the classifier training process.')
    cat_features = list(training_data.select_dtypes(
        include=['category']).columns)
    if training_params.target_col in cat_features:
        cat_features.remove(training_params.target_col)

    X = training_data.drop(training_params.target_col, axis=1)
    y = training_data[training_params.target_col].to_numpy()

    if training_params.validation_set_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=training_params.validation_set_size,
            random_state=42,
            stratify=y
        )
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
    else:
        X_train, y_train = X, y
        val_pool = None

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    class_weights = get_class_weights(y_train)
    clf_params = find_optimal_hyperparameters(
        train_pool, args.tune_parameters, val_pool, class_weights)

    classifier = CatBoostClassifier(**clf_params)
    classifier.fit(train_pool, eval_set=val_pool, use_best_model=True)

    logger.info('Training process finished.')

    # if val_pool:
    #     print_classification_report(classifier, X_val, y_val)

    return classifier


def save_model(model: Type, training_params: TrainingParameters) -> None:
    model_filepath = os.path.join(
        training_params.model_directory,
        training_params.model_name
    )
    if not os.path.exists(training_params.model_directory):
        os.mkdir(training_params.model_directory)

    joblib.dump(model, model_filepath)
    logger.info(f'Model saved to {model_filepath}.')

    return None


def load_model(
    model_filepath: str = './trained_models/trained_classifier.joblib'
) -> CatBoostClassifier:
    try:
        model = joblib.load(model_filepath)
        logger.info(f'Trained model loaded from {model_filepath}')
    except Exception as error:
        logger.error(f'Trained model could not be loaded -> {error}')

    return model


def generate_predictions():
    pass


def main(args: argparse.Namespace) -> None:
    # Load Config, Training Parameters
    logger.info('Generating training parameters.')
    training_params = TrainingParameters()
    if args.config_file:
        logger.info(f'Updating training parameters from {args.config_file} '
                    'config file')
        training_params = TrainingParameters.from_yaml(args.config_file)

    # Load Training Data & Train Model
    training_data = load_sample_data()
    training_data = preprocess_data(training_data)
    classifier = train(args, training_data, training_params)
    # Save Model & Metrics
    save_model(classifier, training_params)

    return None


if __name__ == '__main__':
    # conda activate tf_m1
    parser = argparse.ArgumentParser(
        description='Train a machine learning model'
    )
    parser.add_argument(
        '-t', '--tune_parameters',
        default=False,
        dest='tune_parameters',
        type=str, nargs='?',
        help='Activate or deactivate hyperparameter tuning.'
    )
    parser.add_argument(
        '-c', '--config_file',
        default='',
        dest='config_file',
        type=str, nargs='?',
        help='Specify path to the config file.'
    )
    logger = setup_logging()
    logger.info("Logger initialized in main")

    args = parser.parse_args()
    main(args)
