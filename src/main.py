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
    """Objective function for hyperparameter tuning in CatBoostClassifier.

    This function is designed to be used with an Optuna optimization study.
    It suggests hyperparameters for a given trial, trains a CatBoostClassifier
    with these parameters, and computes its accuracy on the validation pool.

    Args:
        trial (Trial): An Optuna trial object that suggests hyperparameters.
        train_pool (Pool): The training data pool.
        val_pool (Pool, optional): The validation data pool. If None, the
            training pool is used for evaluation. Defaults to None.
        class_weights (Optional[Dict[Any, Any]], optional): Class weights for
            handling class imbalance during model training. Defaults to None.

    Returns:
        float: The accuracy of the CatBoostClassifier model trained with the
            suggested hyperparameters on the validation pool.
    """
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
    """Determines the optimal hyperparameters for a CatBoostClassifier model.

    The function operates in two modes based on the 'tune_parameters' flag:
    1. If 'tune_parameters' is False, it returns a set of default parameters.
    2. If 'tune_parameters' is True, it performs hyperparameter tuning using
       the provided training and validation data pools.

    Hyperparameter tuning is done via a study that maximizes a given metric
    (e.g., accuracy) and uses a random sampler and a median pruner.

    Args:
        train_pool (Pool): The training data pool used for model fitting and
            hyperparameter tuning.
        tune_parameters (bool): Flag indicating whether to tune parameters.
            If False, the function returns default parameters.
        val_pool (Pool, optional): The validation data pool used for evaluating
            the model during the tuning process. Defaults to None.
        class_weights (Optional[Dict[Any, Any]], optional): Class weights to
            handle class imbalance. Defaults to None.

    Returns:
        Dict: A dictionary containing the optimal hyperparameters. If
            'tune_parameters' is False, it returns default hyperparameters.
    """
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

    return optimal_parameters


def train(
    args: argparse.Namespace,
    training_data: pd.DataFrame,
    training_params: TrainingParameters
) -> CatBoostClassifier:
    """Trains a CatBoostClassifier using the provided training data
    and parameters.

    Args:
        args (argparse.Namespace): Contains command line arguments,
            specifically 'tune_parameters' for hyperparameter tuning.
        training_data (pd.DataFrame): DataFrame containing the training data.
        training_params (TrainingParameters): Object containing training
            parameters such as 'target_col', 'validation_set_size', etc.

    Returns:
        CatBoostClassifier: The trained CatBoost classifier model.
    """
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
    """Saves the trained model to a specified file path and in the database.

    Args:
        model (Type): The trained model that needs to be saved. This could be
            any model object that is compatible with joblib for serialization.
        training_params (TrainingParameters): An object containing training
            parameters, including the directory and filename where the model
            should be saved.

    Returns:
        None: This function does not return anything.
    """

    model_filepath = os.path.join(
        training_params.model_directory,
        training_params.model_name
    )
    if not os.path.exists(training_params.model_directory):
        os.mkdir(training_params.model_directory)

    joblib.dump(model, model_filepath)
    logger.info(f'Model saved to {model_filepath}.')

    return None


def main(args: argparse.Namespace) -> None:
    """Main function to execute the model training process.

    This function follows these major steps:
    1. Load configuration and training parameters.
    2. Update training parameters from a configuration file if specified.
    3. Load and preprocess the training data.
    4. Train the model with the given arguments, training data, and parameters.
    5. Save the trained model and its metrics.

    Args:
        args (argparse.Namespace): Arguments received from the command line.
            It includes options like config_file which specifies the path
            to a configuration file for updating training parameters.

    Returns:
        None: This function returns None.

    Note:
        - The function uses a default sample data loader (`load_sample_data`)
          which ideally should be replaced with a `query_training_data`
          function for real-world applications.
        - The `train` function is responsible for the actual training process.
          It takes training data, parameters, and additional args as input.
    """

    # Load Config, Training Parameters
    logger.info('Generating training parameters.')
    training_params = TrainingParameters()
    if args.config_file:
        logger.info(f'Updating training parameters from {args.config_file} '
                    'config file')
        training_params = TrainingParameters.from_yaml(args.config_file)

    # Load Training Data & Train Model
    # Normally query_training_data function should be called in place of
    # load_sample_data function
    training_data = load_sample_data()
    training_data = preprocess_data(training_data)
    classifier = train(args, training_data, training_params)
    # Save Model & Metrics
    save_model(classifier, training_params)

    return None


if __name__ == '__main__':
    # conda activate tf_m1 <venv>
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
