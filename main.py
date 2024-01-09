import os
import argparse
import yaml
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from catboost import CatBoostClassifier, Pool
import optuna
from optuna import Trial
from optuna.samplers import RandomSampler
from optuna.pruners import MedianPruner
from optuna.integration import CatBoostPruningCallback

import joblib
# from dataclass import dataclass

from utils import load_data, load_config

# @dataclass
# class TrainingParameters:
#     """Class to load all training parameters"""
#     pass


def objective_fn(
    trial: Trial,
    train_pool: Pool,
    val_pool: Pool = None
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
        y_pred=cat.predict(val_pool)
    )

    return accuracy


def find_optimal_hyperparameters(
    train_pool: Pool,
    tune_parameters: bool,
    val_pool: Pool = None
) -> Optional[Dict]:
    if not tune_parameters:
        return None
    study = optuna.create_study(
        direction='maximize',
        sampler=RandomSampler(),
        pruner=MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(
            trial,
            train_pool,
            val_pool,
        )
    )

    optimal_parameters = {**study.best_params}

    return optimal_parameters


def train(
    training_data: pd.DataFrame,
    args: argparse.Namespace
) -> CatBoostClassifier:

    X = training_data.drop('target_col', axis=1)
    y = training_data['target_col'].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    train_pool = Pool(X_train, y_train, cat_features)
    val_pool = Pool(X_val, y_val, cat_features)

    clf_params = find_optimal_hyperparameters(
        train_pool, args.tune_parameters, val_pool)

    classifier = CatBoostClassifier(**clf_params)
    classifier.fit(train_pool, eval_set=val_pool, use_best_model=True)

    return classifier


def main(args: argparse.Namespace) -> None:
    load_config(args.config_file)
    training_data = load_data()
    train(training_data, args)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a machine learning model'
    )
    parser.add_argument(
        '-t', '--tune_parameters',
        default=True,
        dest='tune_parameters',
        type=str, nargs='?',
        help='Activate or deactivate hyperparameter tuning.'
    )
    parser.add_argument(
        '-c', '--config_file',
        default='/path/to/config/file',
        dest='config_file',
        type=str, nargs='?',
        help='Specify path to the config file.'
    )
    args = parser.parse_args()
    # main(args)
    print(load_data())
