import os
import re
import numpy as np
import pandas as pd
from typing import Any
import joblib

import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import ExperimentalWarning

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from dataclasses import dataclass

from utils.metrics import RegressionMetrics

import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)


@dataclass
class OptunaStudy:
    """
    Automates the process of tuning the hyperparameters of a model. It does this 
    by using Optuna's library's capabilities to perform an efficient search for 
    the best hyperparameters.
    """
    params: dict
    model_name: str
    task_type: str # 'autoval_reg' or 'autoval_clf'
    scoring: str = 'balanced_accuracy' # Only for classification
    test_size: float = 0.1
    early_stop_patience: int = 5
    early_stop_mode: str = 'max' # 'min' or 'max'
    early_stop_threshold: float = 0.8

    def _create_hyperparams(self, trial: optuna.trial.Trial) -> dict:
        """
        Creates the hyperparameters to be optimized. First, creates the ranges to 
        use based on the params dictionary. Then, stores them into the 
        optuna_hyperparams dictionary.
        Before that, it transforms each hyperparameter based on the type of input.
        """
        optuna_dict = {}

        for key, value in self.params.items():
            if '__' in key:
                key = key.split('__')[-1]
            if isinstance(value, tuple):
                suggest_func = trial.suggest_int if isinstance(value[0], int) else trial.suggest_float
                optuna_dict[key] = suggest_func(key, min(value), max(value))
            elif isinstance(value, list):
                if isinstance(value[0], (int, float)):
                    suggest_func = trial.suggest_int if isinstance(value[0], int) else trial.suggest_float
                    optuna_dict[key] = suggest_func(key, min(value), max(value))
                else:
                    optuna_dict[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, (np.ndarray, np.generic)):
                optuna_dict[key] = trial.suggest_categorical(key, value.tolist())
        return optuna_dict

    def _model_optimizer(self, 
            X_train: pd.DataFrame,
            Y_train: pd.Series,
            model: Any,
            trial: optuna.trial.Trial, 
            cv: int
        ) -> float:
        """
        Implements the model that will be optimized using the hyperparameters
        and optuna and returns the average score from cross-validation.
        """
        # Creates the hyperparams using Optuna and loads them into the model
        optuna_dict = self._create_hyperparams(trial)
        estimator = model.set_params(**optuna_dict)

        # Performs the cross-validation and returns the score based on the task type
        if self._is_regression_task():
            score = self._time_series_cross_val(X_train, Y_train, estimator, cv, trial)
        elif self._is_classification_task():
            scores = cross_val_score(estimator, X_train, Y_train, cv=cv, scoring=self.scoring)
            score = float(np.mean(scores))
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return score

    def _is_regression_task(self) -> bool:
        return any(substring in self.task_type for substring in ['regression', 'reg'])

    def _is_classification_task(self) -> bool:
        return any(substring in self.task_type for substring in ['classification', 'clf'])

    def _time_series_cross_val(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            estimator: Any, 
            cv: int, 
            trial: optuna.trial.Trial
        ) -> float:
        """
        Performs the time series cross validation.
        """
        def calculate_rmse(estimator, X_train, y_train, X_test, y_test):
            estimator.fit(X_train, y_train)
            y_preds = estimator.predict(X_test)
            return RegressionMetrics()(
                metric_name='root_mean_squared_error',
                y_true=y_test,
                y_pred=y_preds,
            )

        scores = []
        tscv = TimeSeriesSplit(n_splits=cv) 
        for idx, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            rmse = calculate_rmse(estimator, X_train, y_train, X_test, y_test)
            
            trial.report(float(rmse), step=idx+1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            scores.append(float(rmse))
            
        return float(np.mean(scores))

    def create_study(self,
        study_name: str,
        storage: optuna.storages.BaseStorage
    ) -> optuna.study.Study:
        """
        Creates the study to be used by Optuna to optimize the hyperparameters.
        """
        if self._is_regression_task():
            direction = 'minimize'
        elif self._is_classification_task():
            direction = 'maximize'
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        sampler = TPESampler(multivariate=True)
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        return study

    def run_optuna(self,
            study: optuna.study.Study,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            model: Any, 
            cv: int,
            n_trials: int = 100,
            n_jobs: int = -1
        ) -> optuna.study.Study:
        """
        Runs the optimization _model_optimizer, using the hyperparam ranges defined by 
        _create_hyperparams and using as many trials as defined by n_trials. Each
        trial uses a different combination of hyperparameters to test.
        """
        callbacks = [
            OptunaEarlyStoppingCallback(patience=self.early_stop_patience, mode=self.early_stop_mode),
            OptunaConsecutiveTrialPruning(consecutive_trials=self.early_stop_patience),
            OptunaEarlyStoppingCallbackThreshold(threshold=self.early_stop_threshold, mode=self.early_stop_mode)
        ]

        func = lambda trial: self._model_optimizer(X_train, y_train, model, trial, cv)
        study.optimize(
            func, 
            n_trials=n_trials, 
            callbacks=callbacks,
            show_progress_bar=True,
            n_jobs=n_jobs
        )
        return study

    def run_parallel_optuna(self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model: Any, 
        cv: int,
        n_trials: int = 100,
        n_jobs: int = -1
    ) -> tuple[optuna.study.Study, str, str]:
        target_variable_name = re.sub(r'\W+', '_', str(y_train.name))
        study_name=f'{self.model_name}_{target_variable_name}_study'
        storage_name=f'sqlite:///{study_name}.db'
        storage = optuna.storages.RDBStorage(
            url=storage_name,
            heartbeat_interval=60,
            grace_period=120,
            engine_kwargs={"connect_args": {"timeout": 30.0}}
        )
        study = self.create_study(
            study_name=study_name,
            storage=storage
        )
        if n_jobs == -1:
            n_jobs = joblib.cpu_count()

        parallel = joblib.Parallel(n_jobs=n_jobs, prefer='processes')
        parallel(joblib.delayed(self.run_optuna)(
                study,
                X_train,
                y_train,
                model,
                cv,
                n_trials=n_trials_i,
                n_jobs=n_jobs
            ) for n_trials_i in self._split_trials(n_trials, n_jobs)
        )
        return study, study_name, storage_name

    def delete_storage(self,
        storage_name: str
    ) -> None:
        """
        Deletes the database storage file for the study.
        """
        storage_file = storage_name.split('///')[-1]
        if os.path.exists(f'{storage_file}'):
            os.remove(f'{storage_file}')

    @staticmethod
    def _split_trials(n_trials, n_jobs):
        # https://github.com/optuna/optuna/issues/2862
        n_per_job, remaining = divmod(n_trials, n_jobs)
        for _ in range(n_jobs):
            yield n_per_job + (1 if remaining > 0 else 0)
            remaining -= 1


class OptunaEarlyStoppingCallback:
    """
    Implements early stopping for Optuna trials if the score is not improving
    for a set number of trials based on the patience parameter.

    Based on code from:
    - https://github.com/optuna/optuna/issues/1001
    - https://github.com/keras-team/keras/blob/v2.12.0/keras/callbacks.py
    - https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
    """
    def __init__(self,
        patience: int = 5,
        mode: str = "max"
    ) -> None:
        self.patience = patience
        self.mode = mode
        self._best_score = None
        self._counter = 0
        # Check if mode is valid and assign correct numpy function
        if mode not in ["min", "max"]:
            raise ValueError('Mode must be either "min" or "max".')
        if self.mode == "min":
            self.monitor = np.less
        elif self.mode == "max":
            self.monitor = np.greater

    def __call__(self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial
    ) -> None:
        # Assign the best score if it is None
        if self._best_score is None:
            self._best_score = study.best_value
        # Check if the score is improving or not
        if self.monitor(study.best_value, self._best_score):
            self._best_score = study.best_value
            self._counter = 0
        else:
            self._counter += 1
        # Check if the counter is greater than the patience and stop the study
        if self._counter >= self.patience:
            study.stop()


class OptunaConsecutiveTrialPruning:
    """
    Implements early stopping for Optuna trials if they are consecutively pruned.

    Based on code from:
    - https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
    """
    def __init__(self, consecutive_trials: int = 5) -> None:
        self.consecutive_trials = consecutive_trials
        self._counter = 0

    def __call__(self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial
    ) -> None:
        # Check if the trial was pruned or not and count the number of pruned trials
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._counter += 1
        else:
            self._counter = 0
        # Check if the counter is greater than the patience and stop the study
        if self._counter >= self.consecutive_trials:
            study.stop()


@dataclass
class OptunaEarlyStoppingCallbackThreshold:
    """
    Implements early stopping for Optuna trials if the score is above a
    threshold based on the task type.
    """
    threshold: float
    mode: str = "max"

    def __post_init__(self) -> None:
        self._best_score = None
        # Check if mode is valid and assign correct numpy function
        if self.mode not in ["min", "max"]:
            raise ValueError('Mode must be either "min" or "max".')
        if self.mode == "min":
            self.monitor = np.less
        elif self.mode == "max":
            self.monitor = np.greater

    def __call__(self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial
    ) -> None:
        # Assign the best score if it is None
        if self._best_score is None:
            self._best_score = study.best_value
        # Check if the score is improving or not
        if self.monitor(study.best_value, self._best_score):
            self._best_score = study.best_value
        # Check if the score is above the threshold and stop the study
        if study.best_value > self.threshold:
            study.stop()