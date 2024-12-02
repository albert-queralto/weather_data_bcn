import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn import metrics as skmetrics
from utils.model_helper_functions import ensure_array


class RegressionMetrics:
    """
    Custom metrics class for measuring performance of a regression model.
        
    This class implements the following regression metrics:
    - mean squared error (MSE), 
    - root mean squared error (RMSE), 
    - mean absolute error (MAE), 
    - coefficient of determination (R^2),
    - root mean squared percentage error (RMSPE).
    """

    def __init__(self):
        self.metric_names = {
            "mean_squared_error": self._mean_squared_error,
            "root_mean_squared_error": self._root_mean_squared_error,
            "mean_absolute_error": self._mean_absolute_error,
            "root_mean_squared_percentage_error": self._root_mean_squared_percentage_error,
            "r2": self._r2,
        }

    def __call__(self,
            metric_name: str,
            y_true: Union[pd.Series, np.ndarray],
            y_pred: Union[pd.Series, np.ndarray], 
        ) -> float:
        if metric_name not in self.metric_names:
            raise NotImplementedError
        return self.metric_names[metric_name](y_true, y_pred)

    @staticmethod
    def _mean_squared_error(
            y_true: Union[pd.Series, np.ndarray], 
            y_pred: Union[pd.Series, np.ndarray]
        ) -> float:
        """
        Calculates the mean squared error (MSE) metric.
        """
        y_true_vals, y_pred_vals = ensure_array(y_true, y_pred)
        mse = np.mean((y_true_vals - y_pred_vals) ** 2)
        return float(mse)

    @staticmethod
    def _root_mean_squared_error(
        y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculates the root mean squared error (RMSE) metric.
        """
        y_true_vals, y_pred_vals = ensure_array(y_true, y_pred)
        mse = np.mean((y_true_vals - y_pred_vals) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)

    @staticmethod
    def _root_mean_squared_percentage_error(
        y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculates the root mean squared percentage error (RMSPE) metric.
        """
        y_true_vals, y_pred_vals = ensure_array(y_true, y_pred)
        percentage_errors = (y_true_vals - y_pred_vals) / (y_true_vals + 1e-10)
        return 100 * np.sqrt(np.mean(np.square(percentage_errors)))
    @staticmethod
    def _mean_absolute_error(
        y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculates the mean absolute error (MAE) metric.
        """
        y_true_vals, y_pred_vals = ensure_array(y_true, y_pred)
        return float(np.mean(np.abs(y_true_vals - y_pred_vals)))
        
    @staticmethod
    def _r2(
            y_true: Union[pd.Series, np.ndarray], 
            y_pred: Union[pd.Series, np.ndarray]
        ) -> float:
        """
        Calculates the coefficient of determination (R^2) metric.
        """
        y_true_vals = np.asarray(y_true)
        y_pred_vals = np.asarray(y_pred)
        ss_res = np.sum((y_true_vals - y_pred_vals) ** 2)
        ss_tot = np.sum((y_true_vals - np.mean(y_true_vals)) ** 2)
        return 1 - (ss_res / ss_tot)


class ClassificationMetrics:
    """
    Custom metrics class for measuring performance of a classification model.
        
    This class implements the following regression metrics:
    - accuracy, 
    - balanced accuracy
    - F1-Score
    - log loss
    """

    def __init__(self):
        self.metric_names = {
            "accuracy": self._accuracy,
            "balanced_accuracy": self._balanced_accuracy,
            "f1_score": self._f1_score,
            "neg_log_loss": self._log_loss
        }

    def __call__(self,
        metric_name: str,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        score_setting: Optional[str] = None
    ) -> float:
        if metric_name not in self.metric_names:
            raise NotImplementedError
        if metric_name == 'f1_score':
            return self.metric_names[metric_name](y_true, y_pred, score_setting)
        return self.metric_names[metric_name](y_true, y_pred)

    @staticmethod
    def _accuracy(
            y_true: Union[pd.Series, np.ndarray], 
            y_pred: Union[pd.Series, np.ndarray]
        ) -> float:
        """
        Calculates the accuracy metric.
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions

        return float(accuracy)

    @staticmethod
    def _balanced_accuracy(
            y_true: Union[pd.Series, np.ndarray], 
            y_pred: Union[pd.Series, np.ndarray]
        ) -> float:
        """
        Calculates the balanced accuracy metric.
        """
        y_true_vals, y_pred_vals = ensure_array(y_true, y_pred)
        cm = skmetrics.confusion_matrix(y_true_vals, y_pred_vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
            balanced_accuracy = np.nanmean(per_class_accuracy)
        return float(balanced_accuracy)
    
    @staticmethod
    def _f1_score(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        score_setting: str
    ) -> float:
        """
        Calculates the balanced accuracy metric.
        """
        y_true_vals = np.asarray(y_true)
        y_pred_vals = np.asarray(y_pred)
        return float(skmetrics.f1_score(y_true_vals, y_pred_vals, average=score_setting))

    @staticmethod
    def _log_loss(
            y_true: Union[pd.Series, np.ndarray], 
            y_pred: Union[pd.Series, np.ndarray]
        ) -> float:
        """
        Calculates the log loss metric.
        """
        y_true_vals, y_pred_vals = ensure_array(y_true, y_pred)
        epsilon = 1e-15
        y_pred_vals = np.clip(y_pred_vals, epsilon, 1 - epsilon)
        log_loss = -np.mean(y_true_vals * np.log(y_pred_vals) + (1 - y_true_vals) * np.log(1 - y_pred_vals))
        return float(log_loss)