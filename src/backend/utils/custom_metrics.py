import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn import metrics as skmetrics
from utils.model_helper_functions import ensure_array_type


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
		"""Initialize the custom metrics class."""
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
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
		"""
		Calculates the mean squared error (MSE) metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.mean_squared_error(y_true_vals, y_pred_vals))

	@staticmethod
	def _root_mean_squared_error(
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
	) -> float:
		"""
		Calculates the root mean squared error (RMSE) metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.root_mean_squared_error(y_true_vals, y_pred_vals))

	@staticmethod
	def _root_mean_squared_percentage_error(
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
	) -> float:
		"""
		Calculates the root mean squared percentage error (RMSPE) metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return 100 * np.sqrt(np.mean(np.square(y_true_vals - y_pred_vals))) / np.mean(y_true_vals + 1e-10)

	@staticmethod
	def _mean_absolute_error(
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
	) -> float:
		"""
		Calculates the mean absolute error (MAE) metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.mean_absolute_error(y_true_vals, y_pred_vals))
		
	@staticmethod
	def _r2(
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
	) -> float:
		"""
		Calculates the coefficient of determination (R^2) metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.r2_score(y_true_vals, y_pred_vals))


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
		"""Initialize the custom metrics class."""
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
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
		"""
		Calculates the accuracy metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.accuracy_score(y_true_vals, y_pred_vals))

	@staticmethod
	def _balanced_accuracy(
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
		"""
		Calculates the balanced accuracy metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.balanced_accuracy_score(y_true_vals, y_pred_vals))

	@staticmethod
	def _f1_score(
		y_true: Union[pd.Series, np.ndarray],
		y_pred: Union[pd.Series, np.ndarray],
		score_setting: str
    ) -> float:
		"""
		Calculates the balanced accuracy metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.f1_score(y_true_vals, y_pred_vals, average=score_setting))

	@staticmethod
	def _log_loss(
		y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
	) -> float:
		"""
		Calculates the balanced accuracy metric.
		"""
		y_true_vals, y_pred_vals = ensure_array_type(y_true, y_pred)
		return float(skmetrics.log_loss(y_true_vals, y_pred_vals))