import numpy as np
import pandas as pd
from dataclasses import dataclass
from utils.custom_logger import CustomLogger

@dataclass
class ValidateRealTimePredictions:
    """
    Generates predictions for the different models trained.
    """
    logger: CustomLogger

    def calculate_forecast_accuracies(self,
            row_y: pd.Series,
            valor_y_pred_row: pd.Series,
            tolerance: float = 1e-3
        ) -> list[float]:
        forecast_accuracies = []
        for real, prediction in zip(row_y, valor_y_pred_row):
            forecast_acc = self.get_forecast_accuracy(real, prediction, tolerance)
            forecast_accuracies.append(forecast_acc)
        return forecast_accuracies

    def get_forecast_accuracy(self,
            real: float, 
            prediction: float,
        ) -> float: 
        try:
            forecast_acc = (1 - (abs(prediction - real) / real)) * 100
            return np.clip(forecast_acc, 0, 100)
        except Exception as e:
            return 0
         

    def get_iqr_quantiles(self, data: pd.Series) -> float:
        q1, q3 = np.quantile(data, [0.25, 0.75])
        return q3 - q1, q1, q3

    def get_thresholds(self,
            residuals: float,
            validation_threshold: float,
            model_lower_threshold: float,
            model_upper_threshold: float
        ) -> tuple[float, float]:
        iqr, q1, q3 = self.get_iqr_quantiles(residuals)

        lower_threshold = q1 - validation_threshold * iqr
        upper_threshold = q3 + validation_threshold * iqr

        # Uses the most restrictive thresholds to validate the data
        if lower_threshold < model_lower_threshold:
            lower_threshold = model_lower_threshold
        if upper_threshold > model_upper_threshold:
            upper_threshold = model_upper_threshold

        return lower_threshold, upper_threshold

    def get_validation_values(self,
            row_y: float,
            y_pred_row: float,
            validation_threshold: float,
            model_lower_threshold: float,
            model_upper_threshold: float
        ) -> int:
            """
            Validates the data from the row using the thresholds stored in the
            ModelVersioningTable.
            """
            residuals = row_y.astype(float) - y_pred_row.astype(float)
            lower_threshold, upper_threshold = self.get_thresholds(residuals,
                validation_threshold, model_lower_threshold, model_upper_threshold)

            return np.logical_or(
                lower_threshold <= residuals,
                residuals <= upper_threshold
            ).astype(int)
            
    def get_suggested_values(self,
            real: float,
            prediction: float,
            validation_boolean: int,
            forecast_accuracy: float,
            accuracy_threshold: float
        ) -> float:
            if validation_boolean == 1 and forecast_accuracy >= accuracy_threshold:
                return real
            else:
                return prediction