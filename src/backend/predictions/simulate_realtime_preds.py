import pandas as pd
from typing import Any
from dataclasses import dataclass
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Custom modules
from utils.model_helper_functions import simulate_real_time_data
from utils.custom_logger import CustomLogger
from predictions.validation import ValidateRealTimePredictions


@dataclass
class SimulateRealTimePredictions:
    """
    Generates predictions for the different models trained.
    """
    latitude: float
    longitude: float
    target_variable: str
    logger: CustomLogger

    def __post_init__(self):
        self.data_validator = ValidateRealTimePredictions(self.logger)

    def predict_next_values(self,
            df: pd.DataFrame,
            validation_threshold: float,
            accuracy_threshold: float,
            regression_models: dict[str, Any],
        ) -> pd.DataFrame:
            """
            Predicts the next values based on the rows of a dataframe, simulating
            real-time predictions. Validates the data based on the model scores and
            the thresholds. Generates new columns with the validated data.
            """
            # Initialize the lists to store the results
            timestamps, model_dates = [], []
            model_names, model_types, model_versions = [], [], []
            latitudes, longitudes, target_variables = [], [], []
            y_reals, y_preds, suggested_values = [], [], []
            data_quality_percentages, validation_booleans = [], []
            feature_importance_variables, feature_importance_values = [], []

            # Create the dataframe to store the results
            preds_df = pd.DataFrame()
            for target_variable, (
                model_date,
                model_name,
                model_type,
                latitude,
                longitude,
                _, # model_feature_names
                model_version,
                _, # model_score
                _, # model_forecast_acc
                scaler,
                polynomial,
                model,
                lower_threshold,
                upper_threshold,
                feature_importance_variable,
                feature_importance_value
            ) in regression_models.items(): # type: ignore
                # ------------------- DEBUGGING -------------------
                self.logger.debug(f"Latitude: {latitude} - Longitude: {longitude} | Predicting for {model_name}_{target_variable}")
                # -------------------------------------------------
                # Simulate real-time predictions
                for row in tqdm(simulate_real_time_data(df)):
                    # ------------------- DEBUGGING -------------------
                    self.logger.debug(f"Latitude: {latitude} - Longitude: {longitude} | Predicting on date: {row.index}")
                    self.logger.debug(f"Latitude: {latitude} - Longitude: {longitude} | Row: {row}")
                    # -------------------------------------------------

                    # Add the variables to their respective lists
                    timestamps.append(row.index[0])
                    model_dates.append(model_date)
                    model_names.append(model_name)
                    model_types.append(model_type)
                    model_versions.append(model_version)
                    latitudes.append(latitude)
                    longitudes.append(longitude)
                    target_variables.append(target_variable)


                    # Apply the transformations to the row
                    X_scaled, y_scaled = self._row_transformations(
                        row,
                        target_variable,
                        scaler,
                        polynomial,
                        model
                    )

                    # Make predictions, unscale them and add the results to y_preds
                    y_pred_row = pd.Series(model.predict(X_scaled), name=y_scaled.name)

                    # Unscale the y and y_pred
                    y_real, y_pred = self._row_unscaling(
                        X=X_scaled,
                        y=y_scaled,
                        y_pred=y_pred_row,
                        scaler=scaler,
                    )
                    
                    y_reals.append(float(y_real.iloc[0]))
                    y_preds.append(float(y_pred.iloc[1]))
                    
                    # Evaluate the quality of the prediction
                    forecast_acc = self.data_validator.get_forecast_accuracy(
                                                y_real.iloc[0], y_pred.iloc[1])
                    data_quality_percentages.append(forecast_acc)

                    # Validate the data based on the thresholds
                    validation_boolean = \
                        self.data_validator.get_validation_values(
                            y_real.iloc[0], y_pred.iloc[1],
                        validation_threshold, lower_threshold, upper_threshold)
                    validation_booleans.append(validation_boolean)

                    # Obtain the suggested values based on the validation boolean
                    suggested_value = self.data_validator.get_suggested_values(
                            y_real.iloc[0], y_pred.iloc[1], validation_boolean,
                            forecast_acc, accuracy_threshold)
                    suggested_values.append(suggested_value)

            # Add the data to the dataframe
            preds_df["timestamp"] = timestamps
            preds_df["model_date"] = model_dates
            preds_df["model_name"] = model_names
            preds_df["model_type"] = model_types
            preds_df["model_version"] = model_versions
            preds_df["latitude"] = latitudes
            preds_df["longitude"] = longitudes
            preds_df["target_variable"] = target_variables
            preds_df["real_value"] = y_reals
            preds_df["predictions"] = y_preds
            preds_df["data_quality_percentage"] = data_quality_percentages
            preds_df["validation_boolean"] = validation_booleans
            preds_df["suggested_value"] = suggested_values

            return preds_df

    def _row_transformations(self,
            row: pd.DataFrame,
            target_variable: str,
            scaler: StandardScaler,
            poly: PolynomialFeatures,
            model: Any
        ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Applies the transformations to a row of the dataframe before using it
        to make predictions.
        """
        # Split into features and target
        row_X = row.drop(columns=target_variable)
        row_y = row[target_variable]
        
        # Transform the data
        X_poly = row_X.copy()
        
        if poly is not None:
            row_X = row_X[poly.feature_names_in_]
            X_poly = X_poly[poly.feature_names_in_]
            X_poly = poly.transform(row_X)
            X_poly = pd.DataFrame(X_poly,
                                index=row_X.index,
                                columns=poly.get_feature_names_out(row_X.columns))
    
        # Combine the X and y
        data = pd.concat([X_poly, row_y], axis=1)
        
        # Add missing columns in data that are in the scaler
        missing_cols = set(scaler.feature_names_in_) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        
        # Reorder the columns based on the scaler feature names
        data = data[scaler.feature_names_in_]
        
        # Scale the data
        data_scaled = scaler.transform(data)
        data_scaled = pd.DataFrame(
            data_scaled,
            index=data.index,
            columns=data.columns
        )

        # Split the data
        X_scaled = data_scaled.drop(columns=row_y.name)
        y_scaled = data_scaled[row_y.name]

        # Reorder and filter the columns based on the model feature names
        X_scaled = X_scaled[model.feature_names_in_]
        
        return X_scaled, y_scaled

    def _row_unscaling(self,
            X: pd.DataFrame,
            y: pd.Series,
            y_pred: pd.Series,
            scaler: StandardScaler,
        ) -> tuple[pd.Series, pd.Series]:
        """
        Applies the transformations to a row of the dataframe before using it
        to make predictions.
        """
        # Combine the X and y, and scale the data
        original_data = pd.concat([X, y], axis=1)
        original_data_unscaled = scaler.inverse_transform(original_data)
        original_data_unscaled = pd.DataFrame(
            original_data_unscaled,
            index=original_data.index,
            columns=original_data.columns
        )

        pred_data = pd.concat([X, y_pred], axis=1)
        pred_data_unscaled = scaler.inverse_transform(pred_data)
        pred_data_unscaled = pd.DataFrame(
            pred_data_unscaled,
            index=pred_data.index,
            columns=pred_data.columns
        )

        # Get the y columns
        y_unscaled = original_data_unscaled[y.name]
        y_pred_unscaled = pred_data_unscaled[y.name]

        return y_unscaled, y_pred_unscaled