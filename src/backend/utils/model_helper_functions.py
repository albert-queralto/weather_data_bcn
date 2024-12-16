import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Generator, Union, Optional, Any
from statsmodels.tsa.stattools import pacf
from loaders.training import ModelVersioningManager


def simulate_real_time_data(df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
    """
    Simulates real-time data using an input dataframe.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe with the data to be predicted.
    """
    for _, row in df.iterrows():
        row_df = pd.DataFrame(row)
        row_df = row_df.T
        yield row_df.reindex(
            pd.date_range(row_df.index[0], periods=1, freq=df.index.freq)
        )

def create_supervised_data_single_variable(
        df: Union[pd.DataFrame, pd.Series], nlags: int = 50
    ) -> pd.DataFrame:
    """
    Creates the supervised data to train the model creating time lags using
    partial autocorrelation.
    """
    df = pd.DataFrame(df) if isinstance(df, pd.Series) else df
    df = df.astype(float)

    df.interpolate(method='linear', inplace=True)
    df = df.ffill().bfill()

    n_lags = (df.shape[0] // 2) - 1
    nlags = min(nlags, n_lags)

    features = pd.DataFrame(index=df.index)

    # Compute the partial autocorrelation function for the time series and handle
    # the case where numpy.linalg.LinAlgError: Singular matrix is raised
    try:
        partial = pacf(df.values.squeeze(), nlags=nlags)
    except np.linalg.LinAlgError:
        partial = np.zeros(nlags + 1)

    lags = [i for i, val in enumerate(partial) if abs(val) >= 0.2]
    if 0 in lags:
        lags.remove(0)
    # Create the lag features
    for lag in lags:
        features[f'lag_{lag}'] = df.shift(lag)

    features = features.dropna()
    features = features.round(2)
    return features
    
def ensure_array(
        y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensures that the input variables are numpy arrays.
        """
        y_true_vals = np.asarray(y_true)
        y_pred_vals = np.asarray(y_pred)
        return y_true_vals, y_pred_vals

def check_trained_models(
        latitude: str,
        longitude: str,
        model_type: str,
        target_variable: str,
        train_models_older_than: int,
        col_prefix: Optional[str],
        logger: logging.Logger,
        model_loader: ModelVersioningManager
    ) -> tuple[bool, list[str]]:
    """
    Checks if there are trained models for the station, model type and 
    target variable. If there are trained models checks if they are up to date.
    """
    # Get the variable column name based on the prefix
    variable_column = f"{col_prefix}\\{latitude}\\{longitude}\\{target_variable}"
    if col_prefix in [None, "None", ""]:
        variable_column = f"{latitude}\\{longitude}\\{target_variable}"
    
    # Get the list of trained models from the database
    trained_models = model_loader.get_recent_models_from_db(
        model_type=model_type,
        latitude=latitude,
        longitude=longitude,
        target_variable=target_variable,
        columns=['target_variable', 'model_version', 'model_date']
    )
    # ------------------- DEBUGGING -------------------
    logger.debug(f"Number of models trained for {variable_column}: {len(trained_models)}")
    logger.debug(f"Trained models: {trained_models}")
    # -------------------------------------------------

    # Get the models_to_train older than train_models_older_than days
    models_to_train = set()
    # ------------------- DEBUGGING -------------------
    logger.debug(f"Target variable: {target_variable}")
    logger.debug(f"Variable column: {variable_column}")
    # -------------------------------------------------

    if trained_models:
        # Get the models older than a certain date
        models_to_train.update(get_models_older_than(
            logger=logger,
            trained_models=trained_models,
            train_models_older_than=train_models_older_than,
            target_variable=target_variable,
            variable_column=variable_column
        ))
    else:
        models_to_train.add(variable_column)
        # ------------------- DEBUGGING -------------------
        logger.debug(f"Model for {variable_column} is not trained. Adding model to training list.")
    # ------------------- DEBUGGING -------------------
    logger.debug(f"Models to train: {models_to_train}")
    logger.debug(f"Are there models to train?: {bool(models_to_train)}")
    # -------------------------------------------------
    return bool(models_to_train), list(models_to_train)

def get_models_older_than(
        logger: logging.Logger,
        trained_models: list[Any],
        train_models_older_than: int,
        target_variable: str,
        variable_column: str
    ) -> set[str]:
        models_to_train = set()

        if not trained_models:
            return {variable_column}
        
        variables_list = [variable for variable, _, _ in trained_models]
        
        for variable, _, date in trained_models:
            date_difference = (
                datetime.now() - pd.Timedelta(days=train_models_older_than)
            )
            # ------------------- DEBUGGING -------------------
            logger.debug(f"Variable: {variable}, date: {date}")
            logger.debug(f"Date difference: {date_difference}")
            # -------------------------------------------------

            match = re.search(f"({target_variable})$", variable)
            # ------------------- DEBUGGING -------------------
            logger.debug(f"Match: {match}")
            # -------------------------------------------------

            if match and date <= date_difference:
                # ------------------- DEBUGGING -------------------
                logger.debug(f"Model for {variable_column} is outdated. Adding model to training list.")
                # -------------------------------------------------
                models_to_train.add(variable_column)
            elif match and date > date_difference:
                # ------------------- DEBUGGING -------------------
                logger.debug(f"Model for {variable_column} is up to date. No action taken.")
                # -------------------------------------------------
                continue
            elif not match and variable_column not in variables_list:
                models_to_train.add(variable_column)
                # ------------------- DEBUGGING -------------------
                logger.debug(f"Model for {variable_column} is not trained. Adding model to training list.")
        return models_to_train

