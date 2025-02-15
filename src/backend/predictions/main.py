"""
Example command line:
---------------------
- python "/root/home/backend/predictions/main.py" -lat 41.389 -lon 2.159 -sd "2015-10-11" -ed "2024-10-12"
"""
import os
import time
import sys
import re
from ast import literal_eval
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

MAIN_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(MAIN_PATH))

from dotenv import load_dotenv
env_path = MAIN_PATH / '.env'
load_dotenv(dotenv_path=env_path)

# Load the database configuration
POSTGRES_DB_HOST = os.getenv('POSTGRES_DB_HOST')
POSTGRES_DB_USER = os.getenv('POSTGRES_DB_USER')
POSTGRES_DB_PASSWORD = os.getenv('POSTGRES_DB_PASSWORD')
POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME')
POSTGRES_DB_PORT = int(os.getenv('POSTGRES_DB_PORT'))
POSTGRES_DB_ENGINE = os.getenv('POSTGRES_DB_ENGINE')
DB_BATCH_SIZE = int(os.getenv('DB_BATCH_SIZE'))

from dependencies.toml import TomlHandler
from dependencies.logger import CustomLogger
from database.connections import DatabaseConnection, ConnectionStringBuilder

import utils.argparsers as arg_parsers

from database.models.training import TrainingConfigurationTable

from loaders.preprocessing import EngineeredFeaturesManager
from loaders.utils import LastProcessedDataManager
from loaders.training import ModelVersioningManager
from loaders.predictions import PredictionsDataManager

from utils.helpers import shift_date_by_window
from predictions.simulate_realtime_preds import SimulateRealTimePredictions


@dataclass
class RealTimePredictions:
    """
    Implements the real-time predictions for the autovalidation system.
    """
    logger: CustomLogger

    def set_database_engines(self) -> None:
        """
        Sets the database engines to load and save the data.
        """
        postgres_connection_string = ConnectionStringBuilder()(
                connection_type=POSTGRES_DB_ENGINE,
                user_name=POSTGRES_DB_USER,
                password=POSTGRES_DB_PASSWORD,
                host=POSTGRES_DB_HOST,
                database_name=POSTGRES_DB_NAME,
                port=POSTGRES_DB_PORT
            )

        self.postgres_connect = DatabaseConnection().connect(postgres_connection_string)

    def data_loading(self,
            start_date: str,
            end_date: str,
            latitude: str,
            longitude: str,
            variable_codes: list[str],
        ) -> pd.DataFrame:
        
        engineered_features_loader = EngineeredFeaturesManager(
            connection=self.postgres_connect,
            logger=self.logger,
        )

        processed_data = pd.DataFrame()
        df = engineered_features_loader.load(
            start_date=start_date,
            end_date=end_date,
            latitude=latitude,
            longitude=longitude,
            filter_variables=variable_codes,
        )
        # ----------------- DEBUGGING -----------------
        self.logger.debug(f"Dataframe columns: {df.columns}")
        self.logger.debug(f"Dataframe head: {df.head()}")
        # ---------------------------------------------
        try:
            preprocessed_df = engineered_features_loader.process_data_loading(df)
        except Exception as e:
            self.logger.error(f"Error loading the data: {e}")
            
        processed_data = pd.concat([processed_data, preprocessed_df], axis=1)

        if processed_data.isnull().any().any():
            processed_data.interpolate(method='time', inplace=True)
            processed_data = processed_data.bfill().ffill()
            processed_data = processed_data.fillna(0)
            processed_data = processed_data.dropna(how='all')
        
        processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]
        # ----------------- DEBUGGING -----------------
        self.logger.debug(f"Processed data columns: {processed_data.columns}")
        self.logger.debug(f"Processed data head: {processed_data.head()}")
        # ---------------------------------------------
        return processed_data

    def create_variables(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Unpacks the variables from the dataframe and returns a dictionary.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe with the parameters used to train models.

        Returns:
        --------
        dict[str, Any]
            Dictionary with the parameters used to train models.
        """
        results_dict = {}
        
        for param in df.parameter_code:
            value = df.loc[(df['parameter_code'] == param), 'parameter_value'].iloc[0]
            if any(x in param for x in ['date_frequency', 'direction', 'early_stop_mode', 'model_metric']):
            # if param in ['date_frequency', 'direction', 'early_stop_mode', 'model_metric']:
                results_dict[param] = value
            else:
                results_dict[param] = literal_eval(value)
        return results_dict

    def predict(self,
            realtime_preds_params: dict[str, Any],
            df: pd.DataFrame,
            latitude: float,
            longitude: float,
            variable_code: str
        ) -> pd.DataFrame:
        model_loader = ModelVersioningManager(
            connection=self.postgres_connect,
            logger=self.logger,
        )

        regression_models = model_loader.load_regression_models(
            latitude=latitude,
            longitude=longitude,
            target_variable=variable_code,
            model_type=realtime_preds_params['model_type'],
        )
        
        real_time_preds = SimulateRealTimePredictions(
            latitude=latitude,
            longitude=longitude,
            target_variable=variable_code,
            logger=self.logger
        )

        return real_time_preds.predict_next_values(
            df=df,
            validation_threshold=realtime_preds_params['iqr_threshold'],
            accuracy_threshold=realtime_preds_params['forecast_acc_threshold'],
            regression_models=regression_models
        )

    def process_preds_df_api(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("The dataframe is empty.")
        
        df['target_variable'] = df['target_variable'].str.replace('valor_', '')

        df_pivot = df.pivot(
            index='timestamp',
            columns='target_variable',
            values=[
                'real_value', 'predictions', 'data_quality_percentage',
                'validation_boolean', 'suggested_value'
            ]	
        )
        df_pivot.columns = [
            '_'.join(col).strip() for col in df_pivot.columns.values
        ]

        df_pivot.index.name = None
        # ----------------- DEBUGGING -----------------
        self.logger.debug(f"Pivot df columns: {df_pivot.columns}")
        self.logger.debug(f"Pivot df index head: {df_pivot.index[:5]}")
        self.logger.debug(f"Pivot df index start date: {df_pivot.index[0]} | end date: {df_pivot.index[-1]}")
        # ---------------------------------------------
        return df_pivot

    def get_date(self,
            start: bool,
            realtime_preds_params: dict[str, Any],
            last_date_loader: LastProcessedDataManager
        ) -> str:
        date = None
        if start:
            date = self._get_last_processed_date(
                realtime_preds_params=realtime_preds_params,
                data_type='data_types_predictions',
                last_date_loader=last_date_loader,
            )
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"Last date for the predictions: {date}")
            # ---------------------------------------------

            if date is None:
                date = self._get_last_processed_date(
                    realtime_preds_params=realtime_preds_params,
                    data_type='data_types_engineered',
                    last_date_loader=last_date_loader,
                )
                date_str = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')

                date_str = shift_date_by_window(
                    date_str=date_str,
                    window=realtime_preds_params['prediction_start_time_window'],
                    direction=realtime_preds_params['prediction_direction'],
                    date_frequency=realtime_preds_params['prediction_date_frequency']
                )
            else:
                date_str = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"Start date is: {date}")
            # ---------------------------------------------

        else:
            date = self._get_last_processed_date(
                realtime_preds_params=realtime_preds_params,
                data_type='data_types_engineered',
                last_date_loader=last_date_loader,
            )
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"Last date for the engineered features: {date}")
            # ---------------------------------------------

            if date is None:
                error_msg = "Last date cannot be None.\nCanceling script to " + \
                            "wait for processing of engineered features or " + \
                            "availability of the database."
                raise ValueError(error_msg)
            else:
                date_str = datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"Last date is: {date}")
            # ---------------------------------------------

        return date_str

    def _get_last_processed_date(self,
            realtime_preds_params: dict[str, Any],
            data_type: str,
            last_date_loader: LastProcessedDataManager
        ) -> Optional[datetime]:
        return last_date_loader.load(
            data_type=realtime_preds_params[data_type]
        )

    def create_start_end_dates(self,
            start_date: Optional[str],
            end_date: Optional[str],
            realtime_preds_params: dict[str, Any],
            last_date_loader: LastProcessedDataManager,
        ) -> tuple[Optional[str], Optional[str]]:
        """
        Creates the start and end dates to load the data from Ecodata if they
        are not provided.
        """
        if start_date in [None, 'None']:
            start_date = self.get_date(
                start=True,
                realtime_preds_params=realtime_preds_params,
                last_date_loader=last_date_loader,
            )

        if end_date in [None, 'None']:
            end_date = self.get_date(
                start=False,
                realtime_preds_params=realtime_preds_params,
                last_date_loader=last_date_loader,
            )

        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

        time_frequency_numbers = list(
            map(int, re.findall(r'\d+', realtime_preds_params['prediction_date_frequency']))
        )
        if (end_datetime - start_datetime) <= pd.Timedelta(time_frequency_numbers[0]*3, unit='min'):
            start_datetime = start_datetime - pd.Timedelta(time_frequency_numbers[0]*4, unit='min')
            start_date = datetime.strftime(start_datetime, '%Y-%m-%d %H:%M:%S')

        return start_date, end_date

    def send_to_api(self,
            predictions_df: pd.DataFrame,
            location_code: str,
            target_variables: list[str]
        ) -> None:
        for variable in target_variables:
            variable_code = variable.replace(f'{location_code}/', '')
            self.api_clients[variable_code]['api_manager'].save(
                predictions_df, self.api_clients[variable_code]['pred_config_json'])

def main(
        logger: CustomLogger,
        start_date: Optional[str],
        end_date: Optional[str],
        latitude: str,
        longitude: str,
        training_params: dict[str, Any],
        realtime_preds_params: dict[str, Any],
    ):
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'training_params': training_params,
        'realtime_preds_params': realtime_preds_params,
    }
    # ----------------- DEBUGGING -----------------
    logger.debug("Arguments used to run the script:")
    # ---------------------------------------------
    for key, value in params.items():
        # ----------------- DEBUGGING -----------------
        logger.debug(f"{key}: {value}")
        # ---------------------------------------------

    realtime_preds = RealTimePredictions(logger=logger)
    realtime_preds.set_database_engines()
        
    training_config_loader = EngineeredFeaturesManager(
            connection=realtime_preds.postgres_connect,
            logger=logger,
        )
    session = training_config_loader.Session()
    training_config = session.query(TrainingConfigurationTable).all()
    
    train_pred_params = pd.DataFrame()
    for config in training_config:
        params_df = pd.DataFrame(config.__dict__.items(), columns=['parameter_code', 'parameter_value'])
        params_df = params_df[params_df['parameter_code'] != '_sa_instance_state']
        params_df = params_df.pivot_table(columns='parameter_code', values='parameter_value', aggfunc='first')
        train_pred_params = pd.concat([train_pred_params, params_df], axis=0)
    train_pred_params.reset_index(drop=True, inplace=True)
    
    training_params_df = train_pred_params.loc[train_pred_params.stage == 'training']
    prediction_params_df = train_pred_params.loc[train_pred_params.stage == 'prediction']

    train_params_dict = realtime_preds.create_variables(df=training_params_df)
    training_params.update(train_params_dict)
    
    pred_params_dict = realtime_preds.create_variables(df=prediction_params_df)
    realtime_preds_params.update(pred_params_dict)
    
    model_versioning_loader = ModelVersioningManager(
        connection=realtime_preds.postgres_connect,
        logger=logger,
    )
    models = model_versioning_loader.get_best_model_from_date_range(
        model_type=realtime_preds_params['model_type'],
        latitude=latitude,
        longitude=longitude,
        columns=['model_feature_names', 'target_variable']
    )
    
    for model in models:
        variable_codes = literal_eval(model[0])
        target_variable = model[1]
    
    last_date_loader = LastProcessedDataManager(
        connection=realtime_preds.postgres_connect,
        logger=realtime_preds.logger,
    )

    start_date, end_date = realtime_preds.create_start_end_dates(
        start_date=start_date,
        end_date=end_date,
        realtime_preds_params=realtime_preds_params,
        last_date_loader=last_date_loader,
    )
    # ----------------- DEBUGGING -----------------
    logger.debug(f"The start and end dates are: {start_date} | {end_date}")
    logger.debug("Load the data to perform predictions...")
    # ---------------------------------------------
    df = realtime_preds.data_loading(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        variable_codes=variable_codes,
    )

    # ----------------- DEBUGGING -----------------
    logger.debug(f"Check null values: {df.info()}")
    logger.debug("Making predictions...")
    # ---------------------------------------------
    
    print(f"Predicting for {target_variable}...")
        
    preds_df = realtime_preds.predict(
        realtime_preds_params=realtime_preds_params, 
        df=df,
        latitude=latitude,
        longitude=longitude,
        variable_code=target_variable
    )
    # ----------------- DEBUGGING -----------------
    logger.debug(preds_df.info())
    logger.debug(preds_df.head())
    
    for col in preds_df.columns:
        if 'predictions' in col:
            preds_df[col] = np.clip(preds_df[col], 0, None)
            preds_df[col] = preds_df[col].rolling(window=100).mean()
            preds_df[col] = preds_df[col].bfill().ffill()
    
    logger.debug("Saving predictions to the database...")
    # ---------------------------------------------

    realtime_preds_loader = PredictionsDataManager(
        db_connection=realtime_preds.postgres_connect,
        logger=logger,
    )
    
    realtime_preds_loader.save(
        df=preds_df,
        batch_size=realtime_preds_params['prediction_batch_size']
    )
    # ----------------- DEBUGGING -----------------
    logger.debug("Predictions saved to the database...")

    logger.debug("Saving the last date processed to the LastProcessedDataTable...")
    # ---------------------------------------------
    last_date_saver = LastProcessedDataManager(
        realtime_preds.postgres_connect,
        logger,
    )
    last_date_saver.save(
        latitude=latitude,
        longitude=longitude,
        data_type=realtime_preds_params['data_types_predictions'],
        timestamp=datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    )

    processed_df = realtime_preds.process_preds_df_api(preds_df)
    # ----------------- DEBUGGING -----------------
    logger.debug(processed_df.columns)
    logger.debug("Saving predictions to the API...")
    # ---------------------------------------------
    
    realtime_preds.save(
        df=processed_df,
        batch_size=DB_BATCH_SIZE,
    )
    
    realtime_preds.postgres_connect.close()
    
    return processed_df


if __name__ == "__main__":
    start_time = time.time()

    parser = arg_parsers.get_parser()
    args = parser.parse_args()
    
    CONFIG_DICT = TomlHandler("config.toml").load()
    LOGGER_CONFIG = TomlHandler("logger.toml").load()

    training_params = CONFIG_DICT.get('realtime_training', {})
    realtime_preds_params = CONFIG_DICT.get('realtime_predictions', {})

    filename = Path(__file__).resolve().stem
    logger = CustomLogger(config=LOGGER_CONFIG, logger_name=filename).setup()

    _ = main(
        logger=logger,
        start_date=args.start_date,
        end_date=args.end_date,
        latitude=args.latitude,
        longitude=args.longitude,
        training_params=training_params,
        realtime_preds_params=realtime_preds_params,
    )

    end_time = time.time()
    # ----------------- DEBUGGING -----------------
    logger.debug(f"Total time: {end_time - start_time} seconds.")
    # ---------------------------------------------