"""
Example commands:
---------
- python "/root/home/backend/training/main.py" -lat 41.389 -lon 2.159 -sd "2015-10-11" -ed "2024-10-12"
"""
import os
import sys
import time
import pandas as pd
from ast import literal_eval

from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

# Add directory to import custom modules
MAIN_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(MAIN_PATH))

# Load env file
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


# Import TOML configuration file
from dependencies.toml import TomlHandler

# Import database connection
from database.connections import DatabaseConnection, ConnectionStringBuilder

# Import custom logger
from dependencies.logger import CustomLogger

# Import argparser
import utils.argparsers as arg_parsers

# Import database tables
from database.models.training import TrainingConfigurationTable

# Import data loaders
from loaders.training import (
    TrainingConfigurationManager,
    ModelVersioningManager,
)
from loaders.preprocessing import EngineeredFeaturesManager
from loaders.utils import LastProcessedDataManager

# Import the modules to train models
from utils.model_helper_functions import check_trained_models
from training.model_trainer import ModelTrainer, DataConfig, TrainingConfig, ModelConfig, DataPreprocessor

# Import predictive models
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


@dataclass
class TrainModels:
    logger: CustomLogger

    def set_database_engines(self) -> None:
        postgres_connection_string = ConnectionStringBuilder()(
                connection_type=POSTGRES_DB_ENGINE,
                user_name=POSTGRES_DB_USER,
                password=POSTGRES_DB_PASSWORD,
                host=POSTGRES_DB_HOST,
                database_name=POSTGRES_DB_NAME,
                port=POSTGRES_DB_PORT
            )

        self.postgres_connect = DatabaseConnection().connect(postgres_connection_string)
        self.engineered_features_loader = EngineeredFeaturesManager(
            connection=self.postgres_connect,
            logger=self.logger,
        )

    def data_loading(self,
            start_date: str,
            end_date: str,
            latitude: str,
            longitude: str,
            variable_codes: list[str],
        ) -> pd.DataFrame:
        processed_data = pd.DataFrame()
        for variable_code in variable_codes:
            df = self.engineered_features_loader.load(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                filter_variables=variable_code,
            )

            preprocessed_df = self.engineered_features_loader.process_data_loading(df)
            processed_data = pd.concat([processed_data, preprocessed_df], axis=1)

            if processed_data.isnull().any().any():
                processed_data.interpolate(method="time", inplace=True)
                processed_data = processed_data.bfill().ffill()
                processed_data = processed_data.fillna(0)
                processed_data = processed_data.dropna(how='all')

        processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]
        return processed_data

    def create_variables(self, df: pd.DataFrame) -> dict[str, Any]:
        results_dict = {}
        
        for param in df.parameter_code:
            value = df.loc[(df['parameter_code'] == param), 'parameter_value'].iloc[0]
            print(param, value, type(value))
            if any(x in param for x in ['date_frequency', 'model_metric', 'early_stop_mode', 'direction']):
                results_dict[param] = value
            else:
                results_dict[param] = literal_eval(value)
        return results_dict

    def model_training(self,
            model_manager: ModelVersioningManager,
            df: pd.DataFrame,
            latitude: str,
            longitude: str,
            target_variable: str,
            training_params: dict[str, Any],
        ) -> None:
        models = self._build_models(model_names=training_params['model_names'])        
        model_params = self._build_model_params_dict(training_params=training_params)

        data_config = DataConfig(
            df=df,
            target_variable=target_variable,
            latitude=latitude,
            longitude=longitude,
            test_size=training_params['test_size'],
            batch_size=DB_BATCH_SIZE,
        )

        data_preprocessor = DataPreprocessor(data_config)

        training_config = TrainingConfig(
            cv_splits=training_params['cv_splits'],
            n_iter=training_params['n_iter'],
            early_stop_patience=training_params['early_stop_patience'],
            early_stop_mode=training_params['early_stop_mode'],
            early_stop_threshold=training_params['early_stop_threshold'],
            n_study_runs=training_params['n_study_runs']
        )
        
        model_config = ModelConfig(
            models=models,
            params=model_params,
            model_type=training_params['model_type'],
            model_metric_name=training_params['model_metric'],
            forecast_accuracy_threshold=training_params['forecast_acc_threshold']
        )

        model_trainer = ModelTrainer(
            model_manager=model_manager,
            logger=self.logger,
            data_preprocessor=data_preprocessor,
            training_config=training_config,
            model_config=model_config            
        )

        model_trainer.train()

    def _build_models(self, model_names: list[str]) -> dict[str, Any]:
        models = {}

        for name in model_names:
            if name in ['linear_regression', 'polynomial']:
                models[name] = LinearRegression()
            elif name == 'xgboost_regression':
                models[name] = XGBRegressor(
                    objective='reg:squarederror',
                    booster='gbtree',
                    eval_metric='rmse'
                )
            elif name == 'mlpregressor':
                models[name] = MLPRegressor()
        return models

    def _build_model_params_dict(self,
            training_params: dict[str, Any]
        ) -> dict[str, dict[str, Any]]:
        model_names = training_params.get('model_names', [])

        return {
            model: value
            for key, value in training_params.items()
            for model in model_names
            if model in key
        }

    def create_start_date(self, 
            start_date: Optional[str],
            latitude: float,
            longitude: float,
        ) -> Optional[str]:
        if start_date in [None, 'None']:
            return self.engineered_features_loader.get_min_date(latitude, longitude)
        return start_date

    def create_end_date(self,
            end_date: Optional[str],
            latitude: float,
            longitude: float,
        ) -> Optional[str]:
        if end_date in [None, 'None']:
            return self.engineered_features_loader.get_max_date(latitude, longitude)
        return end_date
    
    def load_training_configuration(self, training_params: dict[str, Any]) -> dict[str, Any]:
        model_manager = TrainingConfigurationManager(self.postgres_connect, self.logger)
        session = model_manager.Session()
        config = session.query(TrainingConfigurationTable).all()
        
        training_params_df = pd.DataFrame()
        for row in config:
            params = pd.DataFrame(row.__dict__.items(), columns=['parameter_code', 'parameter_value'])
            params = params[params['parameter_code'] != '_sa_instance_state']
            params = params.pivot_table(columns='parameter_code', values='parameter_value', aggfunc='first')
            training_params_df = pd.concat([training_params_df, params], axis=0)
        training_params_df.reset_index(drop=True, inplace=True)
        
        params_dict = self.create_variables(df=training_params_df)
        training_params.update(params_dict)
        return training_params

    def get_training_variables(self) -> list[str]:
        variables = self.engineered_features_loader.get_variables()
        target_variable = [var for var in variables if 'precipitation' in var][0]
        return variables, target_variable

def main(
        logger: CustomLogger,
        latitude: float,
        longitude: float,
        start_date: Optional[str],
        end_date: Optional[str],
        null_filling_params: dict[str, Any],
        training_params: dict[str, Any],
    ) -> None:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        'start_date': start_date,
        'end_date': end_date,
        'null_filling_params': null_filling_params,
        'training_params': training_params,
    }
    logger.debug("Arguments used to run the script:")
    for key, value in params.items():
        logger.debug(f"{key}: {value}")

    train_models = TrainModels(logger)
    train_models.set_database_engines()

    training_params = train_models.load_training_configuration(training_params)
    variables, target_variable = train_models.get_training_variables()

    start_date = train_models.create_start_date(start_date, latitude, longitude)
    end_date = train_models.create_end_date(end_date, latitude, longitude)

    logger.debug(f"The start and end dates are: {start_date} | {end_date}")
    logger.debug("Loading data...")

    df = train_models.data_loading(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        variable_codes=variables,
    )

    logger.debug(f"Data loaded with shape {df.shape}")
    logger.debug(f"Are there nulls?\n{df.isnull().sum()}")

    model_manager = ModelVersioningManager(
        connection=train_models.postgres_connect,
        logger=logger,
    )

    trained_bool, _ = check_trained_models(
        latitude=str(latitude),
        longitude=str(longitude),
        model_type=training_params['model_type'],
        target_variable=target_variable,
        train_models_older_than=training_params['train_models_older_than'],
        col_prefix=training_params['column_prefix'],
        logger=logger,
        model_loader=model_manager
    )

    if trained_bool is True:
        # ----------------- DEBUGGING -----------------
        logger.debug("Training and testing models for data prediction...")
        logger.debug(f"Train models older than {training_params['train_models_older_than']} days...")
        # ---------------------------------------------
        train_models.model_training(
            model_manager=model_manager,
            df=df,
            latitude=latitude,
            longitude=longitude,
            target_variable=target_variable,
            training_params=training_params
        )

        # Save the last date processed to the LastProcessedDataTable
        last_date_loader = LastProcessedDataManager(
            connection=train_models.postgres_connect,
            logger=logger,
        )

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        last_date_loader.save(
            latitude=latitude,
            longitude=longitude,
            data_type=training_params['data_type'],
            timestamp=end_date
        )

    train_models.postgres_connect.close()


if __name__ == "__main__":
    # Start time
    start_time = time.time()

    # Get the parser for the command line arguments
    parser = arg_parsers.get_parser()

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load the configuration file into a dictionary
    CONFIG_DICT = TomlHandler("config.toml").load()
    LOGGER_CONFIG = TomlHandler("logger.toml").load()

    # Set the specific dictionaries for the different parameters
    null_filling_params = CONFIG_DICT.get('null_filling_params', {})
    training_params = CONFIG_DICT.get('realtime_training', {})
    
    # Set the logger for the current module
    filename = Path(__file__).resolve().stem
    logger = CustomLogger(config=LOGGER_CONFIG, logger_name=filename).setup()

    main(
        logger=logger,
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date, # e.g., '2022-01-01 00:00:00'
        end_date=args.end_date, # e.g., '2022-12-31 23:59:59'
        null_filling_params=null_filling_params,
        training_params=training_params,
    )

    # End time
    end_time = time.time()
    # ----------------- DEBUGGING -----------------
    logger.debug(f"Total time: {end_time - start_time} seconds.")
    # ---------------------------------------------