"""
Example command line:
---------
- python "/root/home/backend/preprocessing/main.py" -lat 41.389 -lon 2.159 -sd "2015-10-11" -ed "2024-10-12"
"""

import os
import sys
import time
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import sqlalchemy

MAIN_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(MAIN_PATH))

import utils.argparsers
from dotenv import load_dotenv
env_path = MAIN_PATH / '.env'
load_dotenv(dotenv_path=env_path)

POSTGRES_DB_HOST = os.getenv('POSTGRES_DB_HOST')
POSTGRES_DB_USER = os.getenv('POSTGRES_DB_USER')
POSTGRES_DB_PASSWORD = os.getenv('POSTGRES_DB_PASSWORD')
POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME')
POSTGRES_DB_PORT = int(os.getenv('POSTGRES_DB_PORT'))
POSTGRES_DB_ENGINE = os.getenv('POSTGRES_DB_ENGINE')
DB_BATCH_SIZE = int(os.getenv('DB_BATCH_SIZE'))

from utils.file_handlers import TomlHandler
from utils.custom_logger import CustomLogger
from loaders.api import OpenMeteoDataManager

from utils.helpers import get_class_methods_exclude_dunder, shift_date_by_window

from database.connections import DatabaseConnection, ConnectionStringBuilder
from utils.seasonal_features import CreateSeasonalFeatures
from loaders.preprocessing import EngineeredFeaturesManager
from loaders.utils import LastProcessedDataManager


@dataclass
class Preprocessor:
    """
    Implements the methods to preprocess the data.
    """
    logger: CustomLogger
    connection: sqlalchemy.engine.Connection

    def load_data(self, 
            latitude: float, 
            longitude: float, 
            start_date: str, 
            end_date: str
        ) -> pd.DataFrame:
        openmeteo = OpenMeteoDataManager(logger=self.logger)
        df = openmeteo.load(latitude, longitude, start_date, end_date)
        df.set_index("date", inplace=True)
        return df

    def null_values_filling(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"{df.isnull().sum()}")
        df.interpolate(method='time', inplace=True)
        self.logger.debug(f"{df.isnull().sum()}")
        return df

    def create_seasonal_variables(self) -> tuple[CreateSeasonalFeatures, list[Any], list[str]]:
        seasonal_feats_creator = CreateSeasonalFeatures()
        seasonal_methods = [
            getattr(seasonal_feats_creator, method)
            for method in get_class_methods_exclude_dunder(seasonal_feats_creator)
            if all(
                x not in method
                for x in ["_get_season_date", "_get_day_of_month",]
            )
        ]

        return seasonal_feats_creator, seasonal_methods

    def save_data_to_db(self, df: pd.DataFrame, latitude: float, longitude: float, batch_size: int) -> None:
        saver = EngineeredFeaturesManager(self.connection, self.logger)
        
        df = saver.process_data_saving(df)    
        df["latitude"] = latitude
        df["longitude"] = longitude
        saver.save(df, batch_size)

        last_date_saver = LastProcessedDataManager(self.connection, self.logger)
        last_date_saver.save(latitude, longitude, "preprocessing", df['timestamp'].max())
        self.logger.debug(f"Last date processed {df['timestamp'].max()} saved to the database...")

    def create_start_date(self,
            start_date: Optional[str],
            start_time_window: int,
            direction: str,
            date_frequency: str,
            last_date_processed: Optional[str]
        ) -> Optional[str]:
        if start_date in [None, 'None']:
            date = datetime.now().strftime("%Y-%m-%d")
            if last_date_processed:
                self.logger.debug(f"Last date: {last_date_processed}")
                date = datetime.strftime(last_date_processed, "%Y-%m-%d")
            return shift_date_by_window(date, start_time_window, direction, date_frequency)
        return start_date

    def create_end_date(self,
            end_date: Optional[str],
            
        ) -> tuple[Optional[str], Optional[str]]:
        if end_date in [None, 'None']:
            end_date = datetime.now().strftime("%Y-%m-%d")
        return end_date

def main(
        logger: CustomLogger,
        connection: sqlalchemy.engine.Connection,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        preprocessing_config: dict
    ):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "preprocessing_config": preprocessing_config
    }
    logger.debug("Preprocessing the data with the following parameters:")
    for k, v in params.items():
        logger.debug(f"{k}: {v}")
    
    last_date_loader = LastProcessedDataManager(
        connection=connection,
        logger=logger
    )

    last_date = last_date_loader.load(latitude, longitude, preprocessing_config['data_type'])
    preprocessor = Preprocessor(logger, connection)

    start_time = preprocessor.create_start_date(
        start_date=start_date,
        start_time_window=preprocessing_config['start_time_window'],
        direction=preprocessing_config['direction'],
        date_frequency=preprocessing_config['date_frequency'],
        last_date_processed=last_date
    )
    end_time = preprocessor.create_end_date(end_date)
    logger.debug(f"The start and end dates are: {start_time} | {end_time}")

    logger.debug("Loading the data...")
    df = preprocessor.load_data(latitude, longitude, start_time, end_time)
    
    logger.debug("Filling null values...")
    null_filled_df = preprocessor.null_values_filling(df)
    
    logger.debug("Creating seasonal features...")
    seasonal_feats_creator, seasonal_methods = preprocessor.create_seasonal_variables()
    seasonal_df = seasonal_feats_creator(null_filled_df, seasonal_methods)

    logger.debug("Saving the data to the database...")
    preprocessor.save_data_to_db(seasonal_df, latitude, longitude, DB_BATCH_SIZE)


if __name__ == "__main__":
    start_time = time.time()

    parser = utils.argparsers.get_parser()
    args = parser.parse_args()

    CONFIG_DICT = TomlHandler("config.toml").load()
    preprocessing_config = CONFIG_DICT.get("preprocessing", {})
    LOGGER_CONFIG = TomlHandler("logger.toml").load()

    filename = Path(__file__).resolve().stem
    logger = CustomLogger(config_dict=LOGGER_CONFIG, logger_name=filename).setup_logger()
    
    postgres_connection_string = ConnectionStringBuilder()(
                connection_type=POSTGRES_DB_ENGINE,
                user_name=POSTGRES_DB_USER,
                password=POSTGRES_DB_PASSWORD,
                host=POSTGRES_DB_HOST,
                database_name=POSTGRES_DB_NAME,
                port=POSTGRES_DB_PORT
            )
    postgres_connect = DatabaseConnection().connect(postgres_connection_string)
    
    main(
        logger=logger,
        connection=postgres_connect,
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date,
        end_date=args.end_date,
        preprocessing_config=preprocessing_config
    )