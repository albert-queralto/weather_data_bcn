"""
Example command line:
---------
- python "/root/home/Preprocessing/main.py" -lat -lon -sd "2015-10-11 00:00:00" -ed "2023-03-29 05:30:00"
"""

import os
import re
import sys
import time
import pandas as pd
from ast import literal_eval
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import sqlalchemy
from abc import ABC, abstractmethod

MAIN_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(MAIN_PATH))

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

from utils.helpers import (
    get_class_methods_exclude_dunder,
    get_classes_by_string_name,
)

from database.connections import DatabaseConnection, ConnectionStringBuilder
from utils.seasonal_features import CreateSeasonalFeatures
from loaders.preprocessing import EngineeredFeaturesManager


@dataclass
class Preprocessor:
    """
    Implements the methods to treat outliers and null values in the data used by
    the autovalidation system. Loads the data from the database and saves the
    processed data also in the database.
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


if __name__ == "__main__":
    latitude = 41.389
    longitude = 2.159
    start_date = "2023-03-28"
    end_date = "2023-03-29"
    start_time = time.time()
    
    CONFIG_DICT = TomlHandler("config.toml").load()
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
    
    preprocessor = Preprocessor(logger, postgres_connect)
    logger.debug("Loading the data...")
    df = preprocessor.load_data(latitude, longitude, start_date, end_date)
    
    logger.debug("Filling null values...")
    null_filled_df = preprocessor.null_values_filling(df)
    
    logger.debug("Creating seasonal features...")
    seasonal_feats_creator, seasonal_methods = preprocessor.create_seasonal_variables()
    seasonal_df = seasonal_feats_creator(null_filled_df, seasonal_methods)

    logger.debug("Saving the data to the database...")
    preprocessor.save_data_to_db(seasonal_df, latitude, longitude, DB_BATCH_SIZE)