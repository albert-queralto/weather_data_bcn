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
from abc import ABC, abstractmethod

MAIN_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(MAIN_PATH))

from dotenv import load_dotenv
env_path = MAIN_PATH / '.env'
load_dotenv(dotenv_path=env_path)

POSTGRES_DB_HOST = os.getenv('POSTGRES_DB_HOST')
POSTGRES_DB_USER = os.getenv('POSTGRES_DB_USER')
DB_PASSWORD_POSTGRES = os.getenv('DB_PASSWORD_POSTGRES')
POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME')
POSTGRES_DB_PORT = int(os.getenv('POSTGRES_DB_PORT'))
POSTGRES_DB_ENGINE = os.getenv('POSTGRES_DB_ENGINE')
DB_BATCH_SIZE = int(os.getenv('DB_BATCH_SIZE'))

from utils.file_handlers import TomlHandler
from utils.custom_logger import CustomLogger
from loaders.api import OpenMeteoDataManager

from utils.helpers import (
    shift_date_by_window, 
    get_class_methods_exclude_dunder,
    get_classes_by_string_name,
)

from utils.seasonal_features import CreateSeasonalFeatures

class Preprocessor:
    """
    Implements the methods to treat outliers and null values in the data used by
    the autovalidation system. Loads the data from the database and saves the
    processed data also in the database.
    """
    logger: CustomLogger
    
    def __init__(self, logger: CustomLogger):
        self.logger = logger

    def load_data(self, 
            latitude: float, 
            longitude: float, 
            start_date: str, 
            end_date: str
        ) -> pd.DataFrame:
        openmeteo = OpenMeteoDataManager(logger=self.logger)
        return openmeteo.load(latitude, longitude, start_date, end_date)


if __name__ == "__main__":
    start_time = time.time()
    
    CONFIG_DICT = TomlHandler("config.toml").load()
    LOGGER_CONFIG = TomlHandler("logger.toml").load()

    filename = Path(__file__).resolve().stem
    logger = CustomLogger(config_dict=LOGGER_CONFIG, logger_name=filename).setup_logger()
    
    preprocessor = Preprocessor(logger)
    df = preprocessor.load_data(41.389, 2.159, "2015-10-11", "2023-03-29")
