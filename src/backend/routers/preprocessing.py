import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

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
from database.connections import DatabaseConnection, ConnectionStringBuilder
from loaders.preprocessing import EngineeredFeaturesManager
from loaders.utils import LastProcessedDataManager
from preprocessing.main import Preprocessor

router = APIRouter(
    tags=["preprocessing"],
)

class PreprocessingRequest(BaseModel):
    latitude: str
    longitude: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

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

@router.post("/get_data")
async def get_preprocessing_data(request: PreprocessingRequest):
    loader = EngineeredFeaturesManager(postgres_connect, logger)
    df = loader.load(
        request.latitude, 
        request.longitude, 
        request.start_date, 
        request.end_date
    )
    return {"data": df.to_dict(orient="records")}
    
#     {
#   "latitude": 41.389,
#   "longitude": 2.159,
#   "start_date": "2024-10-01",
#   "end_date": "2024-10-02"
# }
    
@router.post("/preprocessing")
def preprocess_data(request: PreprocessingRequest):
    try:
        params = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "preprocessing_config": preprocessing_config
        }
        logger.debug("Preprocessing the data with the following parameters:")
        for k, v in params.items():
            logger.debug(f"{k}: {v}")
        
        last_date_loader = LastProcessedDataManager(
            connection=postgres_connect,
            logger=logger
        )

        last_date = last_date_loader.load(request.latitude, request.longitude, preprocessing_config['data_type'])
        preprocessor = Preprocessor(logger, postgres_connect)

        start_time = preprocessor.create_start_date(
            start_date=request.start_date,
            start_time_window=preprocessing_config['start_time_window'],
            direction=preprocessing_config['direction'],
            date_frequency=preprocessing_config['date_frequency'],
            last_date_processed=last_date
        )
        end_time = preprocessor.create_end_date(request.end_date)
        logger.debug(f"The start and end dates are: {start_time} | {end_time}")

        logger.debug("Loading the data...")
        df = preprocessor.load_data(request.latitude, request.longitude, start_time, end_time)
        
        logger.debug("Filling null values...")
        null_filled_df = preprocessor.null_values_filling(df)
        
        logger.debug("Creating seasonal features...")
        seasonal_feats_creator, seasonal_methods = preprocessor.create_seasonal_variables()
        seasonal_df = seasonal_feats_creator(null_filled_df, seasonal_methods)

        logger.debug("Saving the data to the database...")
        preprocessor.save_data_to_db(seasonal_df, request.latitude, request.longitude, DB_BATCH_SIZE)

        return {"status": "success", "message": "Data preprocessed and saved to the database."}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    loader = EngineeredFeaturesManager(postgres_connect, logger)
    df = loader.load(
        latitude="41.389", 
        longitude="2.159", 
        start_date="2024-10-01", 
        end_date="2024-10-02"
    )
    print(df.head())