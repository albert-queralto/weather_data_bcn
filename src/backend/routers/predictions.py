import time
import sys
from pathlib import Path
from pydantic import BaseModel

MAIN_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(MAIN_PATH))

from dotenv import load_dotenv
env_path = MAIN_PATH / '.devcontainer' / '.env'
load_dotenv(dotenv_path=env_path)

from dependencies.toml import TomlHandler
from dependencies.logger import CustomLogger
from models.predictions import Predictions, PredictionsResponse
from predictions.main import (
    main as predictions,
    RealTimePredictions
)
from modules.loaders.utils import PredictionsDataManager

from fastapi import APIRouter

router = APIRouter(
    tags=["Predictions"],
)

class Predictions(BaseModel):
    start_date: str
    end_date: str

CONFIG_DICT = TomlHandler("config.toml").load()
LOGGER_CONFIG = TomlHandler("logger.toml").load()
training_params = CONFIG_DICT.get('realtime_training', {})
realtime_preds_params = CONFIG_DICT.get('realtime_predictions', {})

filename = Path(__file__).resolve().stem
logger = CustomLogger(config_dict=LOGGER_CONFIG, logger_name=filename).setup_logger()

@router.post("/predict")
async def predict(params: Predictions):
    start_time = time.time()
    
    result = predictions(
        logger=logger,
        start_date=params.start_date,
        end_date=params.end_date,
        training_params=training_params,
        realtime_preds_params=realtime_preds_params,
    )

    # End time
    end_time = time.time()

    return {"result": result, "execution_time": end_time - start_time}

@router.get("/predictions")
async def get_predictions():
    connector = RealTimePredictions(logger)
    connector.set_database_engines()
    
    predictions_loader = PredictionsDataManager(connector.postgres_connect, logger)
    session = predictions_loader.Session()
    predictions = session.query(predictions_loader.model_table).all()
    
    predictions_list = []
    for prediction in predictions:
        predictions_list.append({
            "timestamp": prediction.timestamp,
            "model_date": prediction.model_date,
            "model_name": prediction.model_name,
            "model_type": prediction.model_type,
            "model_version": prediction.model_version,
            "source_name": prediction.source_name,
            "location_code": prediction.location_code,
            "target_variable": prediction.target_variable,
            "real_value": prediction.real_value,
            "predictions": prediction.predictions,
            "data_quality_percentage": prediction.data_quality_percentage,
            "validation_boolean": prediction.validation_boolean,
            "suggested_value": prediction.suggested_value,
            "update_date": prediction.update_date,
        })
    
    return {"predictions": predictions_list}

@router.post("/predictions")
async def post_predictions(params: PredictionsResponse):
    connector = RealTimePredictions(logger)
    connector.set_database_engines()
    
    predictions_loader = PredictionsDataManager(connector.postgres_connect, logger)
    session = predictions_loader.Session()
    predictions = session.query(predictions_loader.model_table).filter(
        predictions_loader.model_table.timestamp >= params.start_date,
        predictions_loader.model_table.timestamp <= params.end_date
    ).all()
    
    predictions_list = []
    for prediction in predictions:
        predictions_list.append({
            "timestamp": prediction.timestamp,
            "model_date": prediction.model_date,
            "model_name": prediction.model_name,
            "model_type": prediction.model_type,
            "model_version": prediction.model_version,
            "source_name": prediction.source_name,
            "location_code": prediction.location_code,
            "target_variable": prediction.target_variable,
            "real_value": prediction.real_value,
            "predictions": prediction.predictions,
            "data_quality_percentage": prediction.data_quality_percentage,
            "validation_boolean": prediction.validation_boolean,
            "suggested_value": prediction.suggested_value,
            "update_date": prediction.update_date,
        })
    
    return {"predictions": predictions}