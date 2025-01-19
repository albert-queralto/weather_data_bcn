import os
import sys
import time
from typing import Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

MAIN_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(MAIN_PATH))

from dependencies.toml import TomlHandler
from dependencies.logger import CustomLogger
from training.main import main

from fastapi import APIRouter

# Create the API router    
router = APIRouter(
    tags=["Model Training"],
)

class TrainingParams(BaseModel):
    latitude: str
    longitude: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

# Load the configuration file into a dictionary
CONFIG_DICT = TomlHandler("config.toml").load()
LOGGER_CONFIG = TomlHandler("logger.toml").load()

# Set the specific dictionaries for the different parameters
null_filling_params = CONFIG_DICT.get('null_filling_params', {})
training_params = CONFIG_DICT.get('realtime_training', {})

# Set the logger for the current module
filename = Path(__file__).resolve().stem
logger = CustomLogger(config=LOGGER_CONFIG, logger_name=filename).setup()


@router.post("/train")
async def train_model(params: TrainingParams):
    start_time = time.time()
    
    main(
        logger=logger,
        latitude=params.latitude,
        longitude=params.longitude,
        start_date=params.start_date,
        end_date=params.end_date,
        null_filling_params=null_filling_params,
        training_params=training_params,
    )
    
    end_time = time.time()
    
    return {"message": "Model trained successfully!", "execution_time": end_time - start_time}