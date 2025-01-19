"""Entry point into the server."""
import logging
import os
import sys
from pathlib import Path
import mimetypes
import uvicorn
import ssl

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles

MAIN_PATH = Path(__file__).resolve().parents[1]
BACKEND_PATH = Path(__file__).resolve().parents[0]
sys.path.extend([str(MAIN_PATH), str(BACKEND_PATH)])

from routers import (
    preprocessing,
    users,
    training,
    predictions
)

from dotenv import load_dotenv
env_path = MAIN_PATH / '.env'
load_dotenv(dotenv_path=env_path)

SECRET_KEY = os.getenv('SECRET_KEY')
logger = logging.getLogger(__name__)

def start_application():
    app = FastAPI(
        title="API",
        description="API",
        version="0.0.1",
        openapi_tags=[
            {
                "name": "preprocessing",
                "description": "Preprocessing data",
            }
        ],
    )

    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY
    )

    mimetypes.init()
    mimetypes.add_type('application/javascript', '.js')
    mimetypes.add_type('text/css', '.css')
    mimetypes.add_type('text/html', '.html')
    mimetypes.add_type('image/png', '.png')
    mimetypes.add_type('image/jpeg', '.jpeg')
    mimetypes.add_type('image/jpg', '.jpg')

    app.include_router(preprocessing.router)
    app.include_router(users.router)
    app.include_router(training.router)
    app.include_router(predictions.router)
    return app

app = start_application()