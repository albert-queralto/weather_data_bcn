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

from dotenv import load_dotenv
env_path = MAIN_PATH / '.env'
load_dotenv(dotenv_path=env_path)

SECRET_KEY = os.getenv('SECRET_KEY')

from routers import preprocessing

logger = logging.getLogger(__name__)

app = FastAPI(
    title="API",
    description="API",
    version="0.0.1",
    # root_path="/api",
    openapi_tags=[
        {
            "name": "preprocessing",
            "description": "Preprocessing data",
        }
    ],
)

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain(
#     MAIN_PATH / "cert.pem",
#     MAIN_PATH / "key.pem"
# )

# ROOT = Path(__file__).parents[2]

# ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",")

# if ORIGINS:
app.add_middleware(
    SessionMiddleware,
    # allow_origins=ORIGINS,
    # allow_credentials=True,
    # allow_methods=["*"],
    # allow_headers=["*"],
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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=4000, timeout_keep_alive=600) # ssl=ssl_context