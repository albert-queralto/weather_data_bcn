import os
from pathlib import Path
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

MAIN_PATH = Path(__file__).parents[2]

from dotenv import load_dotenv
env_path = MAIN_PATH / ".env"
load_dotenv(env_path)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


def get_is_authenticated(request: Request):
    return request.session.get("is_authenticated", False)

async def get_current_user_role(request: Request):
    auth = HTTPBearer()
    
    try:
        credentials: HTTPAuthorizationCredentials = await auth(request)
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("role", "guest")
    except JWTError:
        raise HTTPException(
            status_code=403,
            detail="Invalid access token"
        )