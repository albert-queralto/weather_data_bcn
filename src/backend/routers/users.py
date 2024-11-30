import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

MAIN_PATH = Path(__file__).resolve().parents[3]
BACKEND_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(BACKEND_PATH))

from auth.hash_password import HashPassword
from auth.jwt_handler import create_access_token
from dependencies.auth import get_current_user_role
from dependencies.toml import TomlHandler
from dependencies.logger import CustomLogger

LOGGER_CONFIG = TomlHandler(f"logger.toml").load()
filename = Path(__file__).resolve().stem
logger = CustomLogger(config=LOGGER_CONFIG, logger_name=filename).setup()

from dotenv import load_dotenv
env_path = MAIN_PATH / ".env"
load_dotenv(env_path)

POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER")
POSTGRES_DB_PASSWORD = os.getenv("POSTGRES_DB_PASSWORD")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_DB_PORT = os.getenv("POSTGRES_DB_PORT")
POSTGRES_DB_ENGINE = os.getenv("POSTGRES_DB_ENGINE")


from database.connections import DatabaseConnection, ConnectionStringBuilder
from database.models.users import UsersTable
from database.schemas.users import UsersSchema, TokenResponse, UserResponse

def set_database_connection():
    connection_str = ConnectionStringBuilder()(
                connection_type=POSTGRES_DB_ENGINE,
                user_name=POSTGRES_DB_USER,
                password=POSTGRES_DB_PASSWORD,
                host=POSTGRES_DB_HOST,
                database_name=POSTGRES_DB_NAME,
                port=POSTGRES_DB_PORT
            ) 
    
    return DatabaseConnection().connect(connection_str)
    

router = APIRouter(
    tags=["Users"]
)

hash_password = HashPassword()

@router.post("/signup", response_model=UserResponse)
async def sign_user_up(
        user: UsersSchema,
    ) -> UserResponse:

    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()

    existing_user = session.query(UsersTable).filter(UsersTable.email == user.email).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists"
        )
    
    hashed_password = hash_password.create(user.password)
    user.password = hashed_password
    user.created_date = user.update_date = datetime.now(timezone.utc)
    
    new_user = UsersTable(**user.model_dump())
    session.add(new_user)
    session.commit()
    session.close()
    
    return JSONResponse(
        content={"message": "User created successfully"},
        status_code=status.HTTP_201_CREATED
    )

@router.post("/signin", response_model=UserResponse)
async def sign_user_in(
        form_data: UsersSchema
    ) -> UserResponse:
    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()
    
    existing_user = session.query(UsersTable).filter(UsersTable.email == form_data.email).first()

    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not exist"
        )
    
    if hash_password.verify(form_data.password, existing_user.password):
        access_token = create_access_token(existing_user.email, existing_user.role)
        token = TokenResponse(access_token=access_token, token_type="bearer")
        user_response = UserResponse(
            email=existing_user.email,
            role=existing_user.role,
            created_date=existing_user.created_date,
            update_date=existing_user.update_date,    
            token=token
        )
        return user_response
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

@router.get("/users")
async def get_all_users() -> dict[str, list[UserResponse]]:
    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()
    
    users = session.query(UsersTable).all()
    session.close()
    
    users_response = [UserResponse(
        email=user.email,
        role=user.role,
        created_date=user.created_date,
        update_date=user.update_date,
        token=TokenResponse(access_token="", token_type="")
    ) for user in users]
    
    return {"users": users_response}

@router.get("/users/{email}")
async def get_user_by_email(
        email: str,
    ) -> UserResponse:
    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()
    
    user = session.query(UsersTable).filter(UsersTable.email == email).first()
    session.close()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not exist"
        )
    
    user_response = UserResponse(
        email=user.email,
        role=user.role,
        created_date=user.created_date,
        update_date=user.update_date,
        token=TokenResponse(access_token="", token_type="")
    )
    
    return user_response

@router.put("/users/{email}")
async def update_user_password(
        user: UsersSchema,
    ) -> dict[str, str]:

    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()
    
    existing_user = session.query(UsersTable).filter(UsersTable.email == user.email).first()
    
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not exist"
        )
    
    hashed_password = hash_password.create(user.password)
    user.password = hashed_password
    # user.role = existing_user.role
    user.update_date = datetime.now(timezone.utc)
    user.created_date = existing_user.created_date
    session.query(UsersTable).filter(UsersTable.email == user.email).update(user.model_dump())
    session.commit()
    session.close()
    
    return JSONResponse(
        content={"message": "User updated successfully"},
        status_code=status.HTTP_200_OK
    )

@router.delete("/users/{email}")
async def delete_user_by_email(
        email: str,
    ) -> dict:
        
    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()
    
    existing_user = session.query(UsersTable).filter(UsersTable.email == email).first()
    
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not exist"
        )
    
    session.query(UsersTable).filter(UsersTable.email == email).delete()
    session.commit()
    session.close()
    
    return JSONResponse(
        content={"message": "User deleted successfully"},
        status_code=status.HTTP_200_OK
    )

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse(
        content={"message": "Logged out successfully"}, 
        status_code=status.HTTP_200_OK
    )
    
if __name__ == "__main__":
    db_connection = set_database_connection()
    Session = sessionmaker(bind=db_connection)
    session = Session()
    
    users = session.query(UsersTable).all()
    session.close()
    
    users_response = [UserResponse(
        email=user.email,
        role=user.role,
        created_date=user.created_date,
        update_date=user.update_date,
        token=TokenResponse(access_token="", token_type="")
    ) for user in users]
    
    print(users_response)
