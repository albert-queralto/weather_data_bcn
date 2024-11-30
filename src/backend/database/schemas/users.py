from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime
from typing import Optional


class UsersSchema(BaseModel):
    email: EmailStr
    password: str
    role: str
    created_date: Optional[datetime] = None
    update_date: Optional[datetime] = None
    
    model_config = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    email: EmailStr
    role: str
    created_date: datetime
    update_date: datetime
    token: TokenResponse
    
    class Config:
        json_schema_extra = {
            "examples": [{
                "email": "user@domain.org",
                "role": "admin",
                "created_date": "2022-01-01 00:00:00",
                "update_date": "2022-01-01 00:00:00",
                "token": {
                    "access_token": "token",
                    "token_type": "bearer"
                }
            }]
        }