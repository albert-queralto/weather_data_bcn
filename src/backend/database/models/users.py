from database.models.base import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship, backref


class UsersTable(Base):
    __tablename__ = 'users' 

    email = Column(
        name = 'email',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )
    
    password = Column(
        name = 'password',
        type_= String,
        primary_key = False,
        nullable = False,
        index = False
    )
    
    role = Column(
        name = 'role',
        type_= String,
        primary_key = False,
        nullable = False,
        index = False
    )

    created_date = Column(
        name = 'created_date',
        type_= DateTime,
        primary_key = False,
        nullable = True,
        index = False
    )

    update_date = Column(
        name = 'update_date',
        type_= DateTime,
        primary_key = False,
        nullable = True,
        index = False
    )