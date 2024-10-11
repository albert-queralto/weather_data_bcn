from sqlalchemy import Column, String, Float, DateTime
from database.models.base import Base


class EngineeredFeaturesTable(Base):
    """
    Definition of the table that contains the engineered features.
    """
    __tablename__ = 'engineered_features'

    timestamp = Column(
        name = 'timestamp',
        type_= DateTime,
        primary_key = True,
        nullable = False,
        index = False
    )

    latitude = Column(
        name = 'latitude',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )
    
    longitude = Column(
        name = 'longitude',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )

    variable_code = Column(
        name = 'variable_code',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )

    value = Column(
        name = 'value',
        type_= Float,
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