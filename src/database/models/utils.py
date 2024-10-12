from sqlalchemy import Column, String, Float, DateTime
from database.models.base import Base

class LastProcessedDataTable(Base):
    """
    Definition of the table that contains the last processed data timestamp.
    """
    __tablename__ = 'last_processed_data'

    latitude = Column(
        name = 'latitude',
        type_= Float,
        primary_key = True,
        nullable = False,
        index = False
    )
    
    longitude = Column(
        name = 'longitude',
        type_= Float,
        primary_key = True,
        nullable = False,
        index = False
    )

    data_type = Column(
        name = 'data_type',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )

    timestamp = Column(
        name = 'timestamp',
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