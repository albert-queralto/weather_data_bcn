from sqlalchemy import Column, Integer, String, Float, DateTime
from database.models.base import Base

class PredictionsTable(Base):
    """
    Definition of the table that contains the predictions.
    """
    __tablename__ = 'predictions'

    timestamp = Column(
        name = 'timestamp',
        type_= DateTime,
        primary_key = True,
        nullable = False,
        index = False
    )

    model_date = Column(
        name = 'model_date',
        type_= DateTime,
        primary_key = False,
        nullable = False,
        index = False
    )

    model_name = Column(
        name = 'model_name',
        type_= String,
        primary_key = False,
        nullable = False,
        index = False
    )

    model_type = Column(
        name = 'model_type',
        type_= String,
        primary_key = False,
        nullable = False,
        index = False
    )

    model_version = Column(
        name = 'model_version',
        type_= Integer,
        primary_key = False,
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

    target_variable = Column(
        name = 'target_variable',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )

    real_value = Column(
        name = 'real_value',
        type_= Float,
        primary_key = False,
        nullable = True,
        index = False
    )

    predictions = Column(
        name = 'predictions',
        type_= Float,
        primary_key = False,
        nullable = True,
        index = False
    )

    data_quality_percentage = Column(
        name = 'data_quality_percentage',
        type_= Float,
        primary_key = False,
        nullable = True,
        index = False
    )

    validation_boolean = Column(
        name = 'validation_boolean',
        type_= Integer,
        primary_key = False,
        nullable = True,
        index = False
    )

    suggested_value = Column(
        name = 'suggested_value',
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