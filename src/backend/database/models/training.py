from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
from database.models.base import Base

class ModelVersioningTable(Base):
    """
    Definition of the table that contains the model versioning.
    """
    __tablename__ = 'model_versioning'

    model_date = Column(
        name = 'model_date',
        type_= DateTime,
        primary_key = True,
        nullable = False,
        index = False
    )

    model_name = Column(
        name = 'model_name',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )

    model_type = Column(
        name = 'model_type',
        type_= String,
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

    target_variable = Column(
        name = 'target_variable',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )

    model_feature_names = Column(
        name = 'model_feature_names',
        type_= String,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_features_count = Column(
        name = 'model_features_count',
        type_= Integer,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_version = Column(
        name = 'model_version',
        type_= Integer,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_parameters = Column(
        name = 'model_parameters',
        type_= String,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_metric_name = Column(
        name = 'model_metric_name',
        type_= String,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_metric_validation_value = Column(
        name = 'model_metric_validation_value',
        type_= Float,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_metric_test_value = Column(
        name = 'model_metric_test_value',
        type_= Float,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_forecast_accuracy = Column(
        name = 'model_forecast_accuracy',
        type_= Float,
        primary_key = False,
        nullable = True,
        index = False
    )

    standard_scaler_binary = Column(
        name = 'standard_scaler_binary',
        type_= LargeBinary,
        primary_key = False,
        nullable = True,
        index = False
    )

    polynomial_transformer_binary = Column(
        name = 'polynomial_transformer_binary',
        type_= LargeBinary,
        primary_key = False,
        nullable = True,
        index = False
    )

    model_binary = Column(
        name = 'model_binary',
        type_= LargeBinary,
        primary_key = False,
        nullable = True,
        index = False
    )
    
    feature_importance_variables = Column(
        name = 'feature_importance_variables',
        type_= String,
        primary_key = False,
        nullable = True,
        index = False
    )
    
    feature_importance_values = Column(
        name = 'feature_importance_values',
        type_= String,
        primary_key = False,
        nullable = True,
        index = False
    )


class TrainingConfigurationTable(Base):
    __tablename__ = 'training_configuration'

    timestamp = Column(
        name = 'timestamp',
        type_= DateTime,
        primary_key = True,
        nullable = False,
        index = False
    )

    stage = Column(
        name='stage',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )
    
    parameter_code = Column(
        name = 'parameter_code',
        type_= String,
        primary_key = True,
        nullable = False,
        index = False
    )
    
    parameter_value = Column(
        name = 'parameter_value',
        type_= String,
        primary_key = False,
        nullable = True,
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