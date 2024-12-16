from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional

class ModelVersioningSchema(BaseModel):
    """
    Definition of the table that contains the model versioning.
    """
    model_date: datetime
    model_name: str
    model_type: str
    latitude: str
    longitude: str
    target_variable: str
    model_feature_names: str
    model_features_count: int
    model_version: int
    model_parameters: str
    model_metric_name: str
    model_metric_validation_value: float
    model_metric_test_value: float
    standard_scaler_binary: bytes
    model_forecast_accuracy: Optional[float] = None
    polynomial_transformer_binary: Optional[bytes] = None
    model_binary: Optional[bytes] = None
    feature_importance_variables: Optional[str] = None
    feature_importance_values: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))


class TrainingConfigurationSchema(BaseModel):
    timestamp: datetime
    stage: str
    parameter_code: str
    parameter_value: str
    created_date: Optional[datetime] = None
    update_date: Optional[datetime] = None

    model_config = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))