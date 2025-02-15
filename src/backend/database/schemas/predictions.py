from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional

class PredictionsSchema(BaseModel):
    """
    Definition of the table that contains the predictions.
    """
    timestamp: datetime
    model_date: datetime
    model_name: str
    model_type: str
    model_version: int
    latitude: str
    longitude: str
    target_variable: str
    real_value: float
    predictions: float
    data_quality_percentage: float
    validation_boolean: int
    suggested_value: float
    update_date: Optional[datetime] = None

    model_config = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))