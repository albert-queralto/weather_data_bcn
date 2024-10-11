from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional


class EngineeredFeaturesSchema(BaseModel):
    """
    Definition of the table that contains the engineered features.
    """
    timestamp: datetime
    latitude: str
    longitude: str
    variable_code: str
    value: Optional[float] = None
    update_date: Optional[datetime] = None

    model_config = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))