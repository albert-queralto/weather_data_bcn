from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional


class LastProcessedDataSchema(BaseModel):
    """
    Definition of the table that contains the last processed data timestamp.
    """
    latitude: float
    longitude: float
    data_type: str
    timestamp: datetime
    update_date: Optional[datetime] = None

    model_config = ConfigDict(protected_namespaces=('protect_me_', 'also_protect_'))