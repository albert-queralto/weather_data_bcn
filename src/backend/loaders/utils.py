
from datetime import datetime
from typing import Any, Hashable, Optional
from dataclasses import dataclass
from loaders.base import DatabaseDataManager

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database.models.utils import LastProcessedDataTable
from database.schemas.utils import LastProcessedDataSchema


@dataclass
class LastProcessedDataManager(DatabaseDataManager):
    """
    Loads the last processed data timestamp.
    """
    def load(self, 
            latitude: float,
            longitude: float,
            data_type: str
        ) -> Optional[datetime]:
        """
        Gets the last processed timestamp for the latitude, longitude
        and data type.
        """
        statement = select(LastProcessedDataTable).where(
            LastProcessedDataTable.latitude == latitude,
            LastProcessedDataTable.longitude == longitude,
            LastProcessedDataTable.data_type == data_type
        )

        with self.Session.begin() as session:
            if query := session.execute(statement).all():
                row = query[0]
                return LastProcessedDataSchema(
                    latitude=row[0].latitude,
                    longitude=row[0].longitude,
                    data_type=row[0].data_type,
                    timestamp=row[0].timestamp,
                    update_date=row[0].update_date
                ).model_dump()['timestamp']

    def save(self, 
            latitude: float,
            longitude: float,
            data_type: str, 
            timestamp: datetime
        ) -> None:
        """
        Saves the last processed timestamp to the database for the latitude,
        longitude and data type.
        """
        df_dict = {
            'latitude': latitude,
            'longitude': longitude,
            'data_type': data_type,
            'timestamp': timestamp
        }

        size = len(df_dict)
        self.logger.debug(f"Inserting {size} rows into the database...")
        self.logger.debug(f"Inserting {df_dict}...")
        self._statement_insert_to_db(df_dict)

    def _statement_insert_to_db(self, data: list[dict[Hashable, Any]]) -> None:
        """
        Inserts the data from the dataframe to the database.
        """
        try:
            statement = insert(LastProcessedDataTable).values(data)
            statement = statement.on_conflict_do_update(
                index_elements=[
                    LastProcessedDataTable.latitude,
                    LastProcessedDataTable.longitude,
                    LastProcessedDataTable.data_type
                ],
                set_=dict(
                    timestamp=statement.excluded.timestamp,
                    update_date=datetime.now()
                )
            )
            
            with self.Session.begin() as session:
                session.execute(statement)
        except Exception as e:
            self.logger.debug(e)
            raise e