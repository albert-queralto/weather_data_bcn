
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Hashable, Optional

from sqlalchemy import (
    select,
    or_,
    and_,
    PrimaryKeyConstraint,
)
from sqlalchemy.dialects.postgresql import insert

from loaders.base import DatabaseDataManager
from database.models.preprocessing import EngineeredFeaturesTable
from database.schemas.preprocessing import EngineeredFeaturesSchema


@dataclass
class EngineeredFeaturesManager(DatabaseDataManager):
    """
    Loads the data from the engineered_features table.
    """
    def load(self,
            latitude: str,
            longitude: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            filter_variables: Optional[list[str]] = None,
        ) -> pd.DataFrame:      
        conditions_list = ['None', None, '']
        conditions = [
            and_(
                EngineeredFeaturesTable.latitude == latitude,
                EngineeredFeaturesTable.longitude == longitude,
            )
        ]

        if start_date not in conditions_list and end_date not in conditions_list:
            conditions.append(EngineeredFeaturesTable.timestamp.between(start_date, end_date))

        if filter_variables not in conditions_list and isinstance(filter_variables, list):
            conditions.append(EngineeredFeaturesTable.variable_code.op('~')(f'{"|".join(filter_variables)}'))

        elif filter_variables not in conditions_list and isinstance(filter_variables, str):
            conditions.append(EngineeredFeaturesTable.variable_code.op('~')(f'{filter_variables}')) 

        statement = select(EngineeredFeaturesTable).where(
            and_(*conditions)
        ).order_by(
            EngineeredFeaturesTable.latitude,
            EngineeredFeaturesTable.longitude,
            EngineeredFeaturesTable.timestamp
        )
        
        with self.Session.begin() as session:
            query = session.execute(statement).all()
            results = [
                EngineeredFeaturesSchema(
                    timestamp=row[0].timestamp,
                    latitude=row[0].latitude,
                    longitude=row[0].longitude,
                    variable_code=row[0].variable_code,
                    value=row[0].value,
                    update_date=row[0].update_date
                ).model_dump()
                for row in query
            ]
            df = pd.DataFrame(results)
        return df

    def process_data_loading(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the data from engineered features. Checks the column names,
        transforms the dataframe to the right format and reindexes the dataframe.
        """
        df = self._check_and_sort_column_names(df=df)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_pivot = df.pivot_table(index='timestamp', columns='variable_code', values='value', aggfunc='first')
        df_pivot.columns = [''.join(col).strip('value') for col in df_pivot.columns.values]
        df_pivot.reset_index(inplace=True)
        df_pivot.set_index('timestamp', inplace=True)
        df_pivot.index.name = None

        # ----------------- DEBUGGING -----------------
        self.logger.debug(f"Dataframe columns:\n{df_pivot.columns}")
        self.logger.debug(f"Dataframe index:\n{df_pivot.index}")
        self.logger.debug(f"Dataframe head:\n{df_pivot.head()}")
        self.logger.debug(f"Dataframe shape:\n{df_pivot.shape}")
        # ---------------------------------------------

        return df_pivot

    def _check_and_sort_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the column names are the expected ones and in the right order.
        If this is fulfilled, then returns a dataframe with renamed columns.
        """
        current_names = list(df.columns)
        expected_names = ["timestamp", "latitude", "longitude", "variable_code", "value", "update_date"]
        for current, expected in zip(current_names, expected_names):
            if expected.lower() not in current.lower():
                raise ValueError(f"Column name {current} is not the expected one: {expected}")
        df.columns = expected_names
        return df.copy()

    def process_data_saving(self, df: pd.DataFrame) -> pd.DataFrame:
        """Includes the preprocessing steps before uploading the data to the
        database."""
        df.reset_index(inplace=True)
        df.rename(columns={'date': 'timestamp'}, inplace=True)
        df = self._unpivot_df(df)
        df['update_date'] = datetime.now(timezone.utc)
        return df

    def _statement_insert_to_db(self, data: list[dict[Hashable, Any]]) -> None:
        """
        Inserts the data from the dataframe to the database.
        """
        try:
            statement = insert(EngineeredFeaturesTable).values(data)
            statement = statement.on_conflict_do_update(
                constraint=PrimaryKeyConstraint(
                    EngineeredFeaturesTable.timestamp,
                    EngineeredFeaturesTable.latitude,
                    EngineeredFeaturesTable.longitude,
                    EngineeredFeaturesTable.variable_code,
                ),
                set_=dict(
                    value=statement.excluded.value,
                    update_date=datetime.now()
                )
            )

            with self.Session.begin() as session:
                session.execute(statement)

        except Exception as e:
            self.logger.debug(e)
            raise e

    def _unpivot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.melt(
            df,
            id_vars=['timestamp'],
            var_name='variable_code',
            value_name='value'
        )

    def get_coordinates(self) -> list[dict[str, str]]:
        """
        Returns the unique coordinates from the database.
        """
        statement = select(EngineeredFeaturesTable.latitude, EngineeredFeaturesTable.longitude).distinct()
        with self.Session.begin() as session:
            query = session.execute(statement).all()
            results = [
                {
                    "latitude": row[0],
                    "longitude": row[1]
                }
                for row in query
            ]
        return results