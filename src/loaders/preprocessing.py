import math
import logging
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Hashable, Optional

from sqlalchemy import (
    select,
    join,
    or_,
    func,
    and_,
    PrimaryKeyConstraint,
    distinct,
    delete,
    sql
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection
from sqlalchemy.orm import sessionmaker

from database.models.preprocessing import EngineeredFeaturesTable
from database.schemas.preprocessing import EngineeredFeaturesSchema

@dataclass
class DatabaseDataManager(ABC):
    """Abstract class to load the data."""
    connection: Connection
    logger: logging.Logger
    
    def __post_init__(self) -> None:
        self.Session = sessionmaker(bind=self.connection)

    def load(self) -> None:
        """Abstract function used to load the data from a database."""

    def save(self, df: pd.DataFrame, batch_size: int) -> None:
        """
        Save the data to the database table.
        """
        df_dict = df.to_dict(orient='records')
        df_dict = [{k: sql.null() if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()} for row in df_dict]

        size = len(df_dict)
        self.logger.debug(f"Inserting {size} rows into the database...")

        if size > batch_size:
            n_batches = size // batch_size

            for i in range(n_batches):
                start = i * batch_size
                end = (i + 1) * batch_size - 1
                self.logger.debug(f"Inserting data from dates {df_dict[start]['timestamp']} to {df_dict[end]['timestamp']}...")
                self._statement_insert_to_db(df_dict[start:end])
            self._statement_insert_to_db(df_dict[n_batches * batch_size:])

        else:
            self._statement_insert_to_db(df_dict)

    def _statement_insert_to_db(self) -> None:
        """Abstract method with the statement used to save the data to the database."""


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
            or_(
                EngineeredFeaturesTable.latitude.like(f'{latitude}%'),
                EngineeredFeaturesTable.longitude.like(f'{longitude}%'),
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