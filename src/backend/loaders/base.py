import math
import logging
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy.engine import Connection
from sqlalchemy.orm import sessionmaker
from sqlalchemy import sql

@dataclass
class DatabaseDataManager(ABC):
    """Abstract class to load the data."""
    connection: Connection
    logger: logging.Logger
    
    def __post_init__(self) -> None:
        self.Session = sessionmaker(bind=self.connection)

    @abstractmethod
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

    @abstractmethod
    def _statement_insert_to_db(self) -> None:
        """Abstract method with the statement used to save the data to the database."""
