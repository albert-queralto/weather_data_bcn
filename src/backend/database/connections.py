from abc import ABC, abstractmethod
from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from typing import Optional


@dataclass
class Connector(ABC):
    """
    Abstract class to connect to an endpoint.
    """
    @abstractmethod
    def connect(self) -> None:
        """
        Connects to the endpoint.
        """


@dataclass
class ConnectionStringBuilder:
    """
    Class to build connection strings.
    """
    def __post_init__(self):
        """Initialize the database connection class."""
        self.db_connections: dict[str, str] = {
            "postgresql": "postgresql://",
            "mysql": "mysql://",
            "oracle": "oracle://",
            "mssql": "mssql+pyodbc://",
            "sqlite": "sqlite://"
        }

    def __call__(self,
        connection_type: str,
        user_name: str,
        password: str,
        host: str,
        database_name: Optional[str],
        port: Optional[str]
    ) -> str:
        """
        Builds the connection string based on the parameters passed.
        """
        if connection_type not in self.db_connections:
            raise NotImplementedError(f"Unsupported database type: {connection_type}")

        if connection_type == 'sqlite':
            connection_string = (
                f"{self.db_connections[connection_type]}"
            )
        else:
            connection_string = (
                f"{self.db_connections[connection_type]}{user_name}:{password}@{host}"
            )

            if port:
                connection_string += f":{port}"

        if database_name:
            connection_string += f"/{database_name}"

        return connection_string


@dataclass
class DatabaseConnection(Connector):
    """
    Class to create database connection based on the passed string to the init.
    """

    def connect(self, connection_string: str) -> Connection:
        """
        Creates the connection to the database based on the db_type and the
        parameters passed.
        """    
        try:
            return create_engine(connection_string, future=True).connect() # type: ignore
        except Exception as e:
            raise e
