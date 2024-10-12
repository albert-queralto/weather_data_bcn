import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging


@dataclass
class ApiDataManager(ABC):
    """Loads the data from APIs."""
    logger: logging.Logger

    @abstractmethod
    def load(self) -> None:
        """Loads the data from a database."""
    
    @abstractmethod
    def save(self) -> None:
        """Saves the data to a database."""


@dataclass
class OpenMeteoDataManager(ApiDataManager):

    def __post_init__(self):
        self.cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        self.retry_session = retry(self.cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = self.retry_session)
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def load(self,
            latitude: float = 41.389,
            longitude: float = 2.159,
            start_date: str = "2024-09-24",
            end_date: str = "2024-10-08",
        ) -> pd.DataFrame:

        variables = ["temperature_2m", "relative_humidity_2m", "precipitation", 
                "pressure_msl", "surface_pressure", "cloud_cover", 
                "wind_speed_10m", "sunshine_duration", "direct_radiation"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": variables
        }

        responses = self.openmeteo.weather_api(self.url, params=params)

        response = responses[0]
        self.logger.debug(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        self.logger.debug(f"Elevation {response.Elevation()} m asl")
        self.logger.debug(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        self.logger.debug(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        for i, var in enumerate(variables):
            hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

        return pd.DataFrame(data=hourly_data)

    def save(self) -> None:
        """Saves the data to an API."""
        pass