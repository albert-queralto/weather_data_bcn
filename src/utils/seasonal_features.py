import pandas as pd
from datetime import date
from helpers import compose_functions

class CreateSeasonalFeatures:
    """ 
    Creates different seasonal features from the input data. The data must have
    a DateTimeIndex
    """
    @staticmethod	
    def __call__(df: pd.DataFrame, methods: list) -> pd.DataFrame:
        seasonal_feats = compose_functions(*methods)
        return seasonal_feats(df)
        
    @staticmethod	
    def _get_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
        """Gets the day of the month from the index of the original data."""
        new_df = df.copy()
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        new_df['day_of_month'] = new_df.index.day
        return new_df

    @staticmethod
    def _get_month(df: pd.DataFrame) -> pd.DataFrame:
        """Gets the month from the index of the original data."""
        new_df = df.copy()
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        new_df['month'] = new_df.index.month
        return new_df
        
    @staticmethod
    def _get_hour(df: pd.DataFrame) -> pd.DataFrame:
        """Gets the hour of the day from the index of the original data."""
        new_df = df.copy()
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        new_df['hour'] = new_df.index.hour
        return new_df
        
    @staticmethod
    def _get_weekday(df: pd.DataFrame) -> pd.DataFrame:
        """Gets the weekday from the index of the original data.
        The value goes from 0: Monday to 6: Sunday."""
        new_df = df.copy()
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        new_df['weekday'] = new_df.index.weekday
        return new_df

    @staticmethod
    def _get_year_season(df: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the year season from the index of the original data.
        The value are:
        - 0: Winter.
        - 1: Spring.
        - 2: Summer.
        - 3: Autumn.
        """
        new_df = df.copy()
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        year_seasons = new_df.index.to_series().apply(CreateSeasonalFeatures._get_season_date)

        seasons_map = {
            "winter": 0,
            "spring": 1,
            "summer": 2,
            "autumn": 3
        }

        new_df['year_season'] = year_seasons.map(seasons_map)
        return new_df

    @staticmethod
    def _get_season_date(time_stamp: pd.Timestamp) -> str:
        """
        Gets the year season from the date.
        """
        datetime_date = time_stamp.date()

        seasons = [('winter', (date(time_stamp.year,  1,  1),  date(time_stamp.year,  3, 20))),
                    ('spring', (date(time_stamp.year,  3, 21),  date(time_stamp.year,  6, 20))),
                    ('summer', (date(time_stamp.year,  6, 21),  date(time_stamp.year,  9, 22))),
                    ('autumn', (date(time_stamp.year,  9, 23),  date(time_stamp.year, 12, 20))),
                    ('winter', (date(time_stamp.year, 12, 21),  date(time_stamp.year, 12, 31)))]
        return next(season for season, (start, end) in seasons 
                                        if start <= datetime_date <= end)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from helpers import get_class_methods_exclude_dunder
    
    index = pd.date_range(start="2024-01-01", end="2024-04-09", freq="D")
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 100, 100)
    
    df = pd.DataFrame(data, index=index)
    
    seasonal_feats_creator = CreateSeasonalFeatures()
    seasonal_methods = [getattr(seasonal_feats_creator, method)
            for method in get_class_methods_exclude_dunder(seasonal_feats_creator)
            if not any(x in method for x in ["_get_season_date"])]
    
    new_df = seasonal_feats_creator(df, seasonal_methods)
    print(new_df)