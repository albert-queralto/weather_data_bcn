"""
Implements different anomaly detection techniques to clean the data.
"""
import numpy as np
import pandas as pd
from typing import Union

class OutlierRemoval:
    """
    Implements different anomaly detection techniques to clean the data.
    """
    def rolling_mean(self,
        df: Union[pd.DataFrame, pd.Series],
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Applies a rolling mean to smooth the data.
        """
        df = self._check_dataframe(df)
        return df.rolling(window_size, center=True).mean()

    def rolling_median(self,
            df: Union[pd.DataFrame, pd.Series],
            window_size: int = 10
        ) -> pd.DataFrame:
        """
        Applies a rolling median to smooth the data.
        """
        df = self._check_dataframe(df)
        return df.rolling(window_size, center=True).median()

    def zscore_outlier_removal(self,
        df: Union[pd.DataFrame, pd.Series],
        threshold: int = 3,
        method: str = 'mean', 
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Removes outliers from a dataframe using the Z-score method.
        """
        df = self._check_dataframe(df)
        cleaned_df = df.copy()
        for col in df.columns:
            z_scores = pd.DataFrame()
            if method == 'mean':
                z_scores = self.zscore_mean_windowed(df[col], window_size)
            elif method == 'median':
                z_scores = self.zscore_median_windowed(df[col], window_size)
            cleaned_df[col].iloc[z_scores > threshold] = np.nan
        return cleaned_df

    def tukey_removal(self,
        df: Union[pd.DataFrame, pd.Series], 
        window_size: int, 
        bound_limit: float
    ) -> pd.DataFrame:
        """
        Detects and removes the outliers in the data using the Tukey test.
        """
        df = self._check_dataframe(df)
        cleaned_df = df.copy()
        for col in df.columns:
            _, outliers_idxs = self.tukey_test_windowed(df[col], window_size, bound_limit)
            cleaned_df[col].iloc[outliers_idxs] = np.nan
        return cleaned_df

    def _check_dataframe(self, df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Checks if the dataframe is a pd.Series and converts it to a pd.DataFrame.
        """
        df = df.copy()
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df.values.reshape(-1, 1), index=df.index, columns=[df.name]).copy()
        return df

    def zscore_mean_windowed(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """
        Computes the z-score for each data point using a rolling window approach.
        """
        return (df - df.rolling(window=window_size).mean()) / df.rolling(window=window_size).std()

    def zscore_median_windowed(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """
        Computes the z-score with the median instead of the mean for each data point
        using a rolling window approach.
        """
        return (df - df.rolling(window=window_size).median()) / df.rolling(window=window_size).std()

    def tukey_test_windowed(self,
        df: pd.DataFrame, 
        window_size: int, 
        bound_limit: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs the Tukey test for each data point using a rolling window approach.
        """
        q1 = df.rolling(window_size).quantile(0.25, interpolation='midpoint')
        q3 = df.rolling(window_size).quantile(0.75, interpolation='midpoint')
        iqr = q3 - q1
        lower_bound = q1 - bound_limit * iqr
        upper_bound = q3 + bound_limit * iqr
        outliers = np.logical_or(df < lower_bound, df > upper_bound)
        outlier_pos = np.nonzero(outliers)[0]
        return outliers, outlier_pos