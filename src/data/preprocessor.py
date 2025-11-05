"""Data preprocessing module with time series support."""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Preprocess data with time series awareness."""

    def __init__(self, config: dict):
        """
        Initialize DataPreprocessor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.cv_config = config.get('cross_validation', {})
        self.scaler = None
        self.feature_names = None

    def create_time_series_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create train/validation/test split for time series data.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            test_size: Proportion of data for test set.
            validation_size: Proportion of remaining data for validation set.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # Ensure data is sorted by index (date)
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index")

        # Sort by index
        X = X.sort_index()
        y = y.sort_index()

        n = len(X)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        # Split data chronologically
        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]

        y_train = y.iloc[:val_idx]
        y_val = y.iloc[val_idx:test_idx]
        y_test = y.iloc[test_idx:]

        logger.info(f"Time series split - Train: {len(X_train)}, "
                   f"Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_time_series_cv(self, n_splits: int = 5) -> TimeSeriesSplit:
        """
        Get TimeSeriesSplit cross-validator.

        Args:
            n_splits: Number of splits.

        Returns:
            TimeSeriesSplit object.
        """
        return TimeSeriesSplit(n_splits=n_splits)

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        scaler_type: str = 'standard'
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
        """
        Scale features using specified scaler.

        Args:
            X_train: Training features.
            X_val: Validation features (optional).
            X_test: Test features (optional).
            scaler_type: Type of scaler ('standard', 'minmax', 'maxabs').

        Returns:
            Scaled DataFrames. If only X_train provided, returns single DataFrame.
            Otherwise returns tuple of scaled DataFrames.
        """
        # Select scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'maxabs':
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Fit on training data only
        self.scaler.fit(X_train)
        logger.info(f"Fitted {scaler_type} scaler on training data")

        # Transform training data
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )

        # Transform validation data if provided
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                index=X_val.index,
                columns=X_val.columns
            )

        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )

        # Return appropriate values
        if X_val is None and X_test is None:
            return X_train_scaled
        elif X_val is None:
            return X_train_scaled, X_test_scaled
        elif X_test is None:
            return X_train_scaled, X_val_scaled
        else:
            return X_train_scaled, X_val_scaled, X_test_scaled

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'forward_fill',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: Input DataFrame.
            strategy: Strategy for handling missing values
                     ('forward_fill', 'backward_fill', 'interpolate', 'drop', 'fill').
            fill_value: Value to use for 'fill' strategy.

        Returns:
            DataFrame with missing values handled.
        """
        df = df.copy()

        if strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif strategy == 'interpolate':
            df = df.interpolate(method='time')
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value must be provided for 'fill' strategy")
            df = df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Handled missing values using '{strategy}' strategy")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.

        Args:
            df: Input DataFrame.
            columns: Columns to check for outliers. If None, uses all numeric columns.
            method: Method for outlier detection ('iqr', 'zscore').
            threshold: Threshold for outlier detection (IQR multiplier or z-score).

        Returns:
            DataFrame with outliers removed.
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        mask = pd.Series(True, index=df.index)

        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask &= z_scores < threshold

        original_len = len(df)
        df = df[mask]
        removed_count = original_len - len(df)

        logger.info(f"Removed {removed_count} outliers using '{method}' method")

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: list,
        lags: list = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lag features for time series.

        Args:
            df: Input DataFrame.
            columns: Columns to create lags for.
            lags: List of lag periods.

        Returns:
            DataFrame with lag features added.
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        logger.info(f"Created {len(columns) * len(lags)} lag features")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: list,
        windows: list = [5, 10, 20],
        stats: list = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            df: Input DataFrame.
            columns: Columns to create rolling features for.
            windows: List of window sizes.
            stats: List of statistics to compute ('mean', 'std', 'min', 'max').

        Returns:
            DataFrame with rolling features added.
        """
        df = df.copy()

        for col in columns:
            for window in windows:
                for stat in stats:
                    if stat == 'mean':
                        df[f'{col}_rolling_{window}_mean'] = df[col].rolling(window=window).mean()
                    elif stat == 'std':
                        df[f'{col}_rolling_{window}_std'] = df[col].rolling(window=window).std()
                    elif stat == 'min':
                        df[f'{col}_rolling_{window}_min'] = df[col].rolling(window=window).min()
                    elif stat == 'max':
                        df[f'{col}_rolling_{window}_max'] = df[col].rolling(window=window).max()

        logger.info(f"Created {len(columns) * len(windows) * len(stats)} rolling features")

        return df

    def get_scaler(self):
        """
        Get the fitted scaler.

        Returns:
            Fitted scaler object.
        """
        return self.scaler

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features.

        Args:
            X: Scaled features.

        Returns:
            Original scale features.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet")

        X_original = pd.DataFrame(
            self.scaler.inverse_transform(X),
            index=X.index,
            columns=X.columns
        )

        return X_original
