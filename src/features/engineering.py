"""Feature engineering module with technical indicators."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Engineer features for stock price prediction."""

    def __init__(self, config: dict):
        """
        Initialize FeatureEngineer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.feature_config = config.get('features', {})

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features based on configuration.

        Args:
            df: Input DataFrame with OHLCV data.

        Returns:
            DataFrame with all engineered features.
        """
        df = df.copy()

        logger.info("Starting feature engineering...")

        # Technical indicators
        if 'technical_indicators' in self.feature_config:
            df = self.add_technical_indicators(df)

        # Lag features
        if 'lag_features' in self.feature_config:
            lags = self.feature_config['lag_features'].get('lags', [1, 2, 3, 5, 10])
            df = self.add_lag_features(df, lags=lags)

        # Rolling statistics
        if 'rolling_stats' in self.feature_config:
            df = self.add_rolling_statistics(df)

        # Price-based features
        df = self.add_price_features(df)

        # Volume-based features
        df = self.add_volume_features(df)

        # Temporal features
        df = self.add_temporal_features(df)

        logger.info(f"Feature engineering completed. Final shape: {df.shape}")

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with technical indicators added.
        """
        df = df.copy()
        ti_config = self.feature_config.get('technical_indicators', {})

        # Simple Moving Averages
        if 'sma_windows' in ti_config:
            for window in ti_config['sma_windows']:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

        # Exponential Moving Averages
        if 'ema_windows' in ti_config:
            for window in ti_config['ema_windows']:
                df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

        # Relative Strength Index (RSI)
        if 'rsi_window' in ti_config:
            df = self._add_rsi(df, window=ti_config['rsi_window'])

        # MACD
        if 'macd' in ti_config:
            macd_config = ti_config['macd']
            df = self._add_macd(
                df,
                fast=macd_config.get('fast', 12),
                slow=macd_config.get('slow', 26),
                signal=macd_config.get('signal', 9)
            )

        # Bollinger Bands
        if 'bollinger_bands' in ti_config:
            bb_config = ti_config['bollinger_bands']
            df = self._add_bollinger_bands(
                df,
                window=bb_config.get('window', 20),
                num_std=bb_config.get('num_std', 2)
            )

        logger.info("Added technical indicators")

        return df

    def _add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        df = df.copy()

        # Calculate price changes
        delta = df['Close'].diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Calculate RS and RSI
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def _add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        df = df.copy()

        # Calculate EMAs
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line
        df['MACD'] = ema_fast - ema_slow

        # Calculate signal line
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

        # Calculate MACD histogram
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        return df

    def _add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: int = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = df.copy()

        # Calculate rolling mean and std
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()

        # Calculate bands
        df['BB_Middle'] = rolling_mean
        df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        df['BB_Lower'] = rolling_mean - (rolling_std * num_std)

        # Calculate bandwidth and %B
        df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        lags: list = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add lag features.

        Args:
            df: Input DataFrame.
            columns: Columns to create lags for. If None, uses price columns.
            lags: List of lag periods.

        Returns:
            DataFrame with lag features added.
        """
        df = df.copy()

        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            columns = [col for col in columns if col in df.columns]

        for col in columns:
            for lag in lags:
                df[f'{col}_Lag_{lag}'] = df[col].shift(lag)

        logger.info(f"Added {len(columns) * len(lags)} lag features")

        return df

    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with rolling statistics added.
        """
        df = df.copy()
        rs_config = self.feature_config.get('rolling_stats', {})

        windows = rs_config.get('windows', [5, 10, 20])
        stats = rs_config.get('stats', ['mean', 'std', 'min', 'max'])
        columns = ['Close', 'Volume']
        columns = [col for col in columns if col in df.columns]

        for col in columns:
            for window in windows:
                for stat in stats:
                    col_name = f'{col}_Rolling_{window}_{stat}'
                    if stat == 'mean':
                        df[col_name] = df[col].rolling(window=window).mean()
                    elif stat == 'std':
                        df[col_name] = df[col].rolling(window=window).std()
                    elif stat == 'min':
                        df[col_name] = df[col].rolling(window=window).min()
                    elif stat == 'max':
                        df[col_name] = df[col].rolling(window=window).max()

        logger.info(f"Added {len(columns) * len(windows) * len(stats)} rolling features")

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with price features added.
        """
        df = df.copy()

        # Daily returns
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Price ranges
        if 'High' in df.columns and 'Low' in df.columns:
            df['Daily_Range'] = df['High'] - df['Low']
            df['Price_Range_Pct'] = (df['High'] - df['Low']) / df['Close']

        # Gap features
        if 'Open' in df.columns and 'Close' in df.columns:
            df['Intraday_Change'] = df['Close'] - df['Open']
            df['Intraday_Change_Pct'] = (df['Close'] - df['Open']) / df['Open']
            df['Overnight_Gap'] = df['Open'] - df['Close'].shift(1)
            df['Overnight_Gap_Pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Momentum
        if 'Close' in df.columns:
            df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
            df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

        logger.info("Added price-based features")

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with volume features added.
        """
        df = df.copy()

        if 'Volume' not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df

        # Volume changes
        df['Volume_Change'] = df['Volume'].pct_change()

        # Volume moving averages
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()

        # Relative volume
        df['Relative_Volume'] = df['Volume'] / df['Volume_MA_10']

        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Volume-Price Trend
        if 'Close' in df.columns:
            df['VPT'] = df['Volume'] * (df['Close'].pct_change())
            df['VPT'] = df['VPT'].fillna(0).cumsum()

        logger.info("Added volume-based features")

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (day of week, month, etc.).

        Args:
            df: Input DataFrame with datetime index.

        Returns:
            DataFrame with temporal features added.
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping temporal features")
            return df

        # Day of week (0 = Monday, 6 = Sunday)
        df['DayOfWeek'] = df.index.dayofweek

        # Month (1-12)
        df['Month'] = df.index.month

        # Quarter (1-4)
        df['Quarter'] = df.index.quarter

        # Day of month
        df['DayOfMonth'] = df.index.day

        # Week of year
        df['WeekOfYear'] = df.index.isocalendar().week

        # Is month end/start
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsMonthStart'] = df.index.is_month_start.astype(int)

        # Is quarter end/start
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        df['IsQuarterStart'] = df.index.is_quarter_start.astype(int)

        logger.info("Added temporal features")

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'Close',
        method: str = 'correlation',
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Select features based on correlation with target.

        Args:
            df: Input DataFrame.
            target_column: Target column name.
            method: Selection method ('correlation').
            threshold: Minimum correlation threshold.

        Returns:
            DataFrame with selected features.
        """
        df = df.copy()

        if method == 'correlation':
            # Calculate correlations with target
            correlations = df.corr()[target_column].abs()

            # Select features above threshold
            selected_features = correlations[correlations >= threshold].index.tolist()

            # Remove target from features
            if target_column in selected_features:
                selected_features.remove(target_column)

            logger.info(f"Selected {len(selected_features)} features with correlation >= {threshold}")

            return df[selected_features + [target_column]]

        else:
            raise ValueError(f"Unknown selection method: {method}")
