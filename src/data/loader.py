"""Data loading module with validation."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load and validate stock price data."""

    def __init__(self, config: dict):
        """
        Initialize DataLoader.

        Args:
            config: Configuration dictionary containing data paths and column names.
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.date_column = self.data_config.get('date_column', 'Date')
        self.target_column = self.data_config.get('target_column', 'Close')

    def load_data(
        self,
        file_path: str,
        parse_dates: bool = True,
        set_index: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV file with validation.

        Args:
            file_path: Path to CSV file.
            parse_dates: Whether to parse date column.
            set_index: Whether to set date column as index.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If required columns are missing.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

            # Validate required columns
            self._validate_columns(df)

            # Parse dates if requested
            if parse_dates and self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                logger.info(f"Parsed {self.date_column} as datetime")

            # Set index if requested
            if set_index and self.date_column in df.columns:
                df.set_index(self.date_column, inplace=True)
                logger.info(f"Set {self.date_column} as index")

            # Validate data quality
            self._validate_data_quality(df)

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def load_train_test_data(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both training and testing datasets.

        Args:
            train_path: Path to training data. If None, uses config.
            test_path: Path to testing data. If None, uses config.

        Returns:
            Tuple of (train_df, test_df).
        """
        # Use paths from config if not provided
        if train_path is None:
            train_path = Path(self.data_config['raw_dir']) / self.data_config['train_file']
        if test_path is None:
            test_path = Path(self.data_config['raw_dir']) / self.data_config['test_file']

        train_df = self.load_data(train_path)
        test_df = self.load_data(test_path)

        logger.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

        return train_df, test_df

    def combine_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label: str = 'dataset'
    ) -> pd.DataFrame:
        """
        Combine train and test datasets with labels.

        Args:
            train_df: Training DataFrame.
            test_df: Testing DataFrame.
            label: Column name for dataset label.

        Returns:
            Combined DataFrame with dataset labels.
        """
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df[label] = 'train'
        test_df[label] = 'test'

        combined = pd.concat([train_df, test_df], axis=0)
        combined.sort_index(inplace=True)

        logger.info(f"Combined dataset shape: {combined.shape}")

        return combined

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = [self.date_column]
        if self.date_column not in df.index.names:
            # Date column should be in columns if not yet set as index
            pass
        else:
            required_columns = []

        # Add target column if specified in config
        if self.target_column:
            required_columns.append(self.target_column)

        # Check for missing columns
        all_columns = list(df.columns) + list(df.index.names)
        missing_columns = [col for col in required_columns if col not in all_columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.debug("Column validation passed")

    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate data quality (missing values, duplicates, etc.).

        Args:
            df: DataFrame to validate.
        """
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in column '{col}'")

        logger.info("Data quality validation completed")

    def get_feature_target_split(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.

        Args:
            df: DataFrame to split.
            target_column: Name of target column. If None, uses config.
            feature_columns: List of feature columns. If None, uses all except target.

        Returns:
            Tuple of (features_df, target_series).
        """
        if target_column is None:
            target_column = self.target_column

        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        X = df[feature_columns]
        y = df[target_column]

        logger.info(f"Features: {X.shape}, Target: {y.shape}")

        return X, y
