"""Sequence generation for LSTM/RNN models."""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SequenceGenerator:
    """Generate sequences for time series prediction with LSTM/RNN models."""

    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 1):
        """
        Initialize SequenceGenerator.

        Args:
            sequence_length: Number of time steps to look back.
            prediction_horizon: Number of steps ahead to predict.
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

    def create_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data.

        Args:
            X: Features DataFrame with datetime index.
            y: Target Series with datetime index.
            fit_scaler: Whether to fit the scaler (True for training data).

        Returns:
            Tuple of (X_sequences, y_sequences) as numpy arrays.
        """
        # Store feature columns
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()

        # Scale features
        if fit_scaler:
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
            logger.info("Fitted scalers on training data")
        else:
            X_scaled = self.feature_scaler.transform(X)
            y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1))

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(X_scaled) - self.prediction_horizon + 1):
            # Input sequence: [i-sequence_length : i]
            X_sequences.append(X_scaled[i - self.sequence_length:i])

            # Target: value at i + prediction_horizon - 1
            y_sequences.append(y_scaled[i + self.prediction_horizon - 1])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        logger.info(f"Created sequences - X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")

        return X_sequences, y_sequences

    def inverse_transform_predictions(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.

        Args:
            y_scaled: Scaled predictions.

        Returns:
            Predictions in original scale.
        """
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        y_original = self.target_scaler.inverse_transform(y_scaled)

        return y_original.flatten()

    def prepare_train_val_test(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
        target_column: str = 'Close',
        feature_columns: Optional[list] = None
    ) -> Tuple:
        """
        Prepare train, validation, and test sequences.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame (optional).
            test_df: Test DataFrame.
            target_column: Name of target column.
            feature_columns: List of feature columns. If None, uses all except target.

        Returns:
            Tuple of sequences: (X_train, y_train, X_val, y_val, X_test, y_test)
            If no validation set, returns None for X_val and y_val.
        """
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in train_df.columns if col != target_column]

        # Extract features and target
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        # Create training sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, fit_scaler=True)

        # Create validation sequences if provided
        if val_df is not None:
            X_val = val_df[feature_columns]
            y_val = val_df[target_column]
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, fit_scaler=False)
        else:
            X_val_seq, y_val_seq = None, None

        # Create test sequences
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, fit_scaler=False)

        logger.info("Prepared all sequences for LSTM training")

        return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq

    def get_feature_scaler(self):
        """Get the fitted feature scaler."""
        return self.feature_scaler

    def get_target_scaler(self):
        """Get the fitted target scaler."""
        return self.target_scaler


class MultiStepSequenceGenerator(SequenceGenerator):
    """Generate sequences for multi-step ahead prediction."""

    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5):
        """
        Initialize MultiStepSequenceGenerator.

        Args:
            sequence_length: Number of time steps to look back.
            prediction_horizon: Number of steps ahead to predict (multi-step).
        """
        super().__init__(sequence_length, prediction_horizon)

    def create_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for multi-step prediction.

        Args:
            X: Features DataFrame.
            y: Target Series.
            fit_scaler: Whether to fit the scaler.

        Returns:
            Tuple of (X_sequences, y_sequences) where y_sequences has shape
            (n_samples, prediction_horizon).
        """
        # Scale features
        if fit_scaler:
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
        else:
            X_scaled = self.feature_scaler.transform(X)
            y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1))

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(X_scaled) - self.prediction_horizon + 1):
            # Input sequence
            X_sequences.append(X_scaled[i - self.sequence_length:i])

            # Target: next prediction_horizon values
            y_sequences.append(y_scaled[i:i + self.prediction_horizon].flatten())

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        logger.info(f"Created multi-step sequences - X: {X_sequences.shape}, y: {y_sequences.shape}")

        return X_sequences, y_sequences


class WalkForwardSequenceGenerator(SequenceGenerator):
    """Generate sequences with walk-forward validation approach."""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        n_splits: int = 5
    ):
        """
        Initialize WalkForwardSequenceGenerator.

        Args:
            sequence_length: Number of time steps to look back.
            prediction_horizon: Number of steps ahead to predict.
            n_splits: Number of walk-forward splits.
        """
        super().__init__(sequence_length, prediction_horizon)
        self.n_splits = n_splits

    def generate_walk_forward_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Generate walk-forward validation splits.

        Args:
            X: Features DataFrame.
            y: Target Series.

        Yields:
            Tuples of (X_train_seq, y_train_seq, X_val_seq, y_val_seq) for each split.
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Define split indices
            train_end = (i + 1) * test_size
            val_end = train_end + test_size

            # Skip if not enough data
            if val_end > n_samples:
                break

            # Split data
            X_train_split = X.iloc[:train_end]
            y_train_split = y.iloc[:train_end]

            X_val_split = X.iloc[train_end:val_end]
            y_val_split = y.iloc[train_end:val_end]

            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(
                X_train_split, y_train_split, fit_scaler=True
            )

            X_val_seq, y_val_seq = self.create_sequences(
                X_val_split, y_val_split, fit_scaler=False
            )

            logger.info(f"Walk-forward split {i+1}/{self.n_splits}")

            yield X_train_seq, y_train_seq, X_val_seq, y_val_seq
