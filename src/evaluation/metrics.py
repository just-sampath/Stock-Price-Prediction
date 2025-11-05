"""Model evaluation metrics module."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics."""

    def __init__(self, config: dict):
        """
        Initialize ModelEvaluator.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        set_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate predictions with multiple metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            set_name: Name of the dataset (e.g., "Train", "Test").

        Returns:
            Dictionary of metric names and values.
        """
        metrics = {}

        # R² Score
        metrics['r2_score'] = r2_score(y_true, y_pred)

        # Mean Squared Error
        metrics['mean_squared_error'] = mean_squared_error(y_true, y_pred)

        # Root Mean Squared Error
        metrics['root_mean_squared_error'] = np.sqrt(metrics['mean_squared_error'])

        # Mean Absolute Error
        metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)

        # Mean Absolute Percentage Error
        metrics['mean_absolute_percentage_error'] = mean_absolute_percentage_error(y_true, y_pred)

        # Directional Accuracy
        metrics['directional_accuracy'] = self.calculate_directional_accuracy(y_true, y_pred)

        # Custom metrics
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['median_absolute_error'] = np.median(np.abs(y_true - y_pred))

        logger.info(f"{set_name} Set Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def calculate_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (for time series).

        Measures how often the model correctly predicts the direction of change.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Directional accuracy (0-1).
        """
        # Calculate actual and predicted changes
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0

        # Calculate accuracy
        correct_directions = actual_direction == predicted_direction
        directional_accuracy = np.mean(correct_directions)

        return directional_accuracy

    def calculate_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate residuals.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Array of residuals.
        """
        return y_true - y_pred

    def create_metrics_dataframe(
        self,
        results_dict: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create DataFrame from metrics dictionary.

        Args:
            results_dict: Dictionary mapping model names to metrics.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        df = pd.DataFrame(results_dict).T

        # Sort by test R² score if available
        if 'test_r2_score' in df.columns:
            df = df.sort_values('test_r2_score', ascending=False)
        elif 'r2_score' in df.columns:
            df = df.sort_values('r2_score', ascending=False)

        return df

    def compare_models(
        self,
        models: Dict[str, object],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dictionary mapping model names to fitted models.
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.

        Returns:
            DataFrame with comparison metrics.
        """
        results = {}

        for name, model in models.items():
            logger.info(f"Evaluating {name}...")

            # Training metrics
            y_train_pred = model.predict(X_train)
            train_metrics = self.evaluate(y_train, y_train_pred, set_name=f"{name} Train")

            # Test metrics
            y_test_pred = model.predict(X_test)
            test_metrics = self.evaluate(y_test, y_test_pred, set_name=f"{name} Test")

            # Combine metrics
            results[name] = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()}
            }

        # Create comparison DataFrame
        comparison_df = self.create_metrics_dataframe(results)

        return comparison_df

    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv,
        scoring: str = 'r2'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            model: Model to evaluate.
            X: Features.
            y: Target.
            cv: Cross-validation splitter.
            scoring: Scoring metric.

        Returns:
            Dictionary with CV scores.
        """
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            f'{scoring}_mean': scores.mean(),
            f'{scoring}_std': scores.std(),
            f'{scoring}_min': scores.min(),
            f'{scoring}_max': scores.max()
        }

        logger.info(f"Cross-validation {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return results

    def calculate_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: float = 0.95
    ) -> tuple:
        """
        Calculate prediction intervals.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            confidence: Confidence level (0-1).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        residuals = self.calculate_residuals(y_true, y_pred)
        residual_std = np.std(residuals)

        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)

        # Calculate intervals
        lower_bound = y_pred - z_score * residual_std
        upper_bound = y_pred + z_score * residual_std

        return lower_bound, upper_bound

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio for trading strategy.

        Args:
            returns: Array of returns.
            risk_free_rate: Risk-free rate of return.

        Returns:
            Sharpe ratio.
        """
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

        return sharpe

    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            prices: Array of prices.

        Returns:
            Maximum drawdown (as decimal).
        """
        # Calculate cumulative returns
        cumulative = np.maximum.accumulate(prices)

        # Calculate drawdowns
        drawdowns = (prices - cumulative) / cumulative

        # Get maximum drawdown
        max_drawdown = np.min(drawdowns)

        return max_drawdown

    def evaluate_trading_strategy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate predictions as a trading strategy.

        Args:
            y_true: True prices.
            y_pred: Predicted prices.

        Returns:
            Dictionary of trading metrics.
        """
        # Calculate returns
        actual_returns = np.diff(y_true) / y_true[:-1]
        predicted_direction = np.diff(y_pred) > 0

        # Strategy returns (go long when predicting up, short when predicting down)
        strategy_returns = np.where(predicted_direction, actual_returns, -actual_returns)

        metrics = {
            'total_return': np.sum(strategy_returns),
            'average_return': np.mean(strategy_returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(strategy_returns),
            'win_rate': np.mean(strategy_returns > 0),
            'max_drawdown': self.calculate_max_drawdown(np.cumprod(1 + strategy_returns))
        }

        logger.info("Trading Strategy Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def print_summary(
        self,
        metrics: Dict[str, float],
        model_name: str = "Model"
    ) -> None:
        """
        Print a formatted summary of metrics.

        Args:
            metrics: Dictionary of metrics.
            model_name: Name of the model.
        """
        print(f"\n{'=' * 60}")
        print(f"{model_name} Performance Summary")
        print(f"{'=' * 60}")

        for metric_name, value in metrics.items():
            formatted_name = metric_name.replace('_', ' ').title()
            if 'accuracy' in metric_name or 'r2' in metric_name:
                print(f"{formatted_name:40s}: {value:.2%}")
            else:
                print(f"{formatted_name:40s}: {value:.4f}")

        print(f"{'=' * 60}\n")
