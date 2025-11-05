"""Visualization module with reusable plotting functions."""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Visualizer:
    """Create visualizations for stock price prediction."""

    def __init__(self, config: dict):
        """
        Initialize Visualizer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.viz_config = config.get('visualization', {})

        # Set style
        self.figure_size = tuple(self.viz_config.get('figure_size', [10, 6]))
        self.dpi = self.viz_config.get('dpi', 100)
        plt.style.use(self.viz_config.get('style', 'seaborn-v0_8'))
        self.palette = self.viz_config.get('palette', 'Set2')

        # Create plot directory if needed
        self.plot_dir = self.viz_config.get('plot_dir', 'reports/figures')
        if self.viz_config.get('save_plots', True):
            Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot actual vs predicted values.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            title: Plot title.
            save_name: Filename to save plot.
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot residuals.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            title: Plot title.
            save_name: Filename to save plot.
        """
        residuals = y_true - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1]), dpi=self.dpi)

        # Residual scatter plot
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Residual distribution
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_time_series(
        self,
        y_true: pd.Series,
        y_pred: Optional[pd.Series] = None,
        title: str = "Time Series Plot",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot time series data.

        Args:
            y_true: True values with datetime index.
            y_pred: Predicted values (optional).
            title: Plot title.
            save_name: Filename to save plot.
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Plot actual values
        ax.plot(y_true.index, y_true.values, label='Actual', linewidth=2)

        # Plot predicted values if provided
        if y_pred is not None:
            ax.plot(y_pred.index, y_pred.values, label='Predicted', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Error Distribution",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot error distribution.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            title: Plot title.
            save_name: Filename to save plot.
        """
        errors = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 1.5), dpi=self.dpi)

        # Histogram
        axes[0, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_title('Error Histogram')
        axes[0, 0].set_xlabel('Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Box plot
        axes[0, 1].boxplot(errors)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_title('Error Box Plot')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Absolute errors
        abs_errors = np.abs(errors)
        axes[1, 1].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_title('Absolute Error Distribution')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 20,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_importance: Dictionary mapping feature names to importance scores.
            title: Plot title.
            top_n: Number of top features to display.
            save_name: Filename to save plot.
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Heatmap",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot correlation heatmap.

        Args:
            df: DataFrame.
            title: Plot title.
            save_name: Filename to save plot.
        """
        # Calculate correlation matrix
        corr = df.corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Plot heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'test_r2_score',
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot model comparison.

        Args:
            comparison_df: DataFrame with model comparison metrics.
            metric: Metric to compare.
            title: Plot title.
            save_name: Filename to save plot.
        """
        if title is None:
            title = f"Model Comparison - {metric.replace('_', ' ').title()}"

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Bar plot
        comparison_df[metric].plot(kind='barh', ax=ax, color=sns.color_palette(self.palette))

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curve",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot learning curve.

        Args:
            train_sizes: Array of training set sizes.
            train_scores: Training scores for each size.
            val_scores: Validation scores for each size.
            title: Plot title.
            save_name: Filename to save plot.
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Plot learning curve
        ax.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
        ax.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_price_with_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        indicators: Optional[List[str]] = None,
        title: str = "Price with Technical Indicators",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot price with technical indicators.

        Args:
            df: DataFrame with price and indicators.
            price_col: Column name for price.
            indicators: List of indicator column names.
            title: Plot title.
            save_name: Filename to save plot.
        """
        fig, ax = plt.subplots(figsize=(self.figure_size[0] * 1.5, self.figure_size[1]), dpi=self.dpi)

        # Plot price
        ax.plot(df.index, df[price_col], label=price_col, linewidth=2)

        # Plot indicators
        if indicators:
            for indicator in indicators:
                if indicator in df.columns:
                    ax.plot(df.index, df[indicator], label=indicator, alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def plot_pairplot(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ) -> None:
        """
        Create pairplot for selected columns.

        Args:
            df: DataFrame.
            columns: List of columns to plot. If None, uses all numeric columns.
            save_name: Filename to save plot.
        """
        if columns:
            plot_df = df[columns]
        else:
            plot_df = df.select_dtypes(include=[np.number])

        # Limit to reasonable number of columns
        if len(plot_df.columns) > 10:
            logger.warning(f"Too many columns ({len(plot_df.columns)}), using first 10")
            plot_df = plot_df.iloc[:, :10]

        sns.pairplot(plot_df, diag_kind='kde', corner=True)

        if save_name:
            self._save_plot(save_name)

        plt.show()

    def _save_plot(self, filename: str) -> None:
        """
        Save plot to file.

        Args:
            filename: Filename for saved plot.
        """
        if self.viz_config.get('save_plots', True):
            filepath = Path(self.plot_dir) / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
