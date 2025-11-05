"""Base model class for stock price prediction."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import joblib

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for prediction models."""

    def __init__(self, config: dict, model_name: str):
        """
        Initialize BaseModel.

        Args:
            config: Configuration dictionary.
            model_name: Name of the model.
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_fitted = False

        # Model persistence config
        self.persistence_config = config.get('model_persistence', {})
        self.save_dir = self.persistence_config.get('save_dir', 'models/trained')

    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the model.

        Returns:
            Model instance.
        """
        pass

    def fit(self, X, y, **kwargs):
        """
        Fit the model.

        Args:
            X: Training features.
            y: Training target.
            **kwargs: Additional arguments for model fitting.

        Returns:
            Self for method chaining.
        """
        if self.model is None:
            self.model = self.build_model()

        logger.info(f"Training {self.model_name}...")

        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

        logger.info(f"{self.model_name} training completed")

        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features for prediction.

        Returns:
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before making predictions")

        return self.model.predict(X)

    def score(self, X, y):
        """
        Calculate score on given data.

        Args:
            X: Features.
            y: True target values.

        Returns:
            Model score (R² for regression).
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before scoring")

        return self.model.score(X, y)

    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save model to disk.

        Args:
            filename: Filename for saved model. If None, uses model_name.

        Returns:
            Path to saved model.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before saving")

        # Create save directory if it doesn't exist
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if filename is None:
            filename = f"{self.model_name.lower().replace(' ', '_')}.pkl"

        filepath = save_dir / filename

        # Save model
        joblib.dump(self.model, filepath)

        logger.info(f"Model saved to {filepath}")

        return str(filepath)

    def load_model(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath: Path to saved model.

        Returns:
            Self for method chaining.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = joblib.load(filepath)
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

        return self

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        if self.model is None:
            return {}

        return self.model.get_params()

    def set_params(self, **params):
        """
        Set model parameters.

        Args:
            **params: Model parameters to set.

        Returns:
            Self for method chaining.
        """
        if self.model is None:
            self.model = self.build_model()

        self.model.set_params(**params)

        return self

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', is_fitted={self.is_fitted})"
