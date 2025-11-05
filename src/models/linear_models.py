"""Linear regression models with pipelines."""

import logging
from typing import Optional

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LinearModels:
    """Factory class for linear models."""

    def __init__(self, config: dict):
        """
        Initialize LinearModels.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.models_config = config.get('models', {})

    def get_linear_regression(self) -> Pipeline:
        """
        Get Linear Regression model with pipeline.

        Returns:
            Pipeline with LinearRegression model.
        """
        lr_config = self.models_config.get('linear_regression', {})

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression(**lr_config))
        ])

        logger.info("Created Linear Regression pipeline")

        return model

    def get_ridge_regression(self, alpha: float = 1.0) -> Pipeline:
        """
        Get Ridge Regression model with pipeline.

        Args:
            alpha: Regularization strength.

        Returns:
            Pipeline with Ridge model.
        """
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha))
        ])

        logger.info(f"Created Ridge Regression pipeline (alpha={alpha})")

        return model

    def get_lasso_regression(self, alpha: float = 1.0) -> Pipeline:
        """
        Get Lasso Regression model with pipeline.

        Args:
            alpha: Regularization strength.

        Returns:
            Pipeline with Lasso model.
        """
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=alpha, max_iter=10000))
        ])

        logger.info(f"Created Lasso Regression pipeline (alpha={alpha})")

        return model

    def get_svr(self) -> Pipeline:
        """
        Get Support Vector Regression model with pipeline.

        Returns:
            Pipeline with SVR model.
        """
        svr_config = self.models_config.get('svr', {})

        model = Pipeline([
            ('scaler', MaxAbsScaler()),
            ('svr', SVR(**svr_config))
        ])

        logger.info("Created SVR pipeline")

        return model


class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""

    def __init__(self, config: dict):
        """
        Initialize LinearRegressionModel.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config, model_name="Linear Regression")

    def build_model(self):
        """Build Linear Regression pipeline."""
        factory = LinearModels(self.config)
        return factory.get_linear_regression()


class RidgeRegressionModel(BaseModel):
    """Ridge Regression model wrapper."""

    def __init__(self, config: dict, alpha: float = 1.0):
        """
        Initialize RidgeRegressionModel.

        Args:
            config: Configuration dictionary.
            alpha: Regularization strength.
        """
        super().__init__(config, model_name="Ridge Regression")
        self.alpha = alpha

    def build_model(self):
        """Build Ridge Regression pipeline."""
        factory = LinearModels(self.config)
        return factory.get_ridge_regression(alpha=self.alpha)


class LassoRegressionModel(BaseModel):
    """Lasso Regression model wrapper."""

    def __init__(self, config: dict, alpha: float = 1.0):
        """
        Initialize LassoRegressionModel.

        Args:
            config: Configuration dictionary.
            alpha: Regularization strength.
        """
        super().__init__(config, model_name="Lasso Regression")
        self.alpha = alpha

    def build_model(self):
        """Build Lasso Regression pipeline."""
        factory = LinearModels(self.config)
        return factory.get_lasso_regression(alpha=self.alpha)


class SVRModel(BaseModel):
    """Support Vector Regression model wrapper."""

    def __init__(self, config: dict):
        """
        Initialize SVRModel.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config, model_name="Support Vector Regression")

    def build_model(self):
        """Build SVR pipeline."""
        factory = LinearModels(self.config)
        return factory.get_svr()
