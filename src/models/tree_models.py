"""Tree-based models with hyperparameter tuning."""

import logging
from typing import Optional, Dict

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TreeModels:
    """Factory class for tree-based models."""

    def __init__(self, config: dict):
        """
        Initialize TreeModels.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.models_config = config.get('models', {})

    def get_decision_tree(self) -> DecisionTreeRegressor:
        """
        Get Decision Tree Regressor model.

        Returns:
            DecisionTreeRegressor model.
        """
        dt_config = self.models_config.get('decision_tree', {})

        model = DecisionTreeRegressor(**dt_config)

        logger.info("Created Decision Tree Regressor")

        return model

    def get_random_forest(self) -> RandomForestRegressor:
        """
        Get Random Forest Regressor model.

        Returns:
            RandomForestRegressor model.
        """
        rf_config = self.models_config.get('random_forest', {})

        model = RandomForestRegressor(**rf_config)

        logger.info("Created Random Forest Regressor")

        return model

    def get_gradient_boosting(self) -> GradientBoostingRegressor:
        """
        Get Gradient Boosting Regressor model.

        Returns:
            GradientBoostingRegressor model.
        """
        gb_config = self.models_config.get('gradient_boosting', {})

        model = GradientBoostingRegressor(**gb_config)

        logger.info("Created Gradient Boosting Regressor")

        return model


class DecisionTreeModel(BaseModel):
    """Decision Tree model wrapper with hyperparameter tuning."""

    def __init__(self, config: dict):
        """
        Initialize DecisionTreeModel.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config, model_name="Decision Tree Regression")

    def build_model(self):
        """Build Decision Tree model."""
        factory = TreeModels(self.config)
        return factory.get_decision_tree()

    def tune_hyperparameters(self, X, y, method: str = 'randomized', cv=5):
        """
        Tune hyperparameters using cross-validation.

        Args:
            X: Training features.
            y: Training target.
            method: 'grid' or 'randomized' search.
            cv: Cross-validation splitter or number of folds.

        Returns:
            Best model.
        """
        tuning_config = self.config.get('hyperparameter_tuning', {})
        param_grid = tuning_config.get('decision_tree_param_grid', {})

        base_model = self.build_model()

        logger.info(f"Tuning {self.model_name} hyperparameters using {method} search...")

        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=tuning_config.get('scoring', 'neg_mean_squared_error'),
                n_jobs=-1,
                verbose=1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=cv,
                n_iter=tuning_config.get('n_iter', 50),
                scoring=tuning_config.get('scoring', 'neg_mean_squared_error'),
                n_jobs=-1,
                random_state=tuning_config.get('random_state', 42),
                verbose=1
            )

        search.fit(X, y)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")

        self.model = search.best_estimator_
        self.is_fitted = True

        return self

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted first")

        # Get feature importances
        importances = self.model.feature_importances_

        return dict(zip(range(len(importances)), importances))


class RandomForestModel(BaseModel):
    """Random Forest model wrapper with hyperparameter tuning."""

    def __init__(self, config: dict):
        """
        Initialize RandomForestModel.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config, model_name="Random Forest Regression")

    def build_model(self):
        """Build Random Forest model."""
        factory = TreeModels(self.config)
        return factory.get_random_forest()

    def tune_hyperparameters(self, X, y, method: str = 'randomized', cv=5):
        """
        Tune hyperparameters using cross-validation.

        Args:
            X: Training features.
            y: Training target.
            method: 'grid' or 'randomized' search.
            cv: Cross-validation splitter or number of folds.

        Returns:
            Best model.
        """
        tuning_config = self.config.get('hyperparameter_tuning', {})
        param_grid = tuning_config.get('random_forest_param_grid', {})

        base_model = self.build_model()

        logger.info(f"Tuning {self.model_name} hyperparameters using {method} search...")

        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=tuning_config.get('scoring', 'neg_mean_squared_error'),
                n_jobs=-1,
                verbose=1
            )
        else:  # randomized
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=cv,
                n_iter=tuning_config.get('n_iter', 50),
                scoring=tuning_config.get('scoring', 'neg_mean_squared_error'),
                n_jobs=-1,
                random_state=tuning_config.get('random_state', 42),
                verbose=1
            )

        search.fit(X, y)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")

        self.model = search.best_estimator_
        self.is_fitted = True

        return self

    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: List of feature names. If None, uses indices.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted first")

        # Get feature importances
        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(zip(feature_names, importances))


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model wrapper."""

    def __init__(self, config: dict):
        """
        Initialize GradientBoostingModel.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config, model_name="Gradient Boosting Regression")

    def build_model(self):
        """Build Gradient Boosting model."""
        factory = TreeModels(self.config)
        return factory.get_gradient_boosting()

    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: List of feature names. If None, uses indices.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted first")

        # Get feature importances
        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(zip(feature_names, importances))
