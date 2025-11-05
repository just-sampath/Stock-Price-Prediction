"""Ensemble models for stock price prediction."""

import logging
from typing import List, Tuple, Optional

from sklearn.ensemble import VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.linear_model import Ridge

from .base_model import BaseModel
from .linear_models import LinearModels
from .tree_models import TreeModels
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EnsembleModels:
    """Factory class for ensemble models."""

    def __init__(self, config: dict):
        """
        Initialize EnsembleModels.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.models_config = config.get('models', {})

    def get_voting_regressor(
        self,
        estimators: List[Tuple[str, object]],
        weights: Optional[List[float]] = None
    ) -> VotingRegressor:
        """
        Get Voting Regressor ensemble.

        Args:
            estimators: List of (name, estimator) tuples.
            weights: Optional weights for each estimator.

        Returns:
            VotingRegressor model.
        """
        model = VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)

        logger.info(f"Created Voting Regressor with {len(estimators)} estimators")

        return model

    def get_stacking_regressor(
        self,
        estimators: List[Tuple[str, object]],
        final_estimator=None
    ) -> StackingRegressor:
        """
        Get Stacking Regressor ensemble.

        Args:
            estimators: List of (name, estimator) tuples for base models.
            final_estimator: Final estimator (meta-learner). If None, uses Ridge.

        Returns:
            StackingRegressor model.
        """
        if final_estimator is None:
            final_estimator = Ridge(alpha=1.0)

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1
        )

        logger.info(f"Created Stacking Regressor with {len(estimators)} base estimators")

        return model

    def get_bagging_regressor(
        self,
        base_estimator,
        n_estimators: int = 10
    ) -> BaggingRegressor:
        """
        Get Bagging Regressor ensemble.

        Args:
            base_estimator: Base estimator to use.
            n_estimators: Number of estimators in the ensemble.

        Returns:
            BaggingRegressor model.
        """
        model = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=self.models_config.get('random_seed', 42)
        )

        logger.info(f"Created Bagging Regressor with {n_estimators} estimators")

        return model


class VotingRegressorModel(BaseModel):
    """Voting Regressor ensemble wrapper."""

    def __init__(self, config: dict, estimators: Optional[List[Tuple[str, object]]] = None):
        """
        Initialize VotingRegressorModel.

        Args:
            config: Configuration dictionary.
            estimators: List of (name, estimator) tuples. If None, uses default set.
        """
        super().__init__(config, model_name="Voting Regressor")
        self.estimators = estimators

    def build_model(self):
        """Build Voting Regressor model."""
        # If no estimators provided, create default ensemble
        if self.estimators is None:
            linear_factory = LinearModels(self.config)
            tree_factory = TreeModels(self.config)

            self.estimators = [
                ('linear', linear_factory.get_linear_regression()),
                ('svr', linear_factory.get_svr()),
                ('rf', tree_factory.get_random_forest()),
                ('gb', tree_factory.get_gradient_boosting())
            ]

        factory = EnsembleModels(self.config)
        return factory.get_voting_regressor(self.estimators)


class StackingRegressorModel(BaseModel):
    """Stacking Regressor ensemble wrapper."""

    def __init__(
        self,
        config: dict,
        estimators: Optional[List[Tuple[str, object]]] = None,
        final_estimator=None
    ):
        """
        Initialize StackingRegressorModel.

        Args:
            config: Configuration dictionary.
            estimators: List of (name, estimator) tuples for base models.
            final_estimator: Final estimator (meta-learner).
        """
        super().__init__(config, model_name="Stacking Regressor")
        self.estimators = estimators
        self.final_estimator = final_estimator

    def build_model(self):
        """Build Stacking Regressor model."""
        # If no estimators provided, create default ensemble
        if self.estimators is None:
            linear_factory = LinearModels(self.config)
            tree_factory = TreeModels(self.config)

            self.estimators = [
                ('dt', tree_factory.get_decision_tree()),
                ('rf', tree_factory.get_random_forest()),
                ('gb', tree_factory.get_gradient_boosting())
            ]

        factory = EnsembleModels(self.config)
        return factory.get_stacking_regressor(self.estimators, self.final_estimator)


class BaggingRegressorModel(BaseModel):
    """Bagging Regressor ensemble wrapper."""

    def __init__(
        self,
        config: dict,
        base_estimator=None,
        n_estimators: int = 10
    ):
        """
        Initialize BaggingRegressorModel.

        Args:
            config: Configuration dictionary.
            base_estimator: Base estimator to use.
            n_estimators: Number of estimators in the ensemble.
        """
        super().__init__(config, model_name="Bagging Regressor")
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def build_model(self):
        """Build Bagging Regressor model."""
        # If no base estimator provided, use DecisionTree
        if self.base_estimator is None:
            tree_factory = TreeModels(self.config)
            self.base_estimator = tree_factory.get_decision_tree()

        factory = EnsembleModels(self.config)
        return factory.get_bagging_regressor(self.base_estimator, self.n_estimators)


class WeightedAverageEnsemble:
    """Simple weighted average ensemble."""

    def __init__(self, models: List[object], weights: Optional[List[float]] = None):
        """
        Initialize WeightedAverageEnsemble.

        Args:
            models: List of fitted models.
            weights: Optional weights for each model. If None, uses equal weights.
        """
        self.models = models
        self.weights = weights

        if self.weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        logger.info(f"Created Weighted Average Ensemble with {len(models)} models")

    def predict(self, X):
        """
        Make predictions using weighted average.

        Args:
            X: Features for prediction.

        Returns:
            Weighted average predictions.
        """
        predictions = []

        for model in self.models:
            predictions.append(model.predict(X))

        # Calculate weighted average
        import numpy as np
        weighted_pred = np.zeros_like(predictions[0])

        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight

        return weighted_pred
