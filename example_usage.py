"""
Example usage of the Stock Price Prediction framework.

This script demonstrates how to use the modular components to:
1. Load and preprocess data
2. Engineer features
3. Train multiple models
4. Evaluate and compare models
5. Visualize results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.linear_models import LinearRegressionModel, SVRModel
from src.models.tree_models import DecisionTreeModel, RandomForestModel
from src.models.ensemble_models import VotingRegressorModel
from src.evaluation.metrics import ModelEvaluator
from src.visualization.plots import Visualizer


def main():
    """Main execution function."""

    # Setup
    print("=" * 80)
    print("Stock Price Prediction - Example Usage")
    print("=" * 80)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    logger = setup_logger('example', level='INFO')
    logger.info("Configuration loaded successfully")

    # Load data
    print("\n2. Loading data...")
    loader = DataLoader(config)
    train_df, test_df = loader.load_train_test_data()

    print(f"   Train set shape: {train_df.shape}")
    print(f"   Test set shape: {test_df.shape}")

    # Feature engineering
    print("\n3. Engineering features...")
    engineer = FeatureEngineer(config)

    # Note: Using only basic features for this example
    # Full feature engineering creates many NaN values with small test set
    train_df = engineer.add_price_features(train_df)
    train_df = engineer.add_volume_features(train_df)
    train_df = engineer.add_temporal_features(train_df)

    test_df = engineer.add_price_features(test_df)
    test_df = engineer.add_volume_features(test_df)
    test_df = engineer.add_temporal_features(test_df)

    print(f"   Features created. New shape: {train_df.shape}")

    # Prepare data
    print("\n4. Preparing data...")
    X_train, y_train = loader.get_feature_target_split(train_df)
    X_test, y_test = loader.get_feature_target_split(test_df)

    # Handle NaN values
    X_train = X_train.fillna(method='ffill').fillna(method='bfill')
    X_test = X_test.fillna(method='ffill').fillna(method='bfill')

    # Ensure indices match
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")

    # Train models
    print("\n5. Training models...")

    models = {}

    # Linear Regression
    print("   - Training Linear Regression...")
    lr_model = LinearRegressionModel(config)
    lr_model.fit(X_train, y_train)
    models['Linear Regression'] = lr_model.model

    # SVR
    print("   - Training Support Vector Regression...")
    svr_model = SVRModel(config)
    svr_model.fit(X_train, y_train)
    models['SVR'] = svr_model.model

    # Decision Tree
    print("   - Training Decision Tree...")
    dt_model = DecisionTreeModel(config)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model.model

    # Random Forest
    print("   - Training Random Forest...")
    rf_model = RandomForestModel(config)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model.model

    # Save best model
    print("\n6. Saving best model...")
    rf_model.save_model()
    print(f"   Model saved to: models/trained/")

    # Evaluate and compare
    print("\n7. Evaluating models...")
    evaluator = ModelEvaluator(config)

    comparison = evaluator.compare_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print("\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)
    print(comparison[['train_r2_score', 'test_r2_score',
                     'test_root_mean_squared_error', 'test_mean_absolute_error']])

    # Visualizations
    print("\n8. Creating visualizations...")
    viz = Visualizer(config)

    # Get predictions from best model (Random Forest)
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Create plots
    print("   - Actual vs Predicted plot...")
    viz.plot_actual_vs_predicted(
        y_test.values,
        y_test_pred,
        title="Random Forest: Actual vs Predicted",
        save_name="rf_actual_vs_pred.png"
    )

    print("   - Residuals plot...")
    viz.plot_residuals(
        y_test.values,
        y_test_pred,
        title="Random Forest: Residuals",
        save_name="rf_residuals.png"
    )

    print("   - Time series plot...")
    y_test_pred_series = pd.Series(y_test_pred, index=y_test.index)
    viz.plot_time_series(
        y_test,
        y_test_pred_series,
        title="Random Forest: Time Series Prediction",
        save_name="rf_timeseries.png"
    )

    print("   - Feature importance plot...")
    importance = rf_model.get_feature_importance(X_train.columns)
    viz.plot_feature_importance(
        importance,
        title="Random Forest: Feature Importance",
        top_n=15,
        save_name="rf_feature_importance.png"
    )

    print("   - Model comparison plot...")
    viz.plot_model_comparison(
        comparison,
        metric='test_r2_score',
        save_name="model_comparison.png"
    )

    print(f"\n   Plots saved to: {viz.plot_dir}/")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    best_model_name = comparison['test_r2_score'].idxmax()
    best_r2 = comparison.loc[best_model_name, 'test_r2_score']
    best_rmse = comparison.loc[best_model_name, 'test_root_mean_squared_error']

    print(f"Best Model: {best_model_name}")
    print(f"Test R² Score: {best_r2:.4f}")
    print(f"Test RMSE: {best_rmse:.2f}")
    print("\nAll models trained, evaluated, and visualizations created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
