"""
Example usage of LSTM/RNN models for stock price prediction.

This script demonstrates:
1. Loading and preprocessing data for LSTM
2. Creating sequences for time series
3. Training LSTM, GRU, and Bidirectional LSTM models
4. Hyperparameter tuning with Keras Tuner
5. Evaluating and comparing models
6. Visualizing predictions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data.loader import DataLoader
from src.data.sequence_generator import SequenceGenerator
from src.models.deep_learning_models import (
    LSTMModel,
    GRUModel,
    BidirectionalLSTMModel,
    LSTMTuner
)
from src.evaluation.metrics import ModelEvaluator
from src.visualization.plots import Visualizer


def main():
    """Main execution function."""

    print("=" * 80)
    print("LSTM/RNN Stock Price Prediction - Example Usage")
    print("=" * 80)

    # Setup
    config = load_config()
    logger = setup_logger('lstm_example', level='INFO')
    logger.info("Starting LSTM example")

    # Load configuration
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 60)
    lstm_units = lstm_config.get('lstm_units', [128, 64, 32])
    dropout_rate = lstm_config.get('dropout_rate', 0.2)
    learning_rate = lstm_config.get('learning_rate', 0.001)
    epochs = lstm_config.get('epochs', 100)
    batch_size = lstm_config.get('batch_size', 32)

    # Load data
    print("\n1. Loading data...")
    loader = DataLoader(config)
    train_df, test_df = loader.load_train_test_data()

    print(f"   Train set: {train_df.shape}")
    print(f"   Test set: {test_df.shape}")

    # Use only basic features for LSTM (to avoid too many NaN from feature engineering)
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'Stock Trading']
    target_column = 'Close'

    # Prepare sequences
    print(f"\n2. Creating sequences (sequence_length={sequence_length})...")
    seq_gen = SequenceGenerator(
        sequence_length=sequence_length,
        prediction_horizon=1
    )

    X_train_seq, y_train_seq, _, _, X_test_seq, y_test_seq = seq_gen.prepare_train_val_test(
        train_df,
        None,  # No validation set for this example
        test_df,
        target_column=target_column,
        feature_columns=feature_columns
    )

    print(f"   Train sequences: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"   Test sequences: X={X_test_seq.shape}, y={y_test_seq.shape}")

    n_features = X_train_seq.shape[2]

    # Split training data for validation
    val_split = 0.2
    n_val = int(len(X_train_seq) * val_split)

    X_val = X_train_seq[-n_val:]
    y_val = y_train_seq[-n_val:]
    X_train = X_train_seq[:-n_val]
    y_train = y_train_seq[:-n_val]

    print(f"   Split train into train={X_train.shape[0]} and val={X_val.shape[0]}")

    # Train LSTM Model
    print("\n3. Training LSTM Model...")
    print(f"   Architecture: {lstm_units} units with {dropout_rate} dropout")

    lstm_model = LSTMModel(
        config=config,
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    lstm_history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate LSTM
    print("\n4. Evaluating LSTM Model...")
    lstm_metrics = lstm_model.evaluate(X_test_seq, y_test_seq)

    # Make predictions
    y_train_pred_scaled = lstm_model.predict(X_train_seq)
    y_test_pred_scaled = lstm_model.predict(X_test_seq)

    # Inverse transform predictions
    y_train_pred = seq_gen.inverse_transform_predictions(y_train_pred_scaled)
    y_test_pred = seq_gen.inverse_transform_predictions(y_test_pred_scaled)

    y_train_actual = seq_gen.inverse_transform_predictions(y_train_seq)
    y_test_actual = seq_gen.inverse_transform_predictions(y_test_seq)

    # Calculate metrics on original scale
    evaluator = ModelEvaluator(config)
    lstm_eval = evaluator.evaluate(y_test_actual, y_test_pred, set_name="LSTM Test")

    # Train GRU Model
    print("\n5. Training GRU Model (faster alternative)...")

    gru_model = GRUModel(
        config=config,
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    gru_history = gru_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    y_gru_pred_scaled = gru_model.predict(X_test_seq)
    y_gru_pred = seq_gen.inverse_transform_predictions(y_gru_pred_scaled)

    gru_eval = evaluator.evaluate(y_test_actual, y_gru_pred, set_name="GRU Test")

    # Train Bidirectional LSTM
    print("\n6. Training Bidirectional LSTM...")

    bilstm_model = BidirectionalLSTMModel(
        config=config,
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=[64, 32],  # Fewer units as bidirectional doubles them
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    bilstm_history = bilstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    y_bilstm_pred_scaled = bilstm_model.predict(X_test_seq)
    y_bilstm_pred = seq_gen.inverse_transform_predictions(y_bilstm_pred_scaled)

    bilstm_eval = evaluator.evaluate(y_test_actual, y_bilstm_pred, set_name="BiLSTM Test")

    # Compare models
    print("\n7. Model Comparison:")
    print("=" * 80)
    comparison_data = {
        'LSTM': lstm_eval,
        'GRU': gru_eval,
        'BiLSTM': bilstm_eval
    }

    comparison_df = pd.DataFrame(comparison_data).T
    print(comparison_df[['r2_score', 'root_mean_squared_error', 'mean_absolute_error']])

    # Save best model
    print("\n8. Saving best model...")
    best_model_name = comparison_df['r2_score'].idxmax()

    if best_model_name == 'LSTM':
        lstm_model.save_model('models/trained/lstm_best.keras')
    elif best_model_name == 'GRU':
        gru_model.save_model('models/trained/gru_best.keras')
    else:
        bilstm_model.save_model('models/trained/bilstm_best.keras')

    print(f"   Best model: {best_model_name}")

    # Hyperparameter Tuning Example (optional, commented out due to time)
    print("\n9. Hyperparameter Tuning (optional, set run_tuning=True to enable)...")
    run_tuning = False  # Set to True to run tuning

    if run_tuning:
        print("   Starting hyperparameter search...")

        tuning_config = config.get('deep_learning_tuning', {})

        tuner = LSTMTuner(
            sequence_length=sequence_length,
            n_features=n_features,
            tuner_type=tuning_config.get('tuner_type', 'bayesian'),
            max_trials=tuning_config.get('max_trials', 10),  # Reduced for demo
            directory=tuning_config.get('directory', 'tuner_results'),
            project_name=tuning_config.get('project_name', 'lstm_stock_prediction')
        )

        tuner.search(
            X_train, y_train,
            X_val, y_val,
            epochs=tuning_config.get('search_epochs', 30),
            batch_size=tuning_config.get('batch_size', 32)
        )

        # Get best model
        best_tuned_model = tuner.get_best_model()
        best_hp = tuner.get_best_hyperparameters()

        print(f"   Best hyperparameters: {best_hp}")

        # Evaluate tuned model
        y_tuned_pred_scaled = best_tuned_model.predict(X_test_seq, verbose=0)
        y_tuned_pred = seq_gen.inverse_transform_predictions(y_tuned_pred_scaled)

        tuned_eval = evaluator.evaluate(y_test_actual, y_tuned_pred, set_name="Tuned LSTM Test")
        print(f"   Tuned model R²: {tuned_eval['r2_score']:.4f}")
    else:
        print("   Skipping hyperparameter tuning (set run_tuning=True to enable)")

    # Visualizations
    print("\n10. Creating visualizations...")
    viz = Visualizer(config)

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(lstm_history.history['loss'], label='LSTM Train')
    axes[0].plot(lstm_history.history['val_loss'], label='LSTM Val')
    axes[0].plot(gru_history.history['loss'], label='GRU Train', linestyle='--')
    axes[0].plot(gru_history.history['val_loss'], label='GRU Val', linestyle='--')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(lstm_history.history['mae'], label='LSTM Train')
    axes[1].plot(lstm_history.history['val_mae'], label='LSTM Val')
    axes[1].plot(gru_history.history['mae'], label='GRU Train', linestyle='--')
    axes[1].plot(gru_history.history['val_mae'], label='GRU Val', linestyle='--')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/lstm_training_history.png', dpi=100)
    print("   Saved training history plot")

    # Actual vs Predicted
    viz.plot_actual_vs_predicted(
        y_test_actual,
        y_test_pred,
        title="LSTM: Actual vs Predicted",
        save_name="lstm_actual_vs_pred.png"
    )

    # Time series plot
    test_dates = test_df.index[sequence_length:]
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, y_test_actual, label='Actual', linewidth=2)
    plt.plot(test_dates, y_test_pred, label='LSTM', linewidth=2, alpha=0.7)
    plt.plot(test_dates, y_gru_pred, label='GRU', linewidth=2, alpha=0.7)
    plt.plot(test_dates, y_bilstm_pred, label='BiLSTM', linewidth=2, alpha=0.7)
    plt.title('LSTM/RNN Models: Time Series Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/lstm_time_series_comparison.png', dpi=100)
    print("   Saved time series comparison plot")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Best Model: {best_model_name}")
    print(f"Test R² Score: {comparison_df.loc[best_model_name, 'r2_score']:.4f}")
    print(f"Test RMSE: {comparison_df.loc[best_model_name, 'root_mean_squared_error']:.2f}")
    print(f"Test MAE: {comparison_df.loc[best_model_name, 'mean_absolute_error']:.2f}")
    print("\nLSTM models trained, evaluated, and visualizations created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
