"""Deep learning models for stock price prediction with LSTM/RNN."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, optimizers
import keras_tuner as kt

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMModel:
    """LSTM model for stock price prediction with multiple hidden layers."""

    def __init__(
        self,
        config: dict,
        sequence_length: int = 60,
        n_features: int = 5,
        lstm_units: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.

        Args:
            config: Configuration dictionary.
            sequence_length: Length of input sequences.
            n_features: Number of features.
            lstm_units: List of LSTM units for each layer.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for optimizer.
        """
        self.config = config
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Returns:
            Compiled Keras model.
        """
        model = models.Sequential(name="LSTM_Model")

        # Input layer
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            # Return sequences for all but the last LSTM layer
            return_sequences = i < len(self.lstm_units) - 1

            model.add(layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                name=f"lstm_{i+1}"
            ))

            # Dropout for regularization
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}"))

        # Dense layers
        model.add(layers.Dense(32, activation='relu', name="dense_1"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_dense"))
        model.add(layers.Dense(16, activation='relu', name="dense_2"))

        # Output layer
        model.add(layers.Dense(1, name="output"))

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )

        logger.info(f"Built LSTM model with {len(self.lstm_units)} LSTM layers")
        model.summary(print_fn=logger.info)

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the LSTM model.

        Args:
            X_train: Training sequences.
            y_train: Training targets.
            X_val: Validation sequences (optional).
            y_val: Validation targets (optional).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            verbose: Verbosity level.

        Returns:
            Training history.
        """
        if self.model is None:
            self.build_model()

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        # Train model
        logger.info(f"Training LSTM model for {epochs} epochs...")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        logger.info("Training completed")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences.

        Returns:
            Predictions.
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")

        return self.model.predict(X, verbose=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Test sequences.
            y: Test targets.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")

        results = self.model.evaluate(X, y, verbose=0)

        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mape': results[2]
        }

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def save_model(self, filepath: Optional[str] = None):
        """
        Save model to disk.

        Args:
            filepath: Path to save model. If None, uses default path.
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        if filepath is None:
            save_dir = Path(self.config.get('model_persistence', {}).get('save_dir', 'models/trained'))
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / "lstm_model.keras"

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath: Path to saved model.
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class GRUModel(LSTMModel):
    """GRU model (faster alternative to LSTM)."""

    def build_model(self) -> keras.Model:
        """Build GRU model architecture."""
        model = models.Sequential(name="GRU_Model")

        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # GRU layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1

            model.add(layers.GRU(
                units=units,
                return_sequences=return_sequences,
                name=f"gru_{i+1}"
            ))

            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}"))

        # Dense layers
        model.add(layers.Dense(32, activation='relu', name="dense_1"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_dense"))
        model.add(layers.Dense(16, activation='relu', name="dense_2"))

        # Output layer
        model.add(layers.Dense(1, name="output"))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )

        logger.info(f"Built GRU model with {len(self.lstm_units)} GRU layers")
        model.summary(print_fn=logger.info)

        self.model = model
        return model


class BidirectionalLSTMModel(LSTMModel):
    """Bidirectional LSTM model for capturing patterns from both directions."""

    def build_model(self) -> keras.Model:
        """Build Bidirectional LSTM model architecture."""
        model = models.Sequential(name="BiLSTM_Model")

        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1

            model.add(layers.Bidirectional(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences
                ),
                name=f"bilstm_{i+1}"
            ))

            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}"))

        # Dense layers
        model.add(layers.Dense(32, activation='relu', name="dense_1"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_dense"))
        model.add(layers.Dense(16, activation='relu', name="dense_2"))

        # Output layer
        model.add(layers.Dense(1, name="output"))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )

        logger.info(f"Built Bidirectional LSTM model with {len(self.lstm_units)} BiLSTM layers")
        model.summary(print_fn=logger.info)

        self.model = model
        return model


class LSTMHyperModel(kt.HyperModel):
    """Hyperparameter tunable LSTM model using Keras Tuner."""

    def __init__(self, sequence_length: int, n_features: int):
        """
        Initialize LSTMHyperModel.

        Args:
            sequence_length: Length of input sequences.
            n_features: Number of features.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Build model with hyperparameters to tune.

        Args:
            hp: HyperParameters instance.

        Returns:
            Compiled Keras model.
        """
        model = models.Sequential(name="LSTM_Tunable")

        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # Tune number of LSTM layers
        n_layers = hp.Int('n_lstm_layers', min_value=1, max_value=4, default=3)

        for i in range(n_layers):
            # Tune units per layer
            units = hp.Int(
                f'lstm_units_{i}',
                min_value=32,
                max_value=256,
                step=32,
                default=128
            )

            return_sequences = i < n_layers - 1

            # Choose between LSTM, GRU, or Bidirectional LSTM
            layer_type = hp.Choice(f'layer_type_{i}', ['lstm', 'gru', 'bilstm'], default='lstm')

            if layer_type == 'lstm':
                model.add(layers.LSTM(units=units, return_sequences=return_sequences))
            elif layer_type == 'gru':
                model.add(layers.GRU(units=units, return_sequences=return_sequences))
            else:  # bilstm
                model.add(layers.Bidirectional(
                    layers.LSTM(units=units, return_sequences=return_sequences)
                ))

            # Tune dropout rate
            dropout_rate = hp.Float(
                f'dropout_{i}',
                min_value=0.0,
                max_value=0.5,
                step=0.1,
                default=0.2
            )
            model.add(layers.Dropout(dropout_rate))

        # Tune dense layer size
        dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16, default=32)
        model.add(layers.Dense(dense_units, activation='relu'))

        dense_dropout = hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        model.add(layers.Dropout(dense_dropout))

        # Output layer
        model.add(layers.Dense(1))

        # Tune learning rate
        learning_rate = hp.Float(
            'learning_rate',
            min_value=1e-5,
            max_value=1e-2,
            sampling='log',
            default=1e-3
        )

        # Tune optimizer
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')

        if optimizer_choice == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )

        return model


class LSTMTuner:
    """LSTM hyperparameter tuner using Keras Tuner."""

    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        tuner_type: str = 'bayesian',
        max_trials: int = 50,
        executions_per_trial: int = 1,
        directory: str = 'tuner_results',
        project_name: str = 'lstm_tuning'
    ):
        """
        Initialize LSTM Tuner.

        Args:
            sequence_length: Length of input sequences.
            n_features: Number of features.
            tuner_type: Type of tuner ('random', 'bayesian', 'hyperband').
            max_trials: Maximum number of trials.
            executions_per_trial: Number of executions per trial.
            directory: Directory to save tuning results.
            project_name: Name of the tuning project.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.tuner_type = tuner_type
        self.max_trials = max_trials

        # Create hypermodel
        hypermodel = LSTMHyperModel(sequence_length, n_features)

        # Select tuner
        if tuner_type == 'random':
            self.tuner = kt.RandomSearch(
                hypermodel,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=directory,
                project_name=project_name
            )
        elif tuner_type == 'bayesian':
            self.tuner = kt.BayesianOptimization(
                hypermodel,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=directory,
                project_name=project_name
            )
        elif tuner_type == 'hyperband':
            self.tuner = kt.Hyperband(
                hypermodel,
                objective='val_loss',
                max_epochs=100,
                factor=3,
                directory=directory,
                project_name=project_name
            )
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")

        logger.info(f"Initialized {tuner_type} tuner with {max_trials} max trials")

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Search for best hyperparameters.

        Args:
            X_train: Training sequences.
            y_train: Training targets.
            X_val: Validation sequences.
            y_val: Validation targets.
            epochs: Number of epochs per trial.
            batch_size: Batch size.
        """
        logger.info("Starting hyperparameter search...")

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Search
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        logger.info("Hyperparameter search completed")

    def get_best_model(self) -> keras.Model:
        """
        Get the best model from tuning.

        Returns:
            Best Keras model.
        """
        best_model = self.tuner.get_best_models(num_models=1)[0]
        logger.info("Retrieved best model")

        return best_model

    def get_best_hyperparameters(self) -> Dict:
        """
        Get the best hyperparameters.

        Returns:
            Dictionary of best hyperparameters.
        """
        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        hp_dict = best_hp.values

        logger.info(f"Best hyperparameters: {hp_dict}")

        return hp_dict

    def results_summary(self, num_trials: int = 10):
        """
        Print summary of tuning results.

        Args:
            num_trials: Number of top trials to show.
        """
        self.tuner.results_summary(num_trials=num_trials)
