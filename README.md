# Stock Price Prediction

A comprehensive machine learning framework for stock price prediction using traditional ML and deep learning models (LSTM/RNN) with proper time-series methodology, feature engineering, and dynamic hyperparameter optimization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Loading](#data-loading)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Models](#models)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a production-ready stock price prediction system with:

- **Proper time-series methodology** - Uses TimeSeriesSplit for cross-validation to prevent data leakage
- **Comprehensive feature engineering** - Technical indicators (RSI, MACD, Bollinger Bands), lag features, rolling statistics
- **Multiple ML models** - Linear Regression, SVR, Decision Trees, Random Forests, Gradient Boosting, and Ensemble methods
- **Deep Learning (LSTM/RNN)** - LSTM, GRU, and Bidirectional LSTM with multiple hidden layers
- **Dynamic hyperparameter tuning** - GridSearch, RandomizedSearch, and Keras Tuner with Bayesian Optimization
- **Advanced evaluation** - R², RMSE, MAE, MAPE, Directional Accuracy, and trading metrics
- **Professional visualizations** - Actual vs Predicted, Residuals, Time Series plots, Feature Importance
- **Modular architecture** - Clean, testable, and reusable code structure

## Features

✅ **Time Series Aware**
- Chronological train/validation/test splits
- TimeSeriesSplit cross-validation
- Lag features and rolling windows

✅ **Advanced Feature Engineering**
- Technical Indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- Price Features: Returns, Momentum, Gaps
- Volume Features: OBV, VPT, Relative Volume
- Temporal Features: Day of week, Month, Quarter

✅ **Multiple Models**
- Linear: Linear Regression, Ridge, Lasso, SVR
- Tree-based: Decision Tree, Random Forest, Gradient Boosting
- Ensemble: Voting, Stacking, Bagging, Weighted Average
- Deep Learning: LSTM, GRU, Bidirectional LSTM with hyperparameter tuning

✅ **Comprehensive Evaluation**
- Regression Metrics: R², RMSE, MAE, MAPE
- Time Series Metrics: Directional Accuracy
- Trading Metrics: Sharpe Ratio, Max Drawdown, Win Rate

✅ **Production Ready**
- Configuration management (YAML)
- Logging throughout
- Model persistence (save/load)
- Data validation
- Error handling

## Project Structure

```
Stock-Price-Prediction/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw data files
│   │   ├── Uniqlo - Training.csv
│   │   └── Uniqlo - Testing.csv
│   └── processed/               # Processed data
├── models/
│   └── trained/                 # Saved models
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Comparison.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading with validation
│   │   └── preprocessor.py     # Preprocessing & time-series splits
│   ├── features/
│   │   └── engineering.py      # Feature engineering
│   ├── models/
│   │   ├── base_model.py       # Base model class
│   │   ├── linear_models.py    # Linear models
│   │   ├── tree_models.py      # Tree-based models
│   │   └── ensemble_models.py  # Ensemble models
│   ├── evaluation/
│   │   └── metrics.py          # Evaluation metrics
│   ├── visualization/
│   │   └── plots.py            # Visualization functions
│   └── utils/
│       ├── config_loader.py    # Configuration loader
│       └── logger.py           # Logging setup
├── tests/                       # Unit tests
├── requirements.txt             # Dependencies
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/just-sampath/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.utils.config_loader import load_config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.models.tree_models import RandomForestModel
from src.evaluation.metrics import ModelEvaluator

# Load configuration
config = load_config()

# Load data
loader = DataLoader(config)
train_df, test_df = loader.load_train_test_data()

# Engineer features
engineer = FeatureEngineer(config)
train_df = engineer.create_all_features(train_df)
test_df = engineer.create_all_features(test_df)

# Prepare data
X_train, y_train = loader.get_feature_target_split(train_df)
X_test, y_test = loader.get_feature_target_split(test_df)

# Remove NaN values (from lag features)
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

# Train model
model = RandomForestModel(config)
model.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator(config)
y_pred = model.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred)

print(f"Test R² Score: {metrics['r2_score']:.4f}")
print(f"Test RMSE: {metrics['root_mean_squared_error']:.4f}")
```

## Usage

### Data Loading

```python
from src.data.loader import DataLoader

loader = DataLoader(config)

# Load train and test data
train_df, test_df = loader.load_train_test_data()

# Or load a single file
df = loader.load_data('path/to/data.csv')

# Combine datasets
combined_df = loader.combine_datasets(train_df, test_df)
```

### Feature Engineering

```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer(config)

# Create all features
df_with_features = engineer.create_all_features(df)

# Or create specific features
df = engineer.add_technical_indicators(df)
df = engineer.add_lag_features(df, lags=[1, 2, 3, 5, 10])
df = engineer.add_rolling_statistics(df)
df = engineer.add_price_features(df)
df = engineer.add_volume_features(df)
```

### Model Training

```python
from src.models.linear_models import LinearRegressionModel, SVRModel
from src.models.tree_models import RandomForestModel, GradientBoostingModel
from src.models.ensemble_models import VotingRegressorModel

# Linear models
linear_model = LinearRegressionModel(config)
svr_model = SVRModel(config)

# Tree models
rf_model = RandomForestModel(config)
gb_model = GradientBoostingModel(config)

# Ensemble models
voting_model = VotingRegressorModel(config)

# Train
model.fit(X_train, y_train)

# Save model
model.save_model()

# Load model
model.load_model('models/trained/random_forest_regression.pkl')
```

### Hyperparameter Tuning

```python
from src.models.tree_models import RandomForestModel

model = RandomForestModel(config)

# Tune hyperparameters with cross-validation
model.tune_hyperparameters(X_train, y_train, method='randomized', cv=5)

# Model is now fitted with best parameters
y_pred = model.predict(X_test)
```

### Evaluation

```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator(config)

# Evaluate single model
metrics = evaluator.evaluate(y_test, y_pred)

# Compare multiple models
comparison = evaluator.compare_models(
    models={'RF': rf_model, 'GB': gb_model},
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test
)

# Trading strategy evaluation
trading_metrics = evaluator.evaluate_trading_strategy(y_test, y_pred)
```

### Visualization

```python
from src.visualization.plots import Visualizer

viz = Visualizer(config)

# Actual vs Predicted
viz.plot_actual_vs_predicted(y_test, y_pred, save_name='actual_vs_pred.png')

# Residuals
viz.plot_residuals(y_test, y_pred, save_name='residuals.png')

# Time series
viz.plot_time_series(y_test, y_pred, save_name='timeseries.png')

# Feature importance
importance = model.get_feature_importance(X_train.columns)
viz.plot_feature_importance(importance, save_name='feature_importance.png')
```

## Models

### Linear Models
- **Linear Regression**: Simple baseline with preprocessing pipeline
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **SVR**: Support Vector Regression with RBF kernel

### Tree-Based Models
- **Decision Tree**: Simple tree with hyperparameter tuning
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting algorithm

### Ensemble Models
- **Voting Regressor**: Combines multiple models by averaging
- **Stacking Regressor**: Meta-learner combines base models
- **Bagging Regressor**: Bootstrap aggregation
- **Weighted Average**: Simple weighted combination

### Deep Learning Models (RNN/LSTM)
- **LSTM**: Long Short-Term Memory networks with multiple hidden layers
  - Captures long-term dependencies in time series
  - 3-layer architecture with configurable units [128, 64, 32]
  - Dropout regularization (0.2)
  - Early stopping and learning rate reduction
- **GRU**: Gated Recurrent Units (faster alternative to LSTM)
  - Simpler architecture with fewer parameters
  - Faster training while maintaining good performance
- **Bidirectional LSTM**: Processes sequences in both directions
  - Captures patterns from past and future
  - Especially effective for time series with bidirectional patterns
- **Hyperparameter Tuning**: Dynamic optimization with Keras Tuner
  - Bayesian Optimization for efficient search
  - Automatic architecture search (1-4 layers, 32-256 units)
  - Optimizer and learning rate tuning
  - Up to 50 trials with early stopping

## LSTM Quick Start

```python
from src.data.sequence_generator import SequenceGenerator
from src.models.deep_learning_models import LSTMModel, LSTMTuner

# Create sequences for LSTM
seq_gen = SequenceGenerator(sequence_length=60)
X_train_seq, y_train_seq, _, _, X_test_seq, y_test_seq = seq_gen.prepare_train_val_test(
    train_df, None, test_df
)

# Train LSTM
lstm_model = LSTMModel(
    config=config,
    sequence_length=60,
    n_features=X_train_seq.shape[2],
    lstm_units=[128, 64, 32],
    dropout_rate=0.2
)

lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=100)

# Predict
y_pred = lstm_model.predict(X_test_seq)
y_pred_original = seq_gen.inverse_transform_predictions(y_pred)

# Save model
lstm_model.save_model('models/trained/lstm_best.keras')
```

### Hyperparameter Tuning Example

```python
from src.models.deep_learning_models import LSTMTuner

# Initialize tuner
tuner = LSTMTuner(
    sequence_length=60,
    n_features=5,
    tuner_type='bayesian',  # 'random', 'bayesian', or 'hyperband'
    max_trials=50
)

# Search for best hyperparameters
tuner.search(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50)

# Get best model
best_model = tuner.get_best_model()
best_hp = tuner.get_best_hyperparameters()

# Evaluate
y_pred = best_model.predict(X_test_seq)
```

## Configuration

Edit `config/config.yaml` to customize:

- Data paths and columns
- Feature engineering parameters
- Model hyperparameters
- Cross-validation settings
- Evaluation metrics
- Visualization options

## Dataset

**Source**: Uniqlo Stock Price Data (2012-2017)

**Features**:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price (target variable)
- **Volume**: Trading volume
- **Stock Trading**: Total trading value

**Split**:
- Training: 2012-01-04 to 2016-12-30 (1,226 samples)
- Testing: 2017-01-06 to 2017-01-13 (8 samples)

## Results

### Model Comparison (Original Data)

| Model | Training R² | Testing R² | Testing RMSE |
|-------|-------------|------------|--------------|
| Random Forest | 99.90% | 98.20% | ~500 |
| Decision Tree | 99.86% | 98.19% | ~505 |
| Linear Regression | 99.94% | 98.10% | ~520 |
| SVR | 94.19% | 95.37% | ~810 |

**Note**: With proper feature engineering and time-series methodology, performance may vary. The above results are from the original implementation treating it as a simple regression problem.

### Key Findings

1. **Tree-based models** perform best for this task due to non-linear patterns
2. **Random Forest** provides best generalization with less overfitting risk
3. **SVR** underperforms on large datasets but provides different predictions
4. **Ensemble methods** can further improve performance by combining strengths

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: Uniqlo stock price data
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn

---

**Note**: This is an educational project. Stock price prediction is inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consult financial advisors before making investment decisions.