"""Unit tests for data loader module."""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.config_loader import load_config
from src.data.loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = load_config()
        cls.loader = DataLoader(cls.config)

    def test_load_config(self):
        """Test configuration loading."""
        self.assertIsNotNone(self.config)
        self.assertIn('data', self.config)

    def test_loader_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.date_column, 'Date')
        self.assertEqual(self.loader.target_column, 'Close')

    def test_load_train_test_data(self):
        """Test loading train and test data."""
        try:
            train_df, test_df = self.loader.load_train_test_data()

            # Check that data is loaded
            self.assertIsInstance(train_df, pd.DataFrame)
            self.assertIsInstance(test_df, pd.DataFrame)

            # Check that data is not empty
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(test_df), 0)

            # Check that required columns exist
            self.assertIn('Close', train_df.columns)
            self.assertIn('Close', test_df.columns)

            # Check index is DatetimeIndex
            self.assertIsInstance(train_df.index, pd.DatetimeIndex)
            self.assertIsInstance(test_df.index, pd.DatetimeIndex)

        except FileNotFoundError:
            self.skipTest("Data files not found")

    def test_feature_target_split(self):
        """Test splitting features and target."""
        try:
            train_df, test_df = self.loader.load_train_test_data()

            X_train, y_train = self.loader.get_feature_target_split(train_df)

            # Check shapes
            self.assertEqual(len(X_train), len(y_train))

            # Check that target is not in features
            self.assertNotIn('Close', X_train.columns)

            # Check that target is a Series
            self.assertIsInstance(y_train, pd.Series)

        except FileNotFoundError:
            self.skipTest("Data files not found")


if __name__ == '__main__':
    unittest.main()
