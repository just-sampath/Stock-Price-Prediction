"""Configuration loader utility."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is not valid YAML.
    """
    if config_path is None:
        # Default to config/config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file: {e}")


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path object pointing to project root.
    """
    return Path(__file__).parent.parent.parent


def get_data_path(relative_path: str) -> Path:
    """
    Get absolute path for data file.

    Args:
        relative_path: Relative path from project root.

    Returns:
        Absolute path to data file.
    """
    return get_project_root() / relative_path
