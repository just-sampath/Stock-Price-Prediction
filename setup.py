"""Setup script for Stock Price Prediction package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name="stock-price-prediction",
    version="1.0.0",
    author="Stock Price Prediction Team",
    author_email="your.email@example.com",
    description="A comprehensive ML framework for stock price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/just-sampath/Stock-Price-Prediction",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.2',
            'pytest-cov>=4.1.0',
            'black>=23.9.1',
            'flake8>=6.1.0',
            'mypy>=1.5.1',
        ],
    },
    entry_points={
        'console_scripts': [
            'stock-predict=example_usage:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
