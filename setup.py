from setuptools import setup, find_packages

setup(
    name="memecoin2",
    version="0.1.0",
    description="Memecoin analysis, feature engineering and ML pipeline",
    author="Memecoin Team",
    packages=find_packages(exclude=("notebooks", "analysis_results", "data", "*.tests")),
    python_requires=">=3.9",
    install_requires=[
        "polars>=0.19",
        "numpy>=1.23",
        "scikit-learn>=1.3",
        "plotly>=5.17",
        "tqdm>=4.66"
        # Heavy deps like torch, lightgbm, xgboost are optional—install as needed.
    ],
) 