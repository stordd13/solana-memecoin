"""
Test fixtures for ML mathematical validation tests.
Provides test data, model instances, and reference calculations.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


@pytest.fixture
def ml_reference_data():
    """Generate reference ML data for mathematical validation."""
    np.random.seed(42)  # Reproducible results
    
    n_samples = 1000
    n_features = 10
    
    # Generate synthetic financial features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some realistic financial relationships
    # Feature 0: RSI-like (bounded 0-100)
    X[:, 0] = 50 + 30 * np.tanh(X[:, 0])
    
    # Feature 1: Returns (small centered around 0)
    X[:, 1] = X[:, 1] * 0.02
    
    # Feature 2: MACD-like (centered around 0)
    X[:, 2] = X[:, 2] * 0.1
    
    # Generate binary targets with some realistic signal
    # Price direction based on weighted combination of features
    signal = (X[:, 0] - 50) * 0.01 + X[:, 1] * 2 + X[:, 2] * 0.5 + np.random.normal(0, 0.1, n_samples)
    y_binary = (signal > 0).astype(int)
    
    # Generate regression targets (price predictions)
    y_regression = 100 + signal * 10 + np.random.normal(0, 1, n_samples)
    
    return {
        'X': X,
        'y_binary': y_binary,
        'y_regression': y_regression,
        'n_samples': n_samples,
        'n_features': n_features,
        'feature_names': [f'feature_{i}' for i in range(n_features)]
    }


@pytest.fixture 
def ml_metrics_references():
    """Generate reference metric calculations for validation."""
    # Simple test case with known values
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.7, 0.3, 0.8, 0.6])
    
    # Calculate reference metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = 0.5  # Default for edge cases
    
    return {
        'y_true': y_true,
        'y_pred': y_pred, 
        'y_proba': y_proba,
        'expected_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    }


@pytest.fixture
def financial_metrics_references():
    """Generate reference financial metrics for validation."""
    # Sample returns data
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.005, 0.01, -0.03])
    
    # Risk-free rate (annualized)
    risk_free_rate = 0.02
    
    # Expected Sharpe ratio calculation
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    expected_sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else np.std(returns, ddof=1)
    expected_sortino = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    expected_max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    expected_calmar = (np.mean(returns) * 252) / abs(expected_max_drawdown) if expected_max_drawdown != 0 else 0
    
    return {
        'returns': returns,
        'risk_free_rate': risk_free_rate,
        'expected_metrics': {
            'sharpe_ratio': expected_sharpe,
            'sortino_ratio': expected_sortino,
            'max_drawdown': expected_max_drawdown,
            'calmar_ratio': expected_calmar,
            'volatility': np.std(returns, ddof=1),
            'mean_return': np.mean(returns)
        }
    }


@pytest.fixture
def temporal_safety_data():
    """Generate data for temporal safety validation."""
    n_tokens = 5
    n_periods = 100
    
    tokens = {}
    for i in range(n_tokens):
        # Generate price data
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_periods))
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=j) for j in range(n_periods)]
        
        df = pl.DataFrame({
            'datetime': timestamps,
            'price': prices,
            'returns': np.concatenate([[np.nan], np.diff(prices) / prices[:-1]]),
            'token_id': [f'token_{i}'] * n_periods
        })
        
        tokens[f'token_{i}'] = df
    
    return tokens


@pytest.fixture
def winsorization_reference_data():
    """Generate reference data for winsorization testing."""
    np.random.seed(42)
    
    # Data with outliers
    normal_data = np.random.normal(0, 1, 95)
    outliers = np.array([10, -8, 12, -15, 20])  # Clear outliers
    data_with_outliers = np.concatenate([normal_data, outliers])
    
    # Expected winsorized values at 5% and 95% percentiles
    p5 = np.percentile(data_with_outliers, 5)
    p95 = np.percentile(data_with_outliers, 95)
    
    expected_winsorized = data_with_outliers.copy()
    expected_winsorized[expected_winsorized < p5] = p5
    expected_winsorized[expected_winsorized > p95] = p95
    
    return {
        'original_data': data_with_outliers,
        'expected_winsorized': expected_winsorized,
        'percentile_5': p5,
        'percentile_95': p95
    }


@pytest.fixture
def model_performance_benchmarks():
    """Expected performance benchmarks for different model types."""
    return {
        'directional_models': {
            'accuracy_threshold': 0.52,  # Slightly better than random (0.5)
            'precision_threshold': 0.50,
            'recall_threshold': 0.50,
            'f1_threshold': 0.50,
            'roc_auc_threshold': 0.52
        },
        'forecasting_models': {
            'r2_threshold': 0.1,      # Modest explanatory power
            'mae_threshold_pct': 0.15,  # MAE within 15% of mean price
            'rmse_threshold_pct': 0.20  # RMSE within 20% of mean price
        }
    }


def calculate_reference_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
    """Calculate reference metrics using sklearn for validation."""
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5
    
    return metrics


def calculate_reference_financial_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
    """Calculate reference financial metrics for validation."""
    returns = np.array(returns)
    
    # Basic statistics
    mean_return = np.mean(returns)
    volatility = np.std(returns, ddof=1)
    
    # Risk-adjusted metrics
    excess_returns = returns - (risk_free_rate / 252)  # Daily excess returns
    sharpe_ratio = np.mean(excess_returns) / volatility if volatility > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else volatility
    sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    annualized_return = mean_return * 252
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }