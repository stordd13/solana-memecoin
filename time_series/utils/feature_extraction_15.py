# feature_extraction_15.py
"""
Exact 15 features as specified by CEO roadmap for memecoin behavioral archetype analysis.
Extracted from raw data according to A/B test requirements (raw vs safe-log returns).
"""

import numpy as np
import polars as pl
from typing import Dict, Optional, Union
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

from .death_detection import detect_token_death, calculate_death_features, safe_divide


# EXACT 15 Features as specified by CEO
ESSENTIAL_FEATURES = {
    # Death characteristics (3)
    'is_dead': bool,                    # Token died or not
    'death_minute': int,                 # When it died (NaN if alive)
    'lifespan_minutes': int,             # Total active minutes
    
    # Core statistics (4) - Computed from raw or log returns
    'mean_return': float,                # Avg return (extreme OK)
    'std_return': float,                 # Volatility
    'max_drawdown': float,               # Peak-to-trough
    'volatility_5min': float,            # First 5 mins vol
    
    # Autocorrelation signature (3)
    'acf_lag_1': float,                  # Momentum
    'acf_lag_5': float,                  # 5-min pattern
    'acf_lag_10': float,                 # 10-min pattern
    
    # Early detection (first 5 mins) (5)
    'return_5min': float,                # Total 5-min return
    'max_return_1min': float,            # Biggest 1-min jump
    'trend_direction_5min': float,       # Slope
    'price_change_ratio_5min': float,    # (max-min)/initial
    'pump_velocity_5min': float          # Speed of increase
}


def safe_acf_calculation(returns: np.ndarray, max_lags: int = 10) -> np.ndarray:
    """
    Calculate autocorrelation function with robust error handling.
    
    Args:
        returns: Array of returns
        max_lags: Maximum number of lags to calculate
        
    Returns:
        Array of ACF values, padded with zeros if calculation fails
    """
    try:
        # Remove NaN/inf values
        clean_returns = returns[np.isfinite(returns)]
        
        if len(clean_returns) < max_lags + 5:  # Need minimum data
            return np.zeros(max_lags + 1)
        
        # Calculate ACF
        acf_values = acf(clean_returns, nlags=max_lags, fft=True)
        
        # Handle any remaining NaN values
        acf_values = np.nan_to_num(acf_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        return acf_values
        
    except Exception:
        # Return zeros if ACF calculation fails
        return np.zeros(max_lags + 1)


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown from peak to trough.
    
    Args:
        prices: Array of token prices
        
    Returns:
        Maximum drawdown as a ratio (0 to 1)
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown at each point
    drawdown = 1.0 - (prices / running_max)
    
    # Return maximum drawdown
    max_dd = np.max(drawdown)
    
    return float(max_dd) if np.isfinite(max_dd) else 0.0


def calculate_trend_slope(prices: np.ndarray) -> float:
    """
    Calculate trend direction using linear regression slope.
    
    Args:
        prices: Array of prices
        
    Returns:
        Slope coefficient (trend direction)
    """
    if len(prices) < 2:
        return 0.0
    
    x = np.arange(len(prices))
    
    # Use numpy polyfit for simple linear regression
    try:
        slope, _ = np.polyfit(x, prices, 1)
        return float(slope) if np.isfinite(slope) else 0.0
    except:
        return 0.0


def extract_features_from_returns(returns: np.ndarray, prices: np.ndarray, 
                                use_log: bool = False) -> Dict[str, float]:
    """
    Extract the exact 15 features from token data.
    
    Args:
        returns: Array of returns (will be converted based on use_log)
        prices: Array of token prices
        use_log: If True, use safe log returns; if False, use raw returns
        
    Returns:
        Dictionary with exactly 15 features as specified by CEO
    """
    # Convert returns based on A/B test parameter
    if use_log:
        # Safe log returns: log(price[t] / price[t-1])
        if len(prices) > 1:
            returns = np.log(prices[1:] / (prices[:-1] + 1e-12))
        else:
            returns = np.array([0.0])
    else:
        # Raw returns: (price[t] - price[t-1]) / price[t-1]
        if len(prices) > 1:
            returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-12)
        else:
            returns = np.array([0.0])
    
    # Clean returns for calculations
    clean_returns = returns[np.isfinite(returns)]
    
    # 1. Death detection
    death_minute = detect_token_death(prices, returns)
    death_features = calculate_death_features(prices, returns, death_minute)
    
    # 2. Core statistics (use only pre-death data if applicable)
    analysis_length = death_minute if death_minute is not None else len(returns)
    analysis_returns = returns[:analysis_length] if analysis_length > 0 else returns
    analysis_prices = prices[:analysis_length+1] if analysis_length > 0 else prices  # +1 for prices
    
    mean_return = np.mean(analysis_returns) if len(analysis_returns) > 0 else 0.0
    std_return = np.std(analysis_returns) if len(analysis_returns) > 0 else 0.0
    max_drawdown = calculate_max_drawdown(analysis_prices)
    
    # Volatility in first 5 minutes
    volatility_5min = np.std(analysis_returns[:5]) if len(analysis_returns) >= 5 else std_return
    
    # 3. Autocorrelation signature
    acf_values = safe_acf_calculation(analysis_returns, max_lags=10)
    acf_lag_1 = acf_values[1] if len(acf_values) > 1 else 0.0
    acf_lag_5 = acf_values[5] if len(acf_values) > 5 else 0.0
    acf_lag_10 = acf_values[10] if len(acf_values) > 10 else 0.0
    
    # 4. Early detection (first 5 minutes)
    early_window = min(5, len(analysis_returns))
    
    if early_window > 0:
        early_returns = analysis_returns[:early_window]
        early_prices = analysis_prices[:early_window+1]
        
        # Total 5-minute return
        if len(early_prices) >= 2:
            return_5min = (early_prices[-1] - early_prices[0]) / (early_prices[0] + 1e-12)
        else:
            return_5min = 0.0
        
        # Maximum 1-minute return
        max_return_1min = np.max(np.abs(early_returns)) if len(early_returns) > 0 else 0.0
        
        # Trend direction (slope)
        trend_direction_5min = calculate_trend_slope(early_prices)
        
        # Price change ratio
        if len(early_prices) >= 2 and early_prices[0] > 0:
            price_range = np.max(early_prices) - np.min(early_prices)
            price_change_ratio_5min = price_range / early_prices[0]
        else:
            price_change_ratio_5min = 0.0
        
        # Pump velocity (speed of increase)
        positive_returns = early_returns[early_returns > 0]
        pump_velocity_5min = np.mean(positive_returns) if len(positive_returns) > 0 else 0.0
    else:
        return_5min = 0.0
        max_return_1min = 0.0
        trend_direction_5min = 0.0
        price_change_ratio_5min = 0.0
        pump_velocity_5min = 0.0
    
    # Compile exact 15 features
    features = {
        # Death characteristics (3)
        'is_dead': death_features['is_dead'],
        'death_minute': death_features['death_minute'] if death_features['death_minute'] is not None else -1,
        'lifespan_minutes': death_features['lifespan_minutes'],
        
        # Core statistics (4)
        'mean_return': float(mean_return) if np.isfinite(mean_return) else 0.0,
        'std_return': float(std_return) if np.isfinite(std_return) else 0.0,
        'max_drawdown': float(max_drawdown) if np.isfinite(max_drawdown) else 0.0,
        'volatility_5min': float(volatility_5min) if np.isfinite(volatility_5min) else 0.0,
        
        # Autocorrelation signature (3)
        'acf_lag_1': float(acf_lag_1) if np.isfinite(acf_lag_1) else 0.0,
        'acf_lag_5': float(acf_lag_5) if np.isfinite(acf_lag_5) else 0.0,
        'acf_lag_10': float(acf_lag_10) if np.isfinite(acf_lag_10) else 0.0,
        
        # Early detection (5)
        'return_5min': float(return_5min) if np.isfinite(return_5min) else 0.0,
        'max_return_1min': float(max_return_1min) if np.isfinite(max_return_1min) else 0.0,
        'trend_direction_5min': float(trend_direction_5min) if np.isfinite(trend_direction_5min) else 0.0,
        'price_change_ratio_5min': float(price_change_ratio_5min) if np.isfinite(price_change_ratio_5min) else 0.0,
        'pump_velocity_5min': float(pump_velocity_5min) if np.isfinite(pump_velocity_5min) else 0.0
    }
    
    return features


def extract_features_from_token_data(token_df: pl.DataFrame, use_log: bool = False) -> Dict[str, float]:
    """
    Extract features from a token DataFrame.
    
    Args:
        token_df: DataFrame with 'price' and 'datetime' columns
        use_log: Whether to use log returns (True) or raw returns (False)
        
    Returns:
        Dictionary with 15 features
    """
    if 'price' not in token_df.columns:
        raise ValueError("Token DataFrame must contain 'price' column")
    
    prices = token_df['price'].to_numpy()
    
    # Calculate returns (will be recalculated inside extract_features_from_returns based on use_log)
    if len(prices) > 1:
        returns = np.diff(prices) / (prices[:-1] + 1e-12)
    else:
        returns = np.array([0.0])
    
    return extract_features_from_returns(returns, prices, use_log=use_log)