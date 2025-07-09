"""
Utility functions for behavioral archetype analysis
Focuses on death detection and feature extraction for memecoin tokens
"""

import numpy as np
import polars as pl
from typing import Optional, Dict, List, Tuple, Union
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')


def detect_token_death(prices: np.ndarray, returns: np.ndarray, window: int = 30) -> Optional[int]:
    """
    Detect if and when a token "dies" (price becomes effectively constant).
    
    Uses multiple criteria to handle tokens with very small values correctly:
    1. Price flatness: all prices in window are identical
    2. Relative volatility: coefficient of variation of returns
    3. Tick frequency: ratio of unique prices in window
    
    Args:
        prices: Array of token prices
        returns: Array of token returns
        window: Window size for death detection (default 30 minutes)
        
    Returns:
        Index where token died, or None if token never died
    """
    if len(prices) < window or len(returns) < window:
        return None
    
    for i in range(len(returns) - window):
        window_returns = returns[i:i+window]
        window_prices = prices[i:i+window]
        
        # Skip if window contains NaN or inf
        if np.any(np.isnan(window_returns)) or np.any(np.isinf(window_returns)):
            continue
        if np.any(np.isnan(window_prices)) or np.any(np.isinf(window_prices)):
            continue
        
        # Check 1: All prices identical (true flatline)
        unique_prices = np.unique(window_prices)
        if len(unique_prices) == 1:
            return i
        
        # Check 2: Relative volatility (handles small value tokens)
        # Use median absolute deviation for robustness
        mad_returns = np.median(np.abs(window_returns - np.median(window_returns)))
        mean_abs_return = np.mean(np.abs(window_returns))
        
        if mean_abs_return > 0:
            relative_volatility = mad_returns / mean_abs_return
            if relative_volatility < 0.01:  # Less than 1% relative volatility
                # Additional check: ensure it's not just a single outlier
                if np.percentile(np.abs(window_returns), 90) < 0.001:
                    return i
        
        # Check 3: Tick frequency (for staircase death patterns)
        unique_price_ratio = len(unique_prices) / len(window_prices)
        if unique_price_ratio < 0.05:  # Less than 5% unique prices
            # Verify it's not just rounding - check actual price range
            price_range = np.max(window_prices) - np.min(window_prices)
            mean_price = np.mean(window_prices)
            if mean_price > 0:
                relative_range = price_range / mean_price
                if relative_range < 0.001:  # Less than 0.1% price range
                    return i
    
    return None


def calculate_death_features(prices: np.ndarray, returns: np.ndarray, 
                           death_minute: Optional[int]) -> Dict[str, Union[float, str, None]]:
    """
    Calculate death-related features for a token.
    
    Args:
        prices: Array of token prices
        returns: Array of token returns  
        death_minute: Index where token died (None if alive)
        
    Returns:
        Dictionary of death features
    """
    features = {
        'is_dead': death_minute is not None,
        'death_minute': death_minute,
        'lifespan_minutes': death_minute if death_minute is not None else len(prices),
        'death_type': None,
        'death_velocity': None,
        'pre_death_volatility': None,
        'death_completeness': None,
        'final_price_ratio': None
    }
    
    if death_minute is not None and death_minute > 0:
        # Death type: sudden vs gradual
        if death_minute < 5:
            features['death_type'] = 'immediate'
        elif death_minute < 60:
            features['death_type'] = 'sudden'
        elif death_minute < 360:
            features['death_type'] = 'gradual'
        else:
            features['death_type'] = 'extended'
        
        # Death velocity: how fast price declined to death
        if death_minute >= 10:
            # Price change in 10 minutes before death
            price_before = prices[max(0, death_minute - 10)]
            price_at_death = prices[death_minute]
            if price_before > 0:
                features['death_velocity'] = (price_before - price_at_death) / price_before
        
        # Pre-death volatility (30 minutes before death or available data)
        lookback = min(30, death_minute)
        if lookback > 0:
            pre_death_returns = returns[max(0, death_minute - lookback):death_minute]
            if len(pre_death_returns) > 0:
                features['pre_death_volatility'] = np.std(pre_death_returns)
        
        # Death completeness: how close to zero did price go
        max_price = np.max(prices[:death_minute]) if death_minute > 0 else prices[0]
        final_price = prices[min(death_minute + 30, len(prices) - 1)]  # 30 min after death
        if max_price > 0:
            features['death_completeness'] = 1.0 - (final_price / max_price)
            features['final_price_ratio'] = final_price / max_price
    
    elif death_minute is None:
        # For alive tokens
        max_price = np.max(prices)
        final_price = prices[-1]
        if max_price > 0:
            features['final_price_ratio'] = final_price / max_price
    
    return features


def extract_lifecycle_features(prices: np.ndarray, returns: np.ndarray,
                             death_minute: Optional[int]) -> Dict[str, float]:
    """
    Extract lifecycle features for a token (pre-death if applicable).
    
    Args:
        prices: Array of token prices
        returns: Array of token returns
        death_minute: Index where token died (None if alive)
        
    Returns:
        Dictionary of lifecycle features
    """
    # Determine analysis period (before death or full series)
    if death_minute is not None and death_minute > 0:
        analysis_prices = prices[:death_minute]
        analysis_returns = returns[:death_minute]
    else:
        analysis_prices = prices
        analysis_returns = returns
    
    features = {}
    
    # Basic statistics
    features['mean_return'] = np.mean(analysis_returns) if len(analysis_returns) > 0 else 0
    features['std_return'] = np.std(analysis_returns) if len(analysis_returns) > 0 else 0
    features['skew_return'] = calculate_skewness(analysis_returns)
    features['kurtosis_return'] = calculate_kurtosis(analysis_returns)
    
    # Price trajectory features
    if len(analysis_prices) > 0:
        features['max_price'] = np.max(analysis_prices)
        features['min_price'] = np.min(analysis_prices)
        features['price_range'] = features['max_price'] - features['min_price']
        
        # Peak timing (when highest price occurred)
        peak_idx = np.argmax(analysis_prices)
        features['peak_timing_ratio'] = peak_idx / len(analysis_prices)
        features['peak_timing_minutes'] = peak_idx
        
        # Drawdown metrics
        drawdown_info = calculate_max_drawdown(analysis_prices)
        features.update(drawdown_info)
        
        # Trend features
        if len(analysis_prices) > 1:
            # Linear regression slope
            x = np.arange(len(analysis_prices))
            slope, _ = np.polyfit(x, analysis_prices, 1)
            features['price_trend'] = slope
            
            # Normalized trend (relative to mean price)
            mean_price = np.mean(analysis_prices)
            if mean_price > 0:
                features['normalized_trend'] = slope / mean_price
    
    # ACF features at multiple lags
    if len(analysis_returns) > 60:
        acf_lags = [1, 2, 5, 10, 20, 60]
        acf_values = acf(analysis_returns, nlags=60, fft=True)
        for lag in acf_lags:
            if lag < len(acf_values):
                features[f'acf_lag_{lag}'] = acf_values[lag]
    
    return features


def calculate_skewness(returns: np.ndarray) -> float:
    """Calculate skewness with handling for edge cases."""
    if len(returns) < 3:
        return 0.0
    
    cleaned = returns[~np.isnan(returns) & ~np.isinf(returns)]
    if len(cleaned) < 3:
        return 0.0
    
    mean = np.mean(cleaned)
    std = np.std(cleaned)
    if std == 0:
        return 0.0
    
    return np.mean(((cleaned - mean) / std) ** 3)


def calculate_kurtosis(returns: np.ndarray) -> float:
    """Calculate kurtosis with handling for edge cases."""
    if len(returns) < 4:
        return 0.0
    
    cleaned = returns[~np.isnan(returns) & ~np.isinf(returns)]
    if len(cleaned) < 4:
        return 0.0
    
    mean = np.mean(cleaned)
    std = np.std(cleaned)
    if std == 0:
        return 0.0
    
    return np.mean(((cleaned - mean) / std) ** 4) - 3.0


def calculate_max_drawdown(prices: np.ndarray) -> Dict[str, float]:
    """Calculate maximum drawdown and related metrics."""
    if len(prices) < 2:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'drawdown_recovery_ratio': 0.0
        }
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown at each point
    drawdown = (running_max - prices) / running_max
    
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    
    # Find drawdown duration (peak to trough)
    if max_dd_idx > 0:
        peak_idx = np.where(prices[:max_dd_idx] == running_max[max_dd_idx])[0]
        if len(peak_idx) > 0:
            duration = max_dd_idx - peak_idx[-1]
        else:
            duration = 0
    else:
        duration = 0
    
    # Check if recovered
    recovery_ratio = 0.0
    if max_dd > 0 and max_dd_idx < len(prices) - 1:
        trough_price = prices[max_dd_idx]
        final_price = prices[-1]
        peak_price = running_max[max_dd_idx]
        if peak_price > trough_price:
            recovery_ratio = (final_price - trough_price) / (peak_price - trough_price)
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_duration': duration,
        'drawdown_recovery_ratio': np.clip(recovery_ratio, 0, 1)
    }


def extract_early_features(prices: np.ndarray, returns: np.ndarray, 
                         window_minutes: int = 5) -> Dict[str, float]:
    """
    Extract features using only the first N minutes of data for early detection.
    
    Args:
        prices: Array of token prices
        returns: Array of token returns
        window_minutes: Number of minutes to use (default 5)
        
    Returns:
        Dictionary of early detection features
    """
    features = {}
    
    # Ensure we have enough data
    max_idx = min(window_minutes, len(prices), len(returns))
    if max_idx < 2:
        return features
    
    early_prices = prices[:max_idx]
    early_returns = returns[:max_idx]
    
    # Return magnitude
    if len(early_returns) > 0:
        features['return_magnitude_5min'] = np.max(np.abs(early_returns))
        features['max_return_5min'] = np.max(early_returns)
        features['min_return_5min'] = np.min(early_returns)
    
    # Volatility
    if len(early_returns) > 1:
        features['volatility_5min'] = np.std(early_returns)
        features['volatility_normalized_5min'] = features['volatility_5min'] / np.mean(np.abs(early_returns)) if np.mean(np.abs(early_returns)) > 0 else 0
    
    # Trend direction
    if len(early_prices) > 1:
        x = np.arange(len(early_prices))
        slope, intercept = np.polyfit(x, early_prices, 1)
        features['trend_direction_5min'] = slope
        
        # Normalized trend
        mean_price = np.mean(early_prices)
        if mean_price > 0:
            features['trend_normalized_5min'] = slope / mean_price
    
    # Autocorrelation at lag 1
    if len(early_returns) > 2:
        acf_values = acf(early_returns, nlags=1, fft=False)
        features['autocorrelation_5min'] = acf_values[1] if len(acf_values) > 1 else 0
    
    # Price movement characteristics
    if len(early_prices) > 0:
        features['price_change_ratio_5min'] = (early_prices[-1] - early_prices[0]) / early_prices[0] if early_prices[0] > 0 else 0
        features['price_range_5min'] = (np.max(early_prices) - np.min(early_prices)) / early_prices[0] if early_prices[0] > 0 else 0
    
    # Tick frequency (unique price ratio)
    features['unique_price_ratio_5min'] = len(np.unique(early_prices)) / len(early_prices)
    
    return features


def prepare_token_data(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Prepare token data for analysis, including death detection.
    
    Args:
        df: Polars DataFrame with 'datetime' and 'price' columns
        
    Returns:
        Tuple of (prices, returns, death_minute)
    """
    # Sort by datetime
    df = df.sort('datetime')
    
    # Extract prices
    prices = df['price'].to_numpy()
    
    # Calculate returns with special handling for extreme values
    returns = np.zeros(len(prices) - 1)
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            # Use log returns for extreme movements (>100%)
            if abs(ret) > 1.0:
                returns[i-1] = np.log(prices[i] / prices[i-1])
            else:
                returns[i-1] = ret
        else:
            returns[i-1] = 0
    
    # Detect death
    death_minute = detect_token_death(prices, returns)
    
    return prices, returns, death_minute


def categorize_by_lifespan(features_df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Categorize tokens by their active lifespan (CEO requirement).
    
    Args:
        features_df: DataFrame with token features including 'lifespan_minutes'
        
    Returns:
        DataFrame with added 'lifespan_category' column
    """
    import pandas as pd
    
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    # Define lifespan categories (CEO's roadmap)
    def get_lifespan_category(lifespan_minutes):
        if lifespan_minutes <= 400:
            return 'Sprint'
        elif lifespan_minutes <= 1200:
            return 'Standard'
        else:
            return 'Marathon'
    
    # Apply categorization
    df['lifespan_category'] = df['lifespan_minutes'].apply(get_lifespan_category)
    
    return df


def categorize_by_lifespan(token_data: Dict, token_limits: Dict) -> Dict:
    """
    Categorize tokens by lifespan and apply limits.
    
    Args:
        token_data: Dictionary of token_name -> polars DataFrame
        token_limits: Dictionary with 'sprint', 'standard', 'marathon' limits
        
    Returns:
        Dictionary with categorized tokens
    """
    import random
    
    categorized = {
        'Sprint': {},
        'Standard': {},
        'Marathon': {}
    }
    
    # Categorize each token
    for token_name, token_df in token_data.items():
        if token_df.is_empty():
            continue
        
        # Get lifespan
        lifespan_minutes = len(token_df)
        
        # Determine category
        if lifespan_minutes <= 400:
            category = 'Sprint'
        elif lifespan_minutes <= 1200:
            category = 'Standard'
        else:
            category = 'Marathon'
        
        categorized[category][token_name] = token_df
    
    # Apply limits (only if limit is not None)
    for category, limit in [('Sprint', token_limits['sprint']), 
                           ('Standard', token_limits['standard']), 
                           ('Marathon', token_limits['marathon'])]:
        if limit is not None and len(categorized[category]) > limit:
            # Randomly sample
            tokens_to_keep = random.sample(list(categorized[category].keys()), limit)
            categorized[category] = {token: categorized[category][token] for token in tokens_to_keep}
    
    return categorized