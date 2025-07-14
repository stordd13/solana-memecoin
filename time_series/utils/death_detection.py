# death_detection.py
"""
Centralized death detection logic for memecoin tokens.
Extracted from archetype_utils.py for use across all phases.
"""

import numpy as np
import polars as pl
from typing import Optional, Dict, Union


def safe_divide(a, b, default=0.0):
    """Safe division with default value for division by zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.full_like(a, default, dtype=float)
    mask = np.abs(b) > 1e-10
    np.divide(a, b, where=mask, out=result)
    return result


def detect_token_death(prices: np.ndarray, returns: np.ndarray, min_death_duration: int = 30) -> Optional[int]:
    """
    Detect if and when a token "dies" by finding the last point of meaningful activity.
    
    Works backwards from the end to find when meaningful trading stopped, then verifies
    that prices are constant from that point to the end for at least min_death_duration.
    
    Args:
        prices: Array of token prices
        returns: Array of token returns
        min_death_duration: Minimum duration of constant prices to consider death (default 30 minutes)
        
    Returns:
        Index where token died (start of death period), or None if token never died
    """
    # Pre-fill gaps with forward-fill, then backward for edges
    prices_series = pl.Series(prices)
    prices = prices_series.forward_fill().backward_fill().to_numpy()
    
    if len(prices) < min_death_duration or len(returns) < min_death_duration:
        return None
    
    # Define what constitutes "meaningful activity"
    def is_meaningful_activity(price_window, return_window):
        """Check if a window contains meaningful trading activity."""
        if len(price_window) == 0 or len(return_window) == 0:
            return False
        
        # Remove NaN values
        valid_prices = price_window[~np.isnan(price_window)]
        valid_returns = return_window[~np.isnan(return_window)]
        
        if len(valid_prices) == 0 or len(valid_returns) == 0:
            return False
        
        # Check 1: Price variety - more than one unique price
        unique_prices = np.unique(valid_prices)
        if len(unique_prices) <= 1:
            return False
        
        # Check 2: Meaningful price movement (relative range > 0.1%)
        price_range = np.max(valid_prices) - np.min(valid_prices)
        mean_price = np.mean(valid_prices)
        relative_range = safe_divide(price_range, mean_price)
        if relative_range < 0.001:  # Less than 0.1% price movement
            return False
        
        # Check 3: Some volatility in returns
        if np.std(valid_returns) < 1e-6:  # Essentially zero volatility
            return False
        
        return True
    
    # Work backwards from the end to find last meaningful activity
    total_length = len(prices)
    window_size = 5  # Check activity in 5-minute windows
    
    last_activity_minute = None
    
    # Start from the end and work backwards
    for end_idx in range(total_length, window_size - 1, -1):
        start_idx = end_idx - window_size
        
        window_prices = prices[start_idx:end_idx]
        window_returns = returns[start_idx:end_idx] if start_idx < len(returns) else returns[max(0, start_idx-1):end_idx-1]
        
        if is_meaningful_activity(window_prices, window_returns):
            last_activity_minute = end_idx
            break
    
    if last_activity_minute is None:
        # No meaningful activity found in entire token lifecycle
        return 0  # Consider dead from the start
    
    # Calculate the duration from last activity to end
    death_duration = total_length - last_activity_minute
    
    # Token is dead if the period from last activity to end is >= min_death_duration
    if death_duration >= min_death_duration:
        return last_activity_minute
    else:
        return None  # Token is alive (has recent activity or death period too short)


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
                features['death_velocity'] = safe_divide(price_before - price_at_death, price_before)
        
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


def categorize_by_lifespan(lifespan_minutes: int) -> str:
    """
    Categorize token by lifespan following CEO requirements.
    
    Args:
        lifespan_minutes: Token lifespan in minutes
        
    Returns:
        Category string: 'sprint', 'standard', or 'marathon'
    """
    if lifespan_minutes < 400:
        return 'sprint'
    elif lifespan_minutes < 1200:
        return 'standard'
    else:
        return 'marathon'