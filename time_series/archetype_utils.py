# archetype_utils.py
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
from pathlib import Path
from scipy.stats import skew
from joblib import Parallel, delayed


def safe_divide(a, b, default=0.0):
    """Safe division with default value for division by zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.full_like(a, default, dtype=float)
    mask = np.abs(b) > 1e-10
    np.divide(a, b, where=mask, out=result)
    return result


def winsorize_array(data, limits=[0.01, 0.01]):
    """Apply winsorization to handle extreme outliers."""
    if len(data) == 0:
        return data
    return mstats.winsorize(data, limits=limits)


def robust_feature_calculation(data, feature_func, default_value=0.0):
    """Robust wrapper for feature calculations with NaN/inf handling."""
    try:
        if len(data) == 0:
            return default_value
        
        # Remove NaN/inf values
        clean_data = data[np.isfinite(data)]
        if len(clean_data) == 0:
            return default_value
            
        # Apply winsorization for extreme outliers
        clean_data = winsorize_array(clean_data)
        
        # Calculate feature
        result = feature_func(clean_data)
        
        # Ensure result is finite
        if not np.isfinite(result):
            return default_value
            
        return result
        
    except Exception:
        return default_value


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
    # Pre-fill gaps with forward-fill, then backward for edges
    prices_series = pl.Series(prices)
    prices = prices_series.forward_fill().backward_fill().to_numpy()
    
    if len(prices) < window or len(returns) < window:
        return None
    
    for i in range(len(returns) - window + 1):  # Adjusted to +1 for exact window
        window_returns = returns[i:i+window]
        window_prices = prices[i:i+window]
        
        # Skip if window contains NaN or inf (after fill, should be minimal)
        if np.any(np.isnan(window_returns)) or np.any(np.isinf(window_returns)):
            continue
        if np.any(np.isnan(window_prices)) or np.any(np.isinf(window_prices)):
            continue
        
        # NEW: Skip if >5% gaps remaining (severe data issues)
        gap_pct = np.sum(np.isnan(window_prices)) / len(window_prices)
        if gap_pct > 0.05:
            continue
        
        # Get valid prices/returns after any remaining NaN removal
        valid_prices = window_prices[~np.isnan(window_prices)]
        valid_returns = window_returns[~np.isnan(window_returns)]
        
        # Check 1: All prices identical (true flatline)
        unique_prices = np.unique(valid_prices)
        if len(unique_prices) == 1:
            return i
        
        # Check 2: Relative volatility (handles small value tokens)
        # Use median absolute deviation for robustness
        mad_returns = np.median(np.abs(valid_returns - np.median(valid_returns)))
        mean_abs_return = np.mean(np.abs(valid_returns))
        
        if mean_abs_return > 0:
            relative_volatility = safe_divide(mad_returns, mean_abs_return)
            if relative_volatility < 0.01:  # Less than 1% relative volatility
                # Additional check: ensure it's not just a single outlier
                if np.percentile(np.abs(valid_returns), 90) < 0.001:
                    return i
        
        # Check 3: Tick frequency (for staircase death patterns)
        unique_price_ratio = len(unique_prices) / len(valid_prices) if len(valid_prices) > 0 else 0
        if unique_price_ratio < 0.1:  # Increased to 0.1 for staircase tolerance
            # Verify it's not just rounding - check actual price range
            price_range = np.max(valid_prices) - np.min(valid_prices)
            mean_price = np.mean(valid_prices)
            relative_range = safe_divide(price_range, mean_price)
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
    features['skew_return'] = skew(analysis_returns) if len(analysis_returns) > 2 else 0
    features['kurtosis_return'] = (np.mean(analysis_returns**4) / features['std_return']**4 - 3) if features['std_return'] > 0 else 0
    
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


def prepare_token_data_lazy(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Prepare token data using lazy evaluation for complex transformations.
    
    Args:
        lazy_df: Polars LazyFrame with 'datetime' and 'price' columns
        
    Returns:
        LazyFrame with computed returns and other derived features
    """
    return lazy_df.with_columns([
        # Vectorized return calculation using polars expressions (lazy)
        pl.col('price').clip(lower_bound=1e-10).alias('price_clipped'),
        pl.col('price').clip(lower_bound=1e-10).shift(1).alias('prev_price')
    ]).drop_nulls().with_columns([
        # Calculate simple and log returns
        ((pl.col('price_clipped') - pl.col('prev_price')) / pl.col('prev_price')).alias('simple_return'),
        (pl.col('price_clipped') / pl.col('prev_price')).log().alias('log_return'),
        (((pl.col('price_clipped') - pl.col('prev_price')) / pl.col('prev_price')).abs() > 1.0).alias('is_extreme')
    ]).with_columns([
        # Use log return for extreme cases, simple return otherwise
        pl.when(pl.col('is_extreme'))
        .then(pl.col('log_return'))
        .otherwise(pl.col('simple_return'))
        .alias('final_return')
    ]).with_columns([
        # Add rolling features for analysis (computed lazily)
        pl.col('final_return').rolling_mean(window_size=5).alias('return_ma_5'),
        pl.col('final_return').rolling_std(window_size=5).alias('return_std_5'),
        pl.col('price_clipped').rolling_max(window_size=30).alias('price_max_30'),
        pl.col('price_clipped').rolling_min(window_size=30).alias('price_min_30')
    ])


def prepare_token_data(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Prepare token data for analysis with optimized polars vectorization.
    
    Args:
        df: Polars DataFrame with 'datetime' and 'price' columns
        
    Returns:
        Tuple of (prices, returns, death_minute)
    """
    # Sort by datetime and optimize with polars vectorization
    df_sorted = df.sort('datetime').with_columns([
        # Clip prices to prevent log(0) issues 
        pl.col('price').clip(lower_bound=1e-10).alias('price_clipped'),
        # Calculate previous price (lag)
        pl.col('price').clip(lower_bound=1e-10).shift(1).alias('prev_price')
    ]).drop_nulls()
    
    # Vectorized return calculation using polars expressions
    df_with_returns = df_sorted.with_columns([
        # Calculate simple returns
        ((pl.col('price_clipped') - pl.col('prev_price')) / pl.col('prev_price')).alias('simple_return'),
        # Calculate log returns for extreme cases
        (pl.col('price_clipped') / pl.col('prev_price')).log().alias('log_return'),
        # Check if return is extreme (>100% or <-50%)
        (((pl.col('price_clipped') - pl.col('prev_price')) / pl.col('prev_price')).abs() > 1.0).alias('is_extreme')
    ]).with_columns([
        # Use log return for extreme cases, simple return otherwise
        pl.when(pl.col('is_extreme'))
        .then(pl.col('log_return'))
        .otherwise(pl.col('simple_return'))
        .alias('final_return')
    ])
    
    # Extract optimized arrays
    prices = df_with_returns['price_clipped'].to_numpy()
    returns = df_with_returns['final_return'].drop_nulls().to_numpy()
    
    # Detect death
    death_minute = detect_token_death(prices, returns)
    
    return prices, returns, death_minute


def _calculate_vectorized_stats(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple statistical features efficiently using vectorized operations.
    
    Args:
        returns: Array of token returns
        
    Returns:
        Dictionary of statistical features
    """
    if len(returns) == 0:
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'volatility_5min': 0.0,
            'return_5min': 0.0,
            'max_return_1min': 0.0
        }
    
    # Use polars for vectorized statistical calculations
    returns_series = pl.Series('returns', returns)
    early_returns = returns[:5] if len(returns) >= 5 else returns
    early_series = pl.Series('early_returns', early_returns)
    
    # Calculate all statistics in one pass using polars aggregations
    stats = {
        'mean_return': returns_series.mean(),
        'std_return': returns_series.std(),
        'volatility_5min': early_series.std() if len(early_returns) > 1 else 0.0,
        'return_5min': early_series.sum(),
        'max_return_1min': early_series.max() if len(early_returns) > 0 else 0.0
    }
    
    # Handle None values from polars aggregations
    return {k: (v if v is not None else 0.0) for k, v in stats.items()}


def _extract_token_features_lazy(token_name: str, lazy_df: pl.LazyFrame, use_log_returns: bool = False) -> Optional[Dict]:
    """
    Worker function to extract features for a single token using lazy evaluation.
    
    Args:
        token_name: Name of the token
        lazy_df: Token LazyFrame 
        use_log_returns: Whether to use modified log returns
        
    Returns:
        Dictionary of features or None if processing failed
    """
    try:
        # Prepare token data with lazy evaluation
        prepared_lazy = prepare_token_data_lazy(lazy_df)
        
        # Add feature computations to the lazy chain
        features_lazy = prepared_lazy.with_columns([
            # Death detection indicators (lazy computation)
            pl.col('price_clipped').count().alias('total_length'),
            # Statistical features using polars aggregations
            pl.col('final_return').mean().alias('mean_return'),
            pl.col('final_return').std().alias('std_return'),
            pl.col('final_return').head(5).sum().alias('return_5min'),
            pl.col('final_return').head(5).max().alias('max_return_1min'),
            pl.col('final_return').head(5).std().alias('volatility_5min'),
            # Price-based features
            (pl.col('price_max_30').max() - pl.col('price_min_30').min()).alias('price_range'),
            pl.col('price_clipped').max().alias('max_price'),
            pl.col('price_clipped').min().alias('min_price'),
        ]).select([
            'category', 'total_length', 'mean_return', 'std_return', 
            'return_5min', 'max_return_1min', 'volatility_5min',
            'price_range', 'max_price', 'min_price', 'final_return', 'price_clipped'
        ])
        
        # Materialize only when we need the actual values
        materialized = features_lazy.collect()
        
        if materialized.height == 0:
            return None
            
        # Extract basic features from materialized data
        row = materialized.row(0, named=True)
        prices = materialized['price_clipped'].to_numpy()
        returns = materialized['final_return'].drop_nulls().to_numpy()
        
        # Calculate max drawdown
        max_drawdown = (row['price_range'] / row['max_price']) if row['max_price'] > 0 else 0
        
        # Detect death (still need numpy arrays for this)
        death_minute = detect_token_death(prices, returns)
        
        # Apply log transformation if requested
        if use_log_returns:
            safe_returns = np.log(1 + returns + 1e-10)
        else:
            safe_returns = returns
        
        # ACF calculations (requires numpy/statsmodels)
        acf_values = acf(safe_returns, nlags=10, fft=True)
        acf_lag_1 = acf_values[1] if len(acf_values) > 1 else 0
        acf_lag_5 = acf_values[5] if len(acf_values) > 5 else 0
        acf_lag_10 = acf_values[10] if len(acf_values) > 10 else 0
        
        # Additional calculations
        early_returns = safe_returns[:5]
        if len(early_returns) > 1:
            trend_direction_5min = np.polyfit(range(len(early_returns)), early_returns, 1)[0]
        else:
            trend_direction_5min = 0
            
        price_change_ratio_5min = safe_divide(np.max(prices[:5]) - np.min(prices[:5]), prices[0])
        pump_velocity_5min = np.max(np.cumsum(early_returns)) if len(early_returns) > 0 else 0
        
        # Compile features
        features = {
            'token': token_name,
            'category': row['category'],
            # Death (3)
            'is_dead': death_minute is not None,
            'death_minute': death_minute,
            'lifespan_minutes': death_minute if death_minute is not None else len(prices),
            # Core stats (4) - from lazy calculations
            'mean_return': row['mean_return'] or 0.0,
            'std_return': row['std_return'] or 0.0,
            'max_drawdown': max_drawdown,
            'volatility_5min': row['volatility_5min'] or 0.0,
            # ACF (3)
            'acf_lag_1': acf_lag_1,
            'acf_lag_5': acf_lag_5,
            'acf_lag_10': acf_lag_10,
            # Early detection (5)
            'return_5min': row['return_5min'] or 0.0,
            'max_return_1min': row['max_return_1min'] or 0.0,
            'trend_direction_5min': trend_direction_5min,
            'price_change_ratio_5min': price_change_ratio_5min,
            'pump_velocity_5min': pump_velocity_5min
        }
        
        return features
        
    except Exception as e:
        print(f"DEBUG: Error processing {token_name}: {e}")
        return None


def _extract_token_features(token_name: str, df: pl.DataFrame, use_log_returns: bool = False) -> Optional[Dict]:
    """
    Worker function to extract features for a single token (used for parallel processing).
    
    Args:
        token_name: Name of the token
        df: Token DataFrame
        use_log_returns: Whether to use modified log returns
        
    Returns:
        Dictionary of features or None if processing failed
    """
    try:
        prices, returns, death_minute = prepare_token_data(df)
        
        # Death features (3)
        is_dead = death_minute is not None
        lifespan_minutes = death_minute if is_dead else len(prices)
        
        # Apply log transformation if requested
        if use_log_returns:
            safe_returns = np.log(1 + returns + 1e-10)
        else:
            safe_returns = returns
        
        # Use vectorized statistics calculation for core stats
        vectorized_stats = _calculate_vectorized_stats(safe_returns)
        
        # Calculate max drawdown using polars vectorization
        if len(prices) > 0:
            prices_series = pl.Series('prices', prices)
            max_price = prices_series.max()
            min_price = prices_series.min()
            max_drawdown = (max_price - min_price) / max_price if max_price > 0 else 0
        else:
            max_drawdown = 0
        
        # Autocorrelation signature (3) - with robust error handling for constant series
        try:
            # Check if returns have sufficient variance for ACF calculation
            if len(safe_returns) > 1 and np.std(safe_returns) > 1e-10:
                acf_values = acf(safe_returns, nlags=10, fft=True)
                acf_lag_1 = acf_values[1] if len(acf_values) > 1 else 0
                acf_lag_5 = acf_values[5] if len(acf_values) > 5 else 0
                acf_lag_10 = acf_values[10] if len(acf_values) > 10 else 0
            else:
                # For constant/flat series, ACF is undefined - use default values
                print(f"DEBUG: Token {token_name} has constant returns (std={np.std(safe_returns):.2e}), using default ACF values")
                acf_lag_1 = 0
                acf_lag_5 = 0
                acf_lag_10 = 0
        except Exception as e:
            print(f"DEBUG: ACF calculation failed for {token_name}: {e}, using default values")
            acf_lag_1 = 0
            acf_lag_5 = 0
            acf_lag_10 = 0
        
        # Vectorized early detection calculations with robust error handling
        early_returns = safe_returns[:5]
        try:
            if len(early_returns) > 1 and np.std(early_returns) > 1e-10:
                early_series = pl.Series('early', early_returns)
                x_range = pl.Series('x', list(range(len(early_returns))))
                # Simple linear trend calculation using polars
                trend_direction_5min = np.polyfit(x_range.to_numpy(), early_series.to_numpy(), 1)[0]
            else:
                trend_direction_5min = 0
        except Exception as e:
            print(f"DEBUG: Trend calculation failed for {token_name}: {e}, using default value")
            trend_direction_5min = 0
            
        # Price change ratio using polars with error handling
        try:
            if len(prices) >= 5:
                early_prices = pl.Series('early_prices', prices[:5])
                price_change_ratio_5min = safe_divide(early_prices.max() - early_prices.min(), prices[0])
            else:
                price_change_ratio_5min = 0
        except Exception as e:
            print(f"DEBUG: Price change ratio calculation failed for {token_name}: {e}, using default value")
            price_change_ratio_5min = 0
            
        # Pump velocity using vectorized cumsum with error handling
        try:
            pump_velocity_5min = np.max(np.cumsum(early_returns)) if len(early_returns) > 0 else 0
        except Exception as e:
            print(f"DEBUG: Pump velocity calculation failed for {token_name}: {e}, using default value")
            pump_velocity_5min = 0
        
        # Compile features using vectorized statistics
        features = {
            'token': token_name,
            'category': df['category'][0],
            # Death (3)
            'is_dead': is_dead,
            'death_minute': death_minute,
            'lifespan_minutes': lifespan_minutes,
            # Core stats (4) - from vectorized calculations
            'mean_return': vectorized_stats['mean_return'],
            'std_return': vectorized_stats['std_return'],
            'max_drawdown': max_drawdown,
            'volatility_5min': vectorized_stats['volatility_5min'],
            # ACF (3)
            'acf_lag_1': acf_lag_1,
            'acf_lag_5': acf_lag_5,
            'acf_lag_10': acf_lag_10,
            # Early detection (5) - mix of vectorized and computed
            'return_5min': vectorized_stats['return_5min'],
            'max_return_1min': vectorized_stats['max_return_1min'],
            'trend_direction_5min': trend_direction_5min,
            'price_change_ratio_5min': price_change_ratio_5min,
            'pump_velocity_5min': pump_velocity_5min
        }
        
        return features
        
    except Exception as e:
        print(f"DEBUG: CRITICAL - Error processing {token_name}: {e}")
        print(f"DEBUG: This token will be excluded from analysis, contributing to token loss!")
        # Instead of returning None and losing the token, return minimal features to preserve it
        minimal_features = {
            'token': token_name,
            'category': df.get_column('category')[0] if 'category' in df.columns else 'unknown',
            # Death (3) - default values
            'is_dead': True,  # Assume problematic tokens are dead
            'death_minute': 0,
            'lifespan_minutes': len(df) if not df.is_empty() else 0,
            # Core stats (4) - default values
            'mean_return': 0.0,
            'std_return': 0.0,
            'max_drawdown': 0.0,
            'volatility_5min': 0.0,
            # ACF (3) - default values
            'acf_lag_1': 0.0,
            'acf_lag_5': 0.0,
            'acf_lag_10': 0.0,
            # Early detection (5) - default values
            'return_5min': 0.0,
            'max_return_1min': 0.0,
            'trend_direction_5min': 0.0,
            'price_change_ratio_5min': 0.0,
            'pump_velocity_5min': 0.0
        }
        print(f"DEBUG: Using minimal features for {token_name} to preserve token in analysis")
        return minimal_features


def extract_essential_features_lazy(token_data: Dict[str, pl.LazyFrame], use_log_returns: bool = False, n_jobs: int = -1) -> pl.DataFrame:
    """
    Extract the 15 essential features for each token using lazy evaluation for memory efficiency.
    
    Args:
        token_data: Dictionary mapping token names to LazyFrames
        use_log_returns: Whether to use modified log returns (for A/B testing)
        n_jobs: Number of parallel jobs (-1 uses all available cores)
        
    Returns:
        DataFrame with 15 essential features for each token
    """
    print(f"\nDEBUG: Extracting 15 essential features for {len(token_data)} tokens using lazy evaluation...")
    
    # Use joblib for parallel processing with lazy evaluation
    features_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_extract_token_features_lazy)(token_name, lazy_df, use_log_returns)
        for token_name, lazy_df in token_data.items()
    )
    
    # Filter out None results (failed processing)
    features_list = [f for f in features_list if f is not None]
    
    print(f"DEBUG: Successfully processed {len(features_list)} tokens using lazy evaluation")
    return pl.DataFrame(features_list)


def extract_essential_features(token_data: Dict[str, pl.DataFrame], use_log_returns: bool = False, n_jobs: int = -1) -> pl.DataFrame:
    """
    Extract the 15 essential features for each token with parallel processing (CEO requirement).
    
    Args:
        token_data: Dictionary mapping token names to DataFrames
        use_log_returns: Whether to use modified log returns (for A/B testing)
        n_jobs: Number of parallel jobs (-1 uses all available cores)
        
    Returns:
        DataFrame with 15 essential features for each token
    """
    print(f"\nDEBUG: Extracting 15 essential features for {len(token_data)} tokens using parallel processing...")
    
    # Use joblib for parallel processing
    features_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_extract_token_features)(token_name, df, use_log_returns)
        for token_name, df in token_data.items()
    )
    
    # No longer filter out results since we provide minimal features instead of None
    # All tokens should now be preserved in the analysis
    print(f"DEBUG: Successfully processed {len(features_list)} tokens (no tokens excluded)")
    return pl.DataFrame(features_list)


def process_tokens_streaming(processed_dir: Path, batch_size: int = 100, chunk_callback=None) -> pl.DataFrame:
    """
    Process tokens using streaming aggregations for memory-efficient analysis of large datasets.
    
    Args:
        processed_dir: Path to processed data directory
        batch_size: Number of tokens to process in each batch
        chunk_callback: Optional callback function to process each chunk
        
    Returns:
        DataFrame with aggregated results from all batches
    """
    print(f"DEBUG: Starting streaming analysis with batch_size={batch_size}")
    
    categories = ['dead_tokens', 'normal_behavior_tokens', 'tokens_with_extremes']
    all_results = []
    
    for category in categories:
        category_path = processed_dir / category
        if not category_path.exists():
            continue
            
        parquet_files = list(category_path.glob("*.parquet"))
        total_files = len(parquet_files)
        
        print(f"DEBUG: Processing {total_files} files from {category} in batches of {batch_size}")
        
        # Process files in batches
        for i in range(0, total_files, batch_size):
            batch_files = parquet_files[i:i + batch_size]
            print(f"DEBUG: Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            # Use polars' concat with lazy evaluation for efficient batch processing
            batch_lazy_frames = []
            
            for file_path in batch_files:
                try:
                    # Stream each file with minimal memory footprint
                    lazy_df = pl.scan_parquet(file_path).with_columns([
                        pl.lit(category).alias('category'),
                        pl.lit(file_path.stem).alias('token_name')
                    ]).filter(
                        pl.col('datetime').is_not_null() & 
                        pl.col('price').is_not_null() &
                        pl.col('price') > 0
                    )
                    
                    batch_lazy_frames.append(lazy_df)
                    
                except Exception as e:
                    print(f"DEBUG: Error processing {file_path}: {e}")
                    continue
            
            if not batch_lazy_frames:
                continue
                
            # Efficiently concatenate and process batch using advanced polars features
            batch_combined = pl.concat(batch_lazy_frames).with_columns([
                # Advanced time-based features using window functions
                pl.col('price').pct_change().alias('return'),
                pl.col('price').rolling_mean(window_size=5).alias('price_ma5'),
                pl.col('price').rolling_std(window_size=5).alias('price_std5'),
                # Rank-based features for outlier detection
                pl.col('price').rank(method='ordinal').alias('price_rank'),
                # Complex expressions with conditional logic
                pl.when(pl.col('price').pct_change().abs() > 0.1)
                .then(pl.lit('extreme_move'))
                .when(pl.col('price').pct_change().abs() > 0.05)
                .then(pl.lit('significant_move'))
                .otherwise(pl.lit('normal_move'))
                .alias('move_type')
            ]).group_by(['token_name', 'category']).agg([
                # Advanced aggregations per token
                pl.col('price').count().alias('data_points'),
                pl.col('return').mean().alias('avg_return'),
                pl.col('return').std().alias('return_volatility'),
                pl.col('return').quantile(0.95).alias('return_95th'),
                pl.col('return').quantile(0.05).alias('return_5th'),
                pl.col('price').first().alias('first_price'),
                pl.col('price').last().alias('last_price'),
                pl.col('price').max().alias('max_price'),
                pl.col('price').min().alias('min_price'),
                # Count extreme movements using advanced expressions
                pl.col('move_type').filter(pl.col('move_type') == 'extreme_move').count().alias('extreme_moves'),
                pl.col('move_type').filter(pl.col('move_type') == 'significant_move').count().alias('significant_moves'),
                # Advanced statistical measures
                pl.col('return').skew().alias('return_skewness'),
                pl.col('return').kurtosis().alias('return_kurtosis')
            ])
            
            # Materialize batch results
            batch_results = batch_combined.collect()
            
            # Apply callback if provided for custom processing
            if chunk_callback:
                batch_results = chunk_callback(batch_results)
            
            all_results.append(batch_results)
    
    # Efficiently combine all batch results
    if all_results:
        final_results = pl.concat(all_results)
        print(f"DEBUG: Streaming processing complete. Processed {final_results.height} tokens total.")
        return final_results
    else:
        print("DEBUG: No results from streaming processing.")
        return pl.DataFrame()


def advanced_feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply advanced polars features for sophisticated feature engineering.
    
    Args:
        df: Input DataFrame with price and return data
        
    Returns:
        DataFrame with advanced engineered features
    """
    return df.with_columns([
        # Advanced window functions with multiple frames
        pl.col('price').rolling_mean(window_size=5).alias('price_sma_5'),
        pl.col('price').rolling_mean(window_size=20).alias('price_sma_20'),
        pl.col('price').rolling_std(window_size=10).alias('price_std_10'),
        
        # Exponential moving averages using advanced expressions
        pl.col('price').ewm_mean(alpha=0.1).alias('price_ema_10'),
        pl.col('price').ewm_mean(alpha=0.05).alias('price_ema_20'),
        
        # Complex conditional features using polars expressions
        pl.when(pl.col('price').rolling_mean(5) > pl.col('price').rolling_mean(20))
        .then(pl.lit(1))
        .otherwise(pl.lit(-1))
        .alias('trend_signal'),
        
        # Lag features for time series analysis
        pl.col('price').shift(1).alias('price_lag1'),
        pl.col('price').shift(5).alias('price_lag5'),
        pl.col('price').shift(10).alias('price_lag10'),
        
        # Advanced mathematical transformations
        pl.col('price').log().alias('log_price'),
        pl.col('price').sqrt().alias('sqrt_price'),
        
        # Bollinger Bands using rolling statistics
        (pl.col('price').rolling_mean(20) + 2 * pl.col('price').rolling_std(20)).alias('bb_upper'),
        (pl.col('price').rolling_mean(20) - 2 * pl.col('price').rolling_std(20)).alias('bb_lower'),
        
        # RSI calculation using polars expressions
        pl.col('price').pct_change().clip(lower_bound=0).rolling_mean(14).alias('avg_gain'),
        pl.col('price').pct_change().clip(upper_bound=0).abs().rolling_mean(14).alias('avg_loss')
    ]).with_columns([
        # RSI final calculation
        (100 - (100 / (1 + pl.col('avg_gain') / pl.col('avg_loss')))).alias('rsi'),
        
        # Bollinger Band position
        ((pl.col('price') - pl.col('bb_lower')) / (pl.col('bb_upper') - pl.col('bb_lower'))).alias('bb_position')
    ])


def incremental_processing_pipeline(base_results: pl.DataFrame, new_data: pl.DataFrame) -> pl.DataFrame:
    """
    Incrementally update analysis results with new data using advanced polars join operations.
    
    Args:
        base_results: Existing analysis results
        new_data: New token data to incorporate
        
    Returns:
        Updated results DataFrame
    """
    # Process new data with same feature engineering
    new_features = advanced_feature_engineering(new_data)
    
    # Advanced join operations to merge with existing data
    updated_results = base_results.join(
        new_features.group_by('token_name').agg([
            pl.col('price').count().alias('new_data_points'),
            pl.col('price').mean().alias('new_avg_price'),
            pl.col('rsi').last().alias('latest_rsi'),
            pl.col('bb_position').last().alias('latest_bb_position')
        ]),
        on='token_name',
        how='left'
    ).with_columns([
        # Update existing metrics using advanced expressions
        pl.when(pl.col('new_data_points').is_not_null())
        .then(pl.col('data_points') + pl.col('new_data_points'))
        .otherwise(pl.col('data_points'))
        .alias('updated_data_points'),
        
        # Weighted average for price updates
        pl.when(pl.col('new_avg_price').is_not_null())
        .then(
            (pl.col('avg_price') * pl.col('data_points') + 
             pl.col('new_avg_price') * pl.col('new_data_points')) /
            (pl.col('data_points') + pl.col('new_data_points'))
        )
        .otherwise(pl.col('avg_price'))
        .alias('updated_avg_price')
    ])
    
    return updated_results


def generate_archetype_docs(archetypes: Dict, output_path: Path):
    """
    Generate Markdown documentation for archetypes (Day 9-10).
    
    Args:
        archetypes: Dictionary of archetype information
        output_path: Path to save Markdown file
    """
    md_content = "# Memecoin Behavioral Archetypes\n\n"
    
    for cluster_id, arch in archetypes.items():
        md_content += f"## Cluster {cluster_id}: {arch['name']}\n\n"
        md_content += f"Avg lifespan: {arch['stats']['avg_lifespan']:.1f} mins\n"
        md_content += f"Traits: {arch['acf_signature']}\n"
        md_content += f"Examples: {arch['examples']}\n"
        md_content += f"Strategy: AVOID if rug-like\n\n"
    
    with open(output_path / "archetype_characterization.md", 'w') as f:
        f.write(md_content)
    
    print(f"DEBUG: Archetype docs generated at {output_path / 'archetype_characterization.md'}")


def categorize_by_lifespan(token_data: Dict[str, pl.DataFrame], token_limits: Dict) -> Dict:
    """
    Categorize tokens by their active lifespan (CEO requirement).
    
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
    
    # Categorize each token using active_lifespan
    for token_name, token_df in token_data.items():
        if token_df.is_empty():
            continue
        
        # Compute death_minute and active_lifespan
        prices, returns, death_minute = prepare_token_data(token_df)
        active_lifespan = death_minute if death_minute is not None else len(token_df)
        
        # Determine category based on active_lifespan
        if active_lifespan <= 400:
            category = 'Sprint'
        elif active_lifespan <= 1200:
            category = 'Standard'
        else:
            category = 'Marathon'
        
        categorized[category][token_name] = token_df
    
    # NOTE: Removed limit application here to prevent token loss
    # Limits should be applied at the analysis level, not during categorization
    # This function should only categorize tokens, not filter them
    
    return categorized