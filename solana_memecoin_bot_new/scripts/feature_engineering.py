import polars as pl
from statsmodels.tsa.stattools import acf
import numpy as np
import sys
import os

# Add config import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from utils import setup_logger

logger = setup_logger(__name__)

def engineer_features(df: pl.DataFrame, lags: int = 10) -> pl.DataFrame:
    """
    Expanded features (20+): Rolling, momentum, RSI, MACD, ACF lags, dump flag (cum returns < threshold in first 2 min).
    Vectorized for efficiency; challenges early dump risks.
    """
    # Check for 'returns' column existence before proceeding
    if "returns" not in df.columns:
        raise ValueError("The 'returns' column must be calculated before engineering features.")

    def acf_per_group(group, lags):
        returns = group["returns"].to_numpy()
        if len(returns) > lags:
            try:
                acf_vals = acf(returns, nlags=lags, fft=False)[1:]
                acf_vals = [val if not np.isnan(val) and not np.isinf(val) else 0.0 for val in acf_vals]
                cols = [pl.lit(val).alias(f"acf_lag_{i}") for i, val in enumerate(acf_vals, 1)]
                return group.with_columns(cols)
            except:
                cols = [pl.lit(0.0).alias(f"acf_lag_{i}") for i in range(1, lags + 1)]
                return group.with_columns(cols)
        else:
            cols = [pl.lit(0.0).alias(f"acf_lag_{i}") for i in range(1, lags + 1)]
            return group.with_columns(cols)
    
    # Step 1: Add ACF features and basic rolling features
    df = df.group_by("token_id").map_groups(lambda g: acf_per_group(g, lags)).with_columns(
        pl.col("max_returns").rolling_mean(5).alias("ma_5"),
        pl.col("max_returns").rolling_mean(15).alias("ma_15"),
        # Improved volatility rolling std with better NaN handling
        pl.col("volatility").fill_nan(0.0).rolling_std(5).alias("vol_std_5_raw"),
        pl.col("returns").shift(1).alias("momentum_lag1"),
        pl.when(pl.col("returns") > 0).then(pl.col("returns")).otherwise(0).rolling_mean(14).alias("up"),
        pl.when(pl.col("returns") < 0).then(pl.col("returns").abs()).otherwise(0).rolling_mean(14).alias("down")
    ).with_columns(
        # Clean up vol_std_5: replace NaN with forward/backward fill, then 0
        pl.col("vol_std_5_raw").forward_fill().backward_fill().fill_nan(0.0).alias("vol_std_5")
    ).drop("vol_std_5_raw")
    
    # Step 2: Add derived features that depend on the first set
    df = df.with_columns(
        (pl.col("ma_5") - pl.col("ma_15")).alias("macd_stub"),
        (100 - (100 / (1 + pl.col("up") / (pl.col("down") + 1e-6)))).alias("rsi_14")
    )
    
    # Dump flag for first 2 min
    result = df.with_columns(
        pl.col("datetime").rank("dense").over("token_id").alias("min_rank")
    ).with_columns(
        pl.when(pl.col("min_rank") <= 2).then(pl.col("returns").cum_sum().over("token_id")).otherwise(None).alias("early_cum_returns")
    ).with_columns(
        pl.when((pl.col("min_rank") <= 2) & (pl.col("early_cum_returns") < config.DUMP_RETURN_THRESHOLD)).then(True).otherwise(False).alias("initial_dump_flag")
    ).with_columns(
        # Improved vol_return_ratio with better division-by-zero protection and NaN handling
        pl.when(pl.col("ma_5").abs() > 1e-8)
        .then(pl.col("vol_std_5") / pl.col("ma_5"))
        .otherwise(0.0)
        .fill_nan(0.0)
        .alias("vol_return_ratio")
    ).drop("min_rank", "early_cum_returns")
    
    # Add NaN count per token for debug (only on float columns)
    if config.DEBUG_MODE:
        float_cols = [col for col in result.columns if col not in ["token_id", "datetime", "split"] and result.select(pl.col(col)).dtypes[0] in [pl.Float64, pl.Float32]]
        nan_counts = result.group_by("token_id").agg([pl.col(col).is_nan().sum().alias(f"{col}_nan_count") for col in float_cols])
        for row in nan_counts.iter_rows(named=True):
            token = row['token_id']
            nan_str = ", ".join([f"{k}: {v}" for k, v in row.items() if k != 'token_id' and v > 0])
            if nan_str:
                logger.debug(f"Token {token}: NaN counts - {nan_str}")
    
    # Final NaN handling - replace any remaining NaN/inf values with 0
    for col in result.columns:
        if col not in ["token_id", "datetime", "split"]:  # Skip non-numeric columns
            col_dtype = result.select(pl.col(col)).dtypes[0]
            if col_dtype in [pl.Float64, pl.Float32]:  # Only apply NaN/inf handling to float columns
                result = result.with_columns(
                    pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite()).then(0.0).otherwise(pl.col(col)).alias(col)
                )
    
    return result

def engineer_early_features(df: pl.DataFrame, window_minutes: int = 10) -> pl.DataFrame:
    """
    Early feature engineering optimized for 10-minute windows with 6-7 minute rolling features.
    Designed to work without NaN issues for early clustering and archetype prediction.
    """
    # Basic validation
    if "returns" not in df.columns:
        raise ValueError("The 'returns' column must be calculated before engineering early features.")
    
    # Step 1: Basic aggregations that work well with short time series
    df = df.with_columns([
        # Price movement features
        pl.col("max_returns").mean().over("token_id").alias("early_mean_returns"),
        pl.col("max_returns").max().over("token_id").alias("early_max_return"),
        pl.col("max_returns").min().over("token_id").alias("early_min_return"),
        pl.col("max_returns").std().over("token_id").alias("early_return_volatility"),
        
        # Price range and momentum
        ((pl.col("max_price") - pl.col("min_price")) / pl.col("avg_price")).over("token_id").alias("early_price_range"),
        pl.col("returns").shift(1).alias("early_momentum_lag1"),
        
        # Volatility aggregations
        pl.col("volatility").mean().over("token_id").alias("early_avg_volatility"),
        pl.col("volatility").max().over("token_id").alias("early_max_volatility"),
    ])
    
    # Step 2: 6-7 minute rolling features (works with 10-minute window, avoids NaN)
    df = df.with_columns([
        # 6-minute rolling features (provides 4 valid data points in 10-min window)
        pl.col("max_returns").rolling_mean(6).alias("rolling_mean_6min"),
        pl.col("max_returns").rolling_std(6).alias("rolling_std_6min"),
        pl.col("volatility").rolling_mean(6).alias("volatility_trend_6min"),
        
        # 7-minute rolling features (provides 3 valid data points in 10-min window) 
        pl.col("max_returns").rolling_mean(7).alias("rolling_mean_7min"),
        pl.col("returns").rolling_sum(7).alias("momentum_7min"),
    ])
    
    # Step 3: Early pattern detection features
    df = df.with_columns([
        # Early dump detection using row number instead of datetime ranking
        pl.arange(pl.len()).over("token_id").alias("early_rank"),
    ]).with_columns([
        # Cumulative returns in first 2 minutes for dump detection (0-based indexing)
        pl.when(pl.col("early_rank") < 2)
        .then(pl.col("returns").cum_sum().over("token_id"))
        .otherwise(None)
        .alias("early_cum_returns_2min")
    ]).with_columns([
        # Early dump flag (0-based indexing)
        pl.when((pl.col("early_rank") < 2) & (pl.col("early_cum_returns_2min") < config.DUMP_RETURN_THRESHOLD))
        .then(True)
        .otherwise(False)
        .alias("early_dump_flag"),
        
        # Price stability measure (coefficient of variation)
        (pl.col("early_return_volatility") / (pl.col("early_mean_returns").abs() + 1e-8)).alias("early_stability_ratio")
    ]).drop(["early_rank", "early_cum_returns_2min"])
    
    # Step 4: Clean NaN/inf values with conservative approach
    early_feature_cols = [
        "early_mean_returns", "early_max_return", "early_min_return", "early_return_volatility",
        "early_price_range", "early_momentum_lag1", "early_avg_volatility", "early_max_volatility",
        "rolling_mean_6min", "rolling_std_6min", "volatility_trend_6min",
        "rolling_mean_7min", "momentum_7min", "early_dump_flag", "early_stability_ratio"
    ]
    
    for col in early_feature_cols:
        if col in df.columns:
            col_dtype = df.select(pl.col(col)).dtypes[0]
            if col_dtype in [pl.Float64, pl.Float32]:
                # Use forward/backward fill, then zero for early features
                df = df.with_columns(
                    pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite())
                    .then(0.0)
                    .otherwise(pl.col(col).forward_fill().backward_fill())
                    .alias(col)
                )
    
    logger.info(f"Early feature engineering completed: {len(early_feature_cols)} features added")
    return df

def add_volume_placeholders(df: pl.DataFrame) -> pl.DataFrame:
    """
    Volume/liquidity placeholders; add ratios for pump/dump detection.
    """
    # Step 1: Create basic placeholder columns and imbalance
    df = df.with_columns(
        pl.lit(0.0).alias("volume"),
        pl.lit(0).alias("buy_count"),
        pl.lit(0).alias("sell_count"),
        pl.lit(1000.0).alias("liquidity")
    ).with_columns(
        (pl.col("buy_count") - pl.col("sell_count")).alias("imbalance"),
        pl.col("volume").rolling_mean(5).alias("avg_volume_5m")
    )
    
    # Step 2: Create derived features that depend on the first set
    result = df.with_columns(
        (pl.col("imbalance") / (pl.col("liquidity") + 1e-6)).alias("imbalance_ratio")
    )
    
    # Final NaN handling for volume features
    for col in result.columns:
        if col not in ["token_id", "datetime", "split"]:  # Skip non-numeric columns
            col_dtype = result.select(pl.col(col)).dtypes[0]
            if col_dtype in [pl.Float64, pl.Float32]:  # Only apply NaN/inf handling to float columns
                result = result.with_columns(
                    pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite()).then(0.0).otherwise(pl.col(col)).alias(col)
                )
    
    return result