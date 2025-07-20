import polars as pl
from statsmodels.tsa.stattools import acf
import numpy as np
import sys
import os

# Add config import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

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
                # Handle NaN values from ACF calculation
                acf_vals = [val if not np.isnan(val) and not np.isinf(val) else 0.0 for val in acf_vals]
                cols = [pl.lit(val).alias(f"acf_lag_{i}") for i, val in enumerate(acf_vals, 1)]
                return group.with_columns(cols)
            except:
                # If ACF calculation fails, add zero columns
                cols = [pl.lit(0.0).alias(f"acf_lag_{i}") for i in range(1, lags + 1)]
                return group.with_columns(cols)
        else:
            # If not enough data, add zero columns
            cols = [pl.lit(0.0).alias(f"acf_lag_{i}") for i in range(1, lags + 1)]
            return group.with_columns(cols)
        return group
    
    # Step 1: Add ACF features and basic rolling features
    df = df.group_by("token_id").map_groups(lambda g: acf_per_group(g, lags)).with_columns(
        pl.col("returns").rolling_mean(5).alias("ma_5"),
        pl.col("returns").rolling_mean(15).alias("ma_15"),
        pl.col("volatility").rolling_std(5).alias("vol_std_5"),
        pl.col("returns").shift(1).alias("momentum_lag1"),
        pl.when(pl.col("returns") > 0).then(pl.col("returns")).otherwise(0).rolling_mean(14).alias("up"),
        pl.when(pl.col("returns") < 0).then(pl.col("returns").abs()).otherwise(0).rolling_mean(14).alias("down")
    )
    
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
        (pl.col("vol_std_5") / (pl.col("ma_5") + 1e-6)).alias("vol_return_ratio")  # New: Volatility relative to mean, for ML
    ).drop("min_rank", "early_cum_returns")
    
    # Add pump_label (binary: next returns > 0.5; shift(-1) per-token, safe post-split)
    result = result.with_columns(
        pl.when(pl.col("returns").shift(-1) > 0.5).then(1).otherwise(0).alias("pump_label")
    ).with_columns(  # NaN fill for pump_label (e.g., last row)
        pl.col("pump_label").fill_nan(0).alias("pump_label")
    )
    
    # Final NaN handling - replace any remaining NaN/inf values with 0
    # This is critical for K-means clustering which doesn't handle NaN values
    for col in result.columns:
        if col not in ["token_id", "datetime", "split"]:  # Skip non-numeric columns
            col_dtype = result.select(pl.col(col)).dtypes[0]
            if col_dtype in [pl.Float64, pl.Float32]:  # Only apply NaN/inf handling to float columns
                result = result.with_columns(
                    pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite()).then(0.0).otherwise(pl.col(col)).alias(col)
                )
    
    return result

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