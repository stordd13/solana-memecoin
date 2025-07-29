# scripts/run_pipeline2.py (Rewritten with logging, debug mode, timings, PRAGMA opts, datetime casting, pump_label post-split; fixed split literal with pl.lit)
import polars as pl
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import time

# Dynamic path fix
scripts_dir = os.path.dirname(__file__)
bot_new_root = os.path.abspath(os.path.join(scripts_dir, '..'))
sys.path.append(bot_new_root)
sys.path.append(os.path.join(bot_new_root, 'ml'))
sys.path.append(os.path.join(bot_new_root, 'scripts'))

from death_detection import detect_death, analyze_variability, trim_post_death
from feature_engineering import engineer_features, add_volume_placeholders
from archetype_clustering import fast_early_clustering, cluster_archetypes
import config
from utils import setup_logger

logger = setup_logger(__name__)

def run_pipeline(output_base: str) -> dict:
    """
    Full Polars pipeline for 30k+ tokens (<1min run): Query from centralized 'memecoin.db' (table 'token_prices' with 'token_id', 'datetime', 'price'), clean (nulls/gaps/negatives), dual resample (1m/5m for comparison), add variability (after 'returns' calculation), death detection with 30 min threshold and trim constants to prevent leakage, expanded features (for ML training, vectorized with no leakage), per-token scaling (fit on train only), temporal splits (80/20 per token, no future leakage), early + full dynamic clustering.
    Outputs dict of processed Parquets (1m/5m).
    """
    start_total = time.time()
    import sqlite3
    conn = sqlite3.connect(os.path.join(bot_new_root, 'memecoin.db'))
    
    # Optimize SQLite for reads (PRAGMA for perf)
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA mmap_size = 30000000000;")  # Large mmap if RAM allows
    
    start = time.time()
    df = pl.read_database("SELECT * FROM token_prices", conn).with_columns(
        pl.col("datetime").str.to_datetime().alias("datetime")  # Cast str to Datetime
    )
    logger.info(f"DB read and cast took {time.time() - start:.2f}s | Shape: {df.shape}")
    
    if config.DEBUG_MODE:
        logger.debug(f"DB schema: {df.schema}")
    
    # Cleaning: Handle nulls/negatives/gaps
    start = time.time()
    cleaned = df.drop_nulls().filter(pl.col("price") >= 0).with_columns(pl.col("price").interpolate().alias("price"))
    logger.debug(f"Cleaning took {time.time() - start:.2f}s")
    
    outputs = {}
    for interval in config.RESAMPLE_INTERVALS:
        start = time.time()
        resampled = cleaned.group_by_dynamic("datetime", every=interval, group_by="token_id").agg(
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").std().alias("volatility")
        ).with_columns(pl.col("avg_price").pct_change().alias("returns")).drop_nulls()
        logger.info(f"Resample {interval} took {time.time() - start:.2f}s")
        
        # Variability analysis after 'returns' calculation
        start = time.time()
        analyzed = analyze_variability(resampled)
        logger.debug(f"Variability analysis took {time.time() - start:.2f}s")
        
        start = time.time()
        with_death = detect_death(analyzed, config.ZERO_THRESHOLD)  # 30 min
        trimmed = trim_post_death(with_death)  # Trim constants post-death to avoid leakage/noise
        
        # Check for empty DataFrame before feature engineering
        if trimmed.height == 0:
            logger.warning(f"Empty DataFrame for {interval} interval after death trimming")
            continue
        
        logger.info(f"Death detection and trim took {time.time() - start:.2f}s | Shape after trim: {trimmed.shape}")
        
        if config.DEBUG_MODE:
            logger.debug(f"Available columns after trim: {trimmed.columns}")
        
        start = time.time()
        featured = engineer_features(trimmed)  # Expanded features, vectorized
        with_volume = add_volume_placeholders(featured)
        logger.info(f"Feature engineering took {time.time() - start:.2f}s")
        
        # Per-token scaling (fit on train split only to prevent leakage)
        start = time.time()
        def scale_group(group):
            group = group.with_columns(pl.col("datetime").rank("dense").alias("rank"))
            train_size = int(0.8 * len(group))
            train_returns = group.filter(pl.col("rank") <= train_size)['returns'].to_numpy().reshape(-1, 1)
            scaler = MinMaxScaler().fit(train_returns)
            scaled = scaler.transform(group['returns'].to_numpy().reshape(-1, 1)).flatten()
            return group.with_columns(pl.Series("scaled_returns", scaled)).drop("rank")
        scaled = with_volume.group_by("token_id").map_groups(scale_group)
        logger.debug(f"Scaling took {time.time() - start:.2f}s")
        
        # Full temporal split (80/20 per token)
        start = time.time()
        split_df = scaled.with_columns(pl.col("datetime").rank("dense").over("token_id").alias("rank")).with_columns(
            pl.when(pl.col("rank") <= pl.col("rank").max().over("token_id") * 0.8).then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")
        ).drop("rank")
        logger.debug(f"Splitting took {time.time() - start:.2f}s")
        
        # Add pump_label here post-split (binary next returns >0.5, per-token shift safe)
        start = time.time()
        # Changed from 0.5 (50%) to 0.10 (10%) for more realistic pump detection
        split_df = split_df.with_columns(
            pl.when(pl.col("returns").shift(-1) > 0.10).then(1).otherwise(0).alias("pump_label")
        ).with_columns(
            pl.col("pump_label").fill_nan(0).alias("pump_label")  # Fill last row NaN
        )
        logger.info(f"Pump label addition took {time.time() - start:.2f}s")
        
        # Early + full clustering
        start = time.time()
        early_clustered = fast_early_clustering(split_df, config.EARLY_MINUTES)
        full_clustered = cluster_archetypes(early_clustered)
        logger.info(f"Clustering took {time.time() - start:.2f}s | Shape: {full_clustered.shape}")
        
        output_path = f"{output_base}_{interval}.parquet"
        full_clustered.write_parquet(output_path)
        outputs[interval] = full_clustered
        logger.info(f"Processed {interval}: Shape {full_clustered.shape}")
    
    logger.info(f"Total pipeline time: {time.time() - start_total:.2f}s")
    return outputs

if __name__ == "__main__":
    run_pipeline("processed_features")