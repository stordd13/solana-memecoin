# scripts/run_pipeline3.py - Unified pipeline without archetype clustering for fast trading strategy
# Core: Load raw Parquets, clean, resample, engineer features, scale/split temporally (no archetypes)
# Output: processed_features_{interval}_unified.parquet ready for unified transformer/RL training
import polars as pl
from sklearn.preprocessing import RobustScaler
import sys
import os
import glob
import time
import numpy as np
import gc
import psutil

# Dynamic path fix
scripts_dir = os.path.dirname(__file__)
bot_new_root = os.path.abspath(os.path.join(scripts_dir, '..'))
sys.path.append(bot_new_root)
sys.path.append(os.path.join(bot_new_root, 'ml'))
sys.path.append(os.path.join(bot_new_root, 'scripts'))

from death_detection import backward_raw_check, analyze_variability, detect_death, trim_post_death
from feature_engineering import engineer_features, engineer_early_features, add_volume_placeholders
import config
from utils import setup_logger

logger = setup_logger(__name__)

def log_memory_usage(stage: str):
    """Log current memory usage for monitoring pipeline efficiency"""
    try:
        memory_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Memory usage at {stage}: {memory_gb:.2f}GB")
        return memory_gb
    except:
        logger.warning(f"Could not get memory usage for {stage}")
        return 0.0

def aggressive_memory_cleanup():
    """Perform aggressive memory cleanup and garbage collection"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    log_memory_usage("after cleanup")

def load_and_clean_files(data_dir: str) -> list[pl.LazyFrame]:
    """Loads and cleans Parquet files lazily (per-file dtype cast, fills, filters). Skips metadata."""
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if not all_files:
        raise ValueError("No Parquet files found in data/raw/dataset/")
    
    # Skip metadata files (files starting with __ or containing 'tokens_list')
    parquet_files = [
        f for f in all_files 
        if not (os.path.basename(f).startswith('__') or 'tokens_list' in os.path.basename(f))
    ]
    logger.info(f"Found {len(parquet_files)} Parquet files to process")
    
    lazy_frames = []
    for file_path in parquet_files:
        try:
            # LazyFrame with optimizations
            lazy_df = (
                pl.scan_parquet(file_path)
                .with_columns([
                    # Rename price to avg_price and ensure it's Float64
                    pl.col("price").cast(pl.Float64).alias("avg_price"),
                    # Add token_id from filename
                    pl.lit(os.path.basename(file_path).replace('.parquet', '')).alias("token_id")
                ])
                .select(["token_id", "datetime", "avg_price"])  # Select only needed columns
                .filter(pl.col("avg_price") > 0)  # Remove invalid prices
                .fill_nan(None)  # Convert NaN to null for better handling
            )
            lazy_frames.append(lazy_df)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    logger.info(f"Successfully loaded {len(lazy_frames)} files as lazy frames")
    return lazy_frames

def resample_and_engineer(combined_df: pl.LazyFrame, interval: str) -> pl.LazyFrame:
    """Resample to interval and engineer features (unified for all tokens)"""
    start = time.time()
    logger.info(f"Resampling to {interval} and engineering features...")
    
    # Resample to target interval
    if interval == "1m":
        freq = "1m"
    elif interval == "5m":
        freq = "5m"
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    
    # Group by token and resample
    resampled = (
        combined_df
        .sort(["token_id", "datetime"])
        .group_by("token_id")
        .agg([
            pl.col("datetime").dt.truncate(freq).alias("datetime_resampled"),
            pl.col("avg_price").alias("avg_price_list")
        ])
        .explode(["datetime_resampled", "avg_price_list"])
        .group_by(["token_id", "datetime_resampled"])
        .agg([
            pl.col("avg_price_list").mean().alias("avg_price"),
            pl.col("avg_price_list").max().alias("max_price"),
            pl.col("avg_price_list").min().alias("min_price"),
            pl.col("avg_price_list").std().alias("volatility")
        ])
        .rename({"datetime_resampled": "datetime"})
        .sort(["token_id", "datetime"])
    )
    
    # Add returns calculation
    resampled = resampled.with_columns([
        # Calculate returns
        (pl.col("avg_price") - pl.col("avg_price").shift(1)).alias("price_diff"),
        pl.col("avg_price").shift(1).alias("prev_price")
    ]).with_columns([
        # Calculate percentage returns
        pl.when(pl.col("prev_price") > 0)
        .then((pl.col("price_diff") / pl.col("prev_price")) * 100)
        .otherwise(0.0)
        .alias("returns")
    ]).drop(["price_diff", "prev_price"])
    
    # Add max_returns column (same as returns for now)
    resampled = resampled.with_columns([
        pl.col("returns").alias("max_returns")
    ])
    
    logger.info(f"Resampling to {interval} took {time.time() - start:.2f}s")
    return resampled

def add_token_metadata(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add LEGITIMATE token metadata features (NO DATA LEAKAGE)"""
    logger.info("Adding legitimate token metadata features (no future data)...")
    
    # Calculate point-in-time features WITHOUT using future data
    with_metadata = (
        df
        .sort(["token_id", "datetime"])  # Ensure chronological order
        .with_columns([
            # Token start time (legitimate - no future data)
            pl.col("datetime").first().over("token_id").alias("token_start"),
            # Initial price (legitimate - launch price)
            pl.col("avg_price").first().over("token_id").alias("initial_price")
        ])
        .with_columns([
            # Minutes since token start (legitimate - current elapsed time)
            (pl.col("datetime") - pl.col("token_start")).dt.total_minutes().alias("minutes_since_start"),
            # Current return from initial price (legitimate - uses current price vs launch)
            (pl.col("avg_price") / pl.col("initial_price") - 1).alias("current_total_return"),
            # Rolling volatility regime (last 10 periods only - no future data)
            pl.col("volatility").rolling_mean(10).over("token_id").alias("recent_avg_volatility")
        ])
        .with_columns([
            # Volatility regime based on recent data only
            pl.when(pl.col("recent_avg_volatility") > 0.1).then(pl.lit("high"))
            .when(pl.col("recent_avg_volatility") < 0.02).then(pl.lit("low"))
            .otherwise(pl.lit("medium")).alias("volatility_regime"),
            # Age category (legitimate time-based feature)
            pl.when(pl.col("minutes_since_start") < 60).then(pl.lit("new"))
            .when(pl.col("minutes_since_start") < 240).then(pl.lit("developing"))
            .otherwise(pl.lit("mature")).alias("token_age_category")
        ])
        .drop(["token_start"])  # Remove intermediate column
    )
    
    return with_metadata

def temporal_split_and_label(processed: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Temporal splits within each token + pump labeling (unified approach)"""
    start = time.time()
    
    # Convert LazyFrame to DataFrame for processing
    if isinstance(processed, pl.LazyFrame):
        processed = processed.collect()
    
    logger.info("Creating temporal splits and pump labels...")
    
    # Add temporal split (80% train, 20% test within each token)
    split_df = processed.with_columns(
        pl.col("datetime").rank("dense").over("token_id").alias("rank")
    ).with_columns(
        pl.when(pl.col("rank") <= pl.col("rank").max().over("token_id") * 0.8)
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).drop("rank")
    
    # Add pump label using configured threshold (10% instead of 50%)
    split_df = split_df.with_columns(
        pl.when(pl.col("returns").shift(-1) > config.PUMP_RETURN_THRESHOLD)
        .then(1)
        .otherwise(0)
        .alias("pump_label")
    ).with_columns(
        pl.col("pump_label").fill_null(0).alias("pump_label")  # Fill last row
    )
    
    logger.info(f"Temporal splits and labeling took {time.time() - start:.2f}s")
    
    # Log pump statistics
    pump_rate = split_df['pump_label'].sum() / split_df.height
    logger.info(f"Overall pump rate with {config.PUMP_RETURN_THRESHOLD*100}% threshold: {pump_rate:.3%}")
    
    return split_df

def scale_per_token_unified(with_metadata: pl.DataFrame) -> pl.DataFrame:
    """Per-token scaling for unified model (no archetype separation)"""
    start = time.time()
    logger.info("Scaling features per token...")
    
    def scale_group(group):
        """Scale features within each token using train data only"""
        if len(group) < 2:
            logger.debug(f"Skipping small group for token {group['token_id'][0]} (len={len(group)})")
            return group.with_columns(pl.lit(0.0).alias("scaled_returns"))
        
        # Split by train/test
        group = group.with_columns(pl.col("datetime").rank("dense").alias("rank"))
        train_size = int(0.8 * len(group))
        
        if train_size < 1:
            logger.debug(f"No train data for token {group['token_id'][0]}, setting scaled to 0")
            return group.with_columns(pl.lit(0.0).alias("scaled_returns")).drop("rank")
        
        train_returns = group.filter(pl.col("rank") <= train_size)['returns'].to_numpy().reshape(-1, 1)
        
        # Check for valid data
        if np.all(np.isnan(train_returns)) or len(train_returns) < 2 or np.nanvar(train_returns) < 1e-6:
            logger.debug(f"Token {group['token_id'][0]}: Invalid train data, setting scaled to 0")
            return group.with_columns(pl.lit(0.0).alias("scaled_returns")).drop("rank")
        
        # Scale using RobustScaler (good for extreme crypto volatility)
        try:
            scaler = RobustScaler().fit(train_returns)
            scaled = scaler.transform(group['returns'].to_numpy().reshape(-1, 1)).flatten()
            return group.with_columns(pl.Series("scaled_returns", scaled)).drop("rank")
        except Exception as e:
            logger.debug(f"Scaling failed for token {group['token_id'][0]}: {e}")
            return group.with_columns(pl.lit(0.0).alias("scaled_returns")).drop("rank")
    
    scaled = with_metadata.group_by("token_id").map_groups(scale_group)
    logger.info(f"Per-token scaling took {time.time() - start:.2f}s")
    
    return scaled

def main():
    """Main pipeline execution - unified approach without archetypes"""
    logger.info("Starting unified pipeline (no archetypes) for fast trading strategy")
    log_memory_usage("start")
    
    # Load data files
    data_dir = config.DATA_RAW_DIR
    logger.info(f"Loading data from {data_dir}")
    lazy_frames = load_and_clean_files(data_dir)
    
    if not lazy_frames:
        logger.error("No data files loaded, exiting")
        return
    
    # Combine all lazy frames
    logger.info("Combining all data files...")
    combined_lazy = pl.concat(lazy_frames)
    log_memory_usage("after data loading")
    
    # Load death summary for filtering
    death_summary_path = os.path.join(bot_new_root, 'data/processed/death_summary.parquet')
    if os.path.exists(death_summary_path):
        logger.info("Loading death summary for data quality filtering...")
        death_summary = pl.scan_parquet(death_summary_path)
        
        # Join death info and filter out data after death time
        trimmed = (
            combined_lazy
            .join(death_summary, on="token_id", how="left")
            .filter(
                # Keep data before death_time, or all data if no death_time
                pl.col("death_time").is_null() | 
                (pl.col("datetime") <= pl.col("death_time"))
            )
            .drop(["death_time", "death_hours", "max_streak"])  # Clean up death summary columns
        )
    else:
        logger.warning("Death summary not found, skipping death filtering")
        trimmed = combined_lazy
    
    aggressive_memory_cleanup()
    
    # Process both intervals
    for interval in config.RESAMPLE_INTERVALS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {interval} interval")
        logger.info(f"{'='*50}")
        
        # Resample and add basic features
        resampled = resample_and_engineer(trimmed, interval)
        
        # Add volume placeholders (for compatibility)
        with_volume = add_volume_placeholders(resampled)
        
        # Engineer features
        logger.info("Engineering features...")
        featured = engineer_features(with_volume)
        
        # Add token metadata for unified model
        with_metadata = add_token_metadata(featured)
        
        # Collect to DataFrame for further processing
        logger.info("Collecting data for temporal processing...")
        df_collected = with_metadata.collect()
        log_memory_usage(f"after {interval} collection")
        
        # Temporal splits and pump labeling
        split_labeled = temporal_split_and_label(df_collected, interval)
        
        # Per-token scaling (no archetype grouping)
        final_df = scale_per_token_unified(split_labeled)
        
        # Save unified dataset
        output_filename = f"processed_features_{interval}_unified.parquet"
        output_path = os.path.join(bot_new_root, output_filename)
        
        logger.info(f"Saving {interval} processed data...")
        final_df.write_parquet(output_path)
        
        # Log final statistics
        logger.info(f"\n{interval} Processing Complete:")
        logger.info(f"Final shape: {final_df.shape}")
        logger.info(f"Unique tokens: {final_df['token_id'].n_unique()}")
        logger.info(f"Pump rate: {(final_df['pump_label'].sum() / final_df.height):.3%}")
        logger.info(f"Train/test split: {final_df['split'].value_counts()}")
        logger.info(f"Saved to: {output_path}")
        
        # Memory cleanup between intervals
        del df_collected, split_labeled, final_df
        aggressive_memory_cleanup()
    
    logger.info("\n🎉 Unified pipeline completed successfully!")
    logger.info("Ready for transformer and RL training without archetypes")
    log_memory_usage("final")

if __name__ == "__main__":
    main()