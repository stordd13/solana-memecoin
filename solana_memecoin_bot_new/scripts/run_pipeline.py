# scripts/run_pipeline.py - Stabilized pipeline for Solana memecoin data processing (Polars lazy ops, ~30k tokens, <1 min runtime)
# Core: Load raw Parquets per-file (clean/cast), concat/join death_summary lazily, resample/trim, engineer features (NaN fills), scale/split temporally (leakage prevention), cluster archetypes.
# Outputs: Parquet per interval (e.g., processed_features_5m.parquet) with archetype, pump_label, ready for per-archetype ML/RL.
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
from archetype_clustering import fast_early_clustering, cluster_archetypes
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
    
    dfs = []
    logger.info(f"Found {len(all_files)} total parquet files")
    
    for file_path in all_files:
        start = time.time()
        try:
            lazy_df = pl.scan_parquet(file_path)
            schema = lazy_df.collect_schema()
            
            if config.DEBUG_MODE:
                logger.debug(f"File: {os.path.basename(file_path)} | Schema: {dict(schema)}")
            
            # Skip metadata
            if os.path.basename(file_path).startswith('_tokens_list'):
                continue
            
            if 'datetime' in schema and 'price' in schema and len(schema) == 2:
                token_id = os.path.splitext(os.path.basename(file_path))[0]
                # Conditional datetime cast
                if schema['datetime'] == pl.String:
                    lazy_df = lazy_df.with_columns(pl.col("datetime").str.to_datetime().alias("datetime"))
                elif schema['datetime'] != pl.Datetime:
                    lazy_df = lazy_df.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))
                # Per-file clean: Forward/backward fill NaN, filter negatives, interpolate
                lazy_df = lazy_df.with_columns(pl.col("price").forward_fill().backward_fill()).filter(pl.col("price") >= 0).with_columns(pl.col("price").interpolate().alias("price"))
                lazy_df = lazy_df.with_columns(pl.lit(token_id).alias("token_id"))
                dfs.append(lazy_df)
                logger.info(f"Loaded and cleaned {os.path.basename(file_path)} in {time.time() - start:.2f}s")
            else:
                logger.warning(f"Skipped {os.path.basename(file_path)} (schema mismatch)")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Loaded {len(dfs)} token files")
    if not dfs:
        raise ValueError("No valid token files")
    return dfs

def perform_resample(sorted_data: pl.LazyFrame, interval: str) -> pl.LazyFrame:
    """Resamples with mean/max/min prices (capture extremes), std volatility. Computes returns on max (for pumps), avg (stability), intra max/min return."""
    resampled = sorted_data.group_by_dynamic("datetime", every=interval, group_by="token_id").agg(
        pl.col("price").mean().alias("avg_price"),
        pl.col("price").max().alias("max_price"),  # For extreme pumps
        pl.col("price").min().alias("min_price"),  # For dumps
        pl.col("price").std().alias("volatility"),
        ((pl.col("price").max() / (pl.col("price").min() + 1e-10)) - 1).alias("intra_interval_max_return")  # Intra-window extreme
    ).with_columns(
        pl.col("max_price").pct_change().alias("max_returns"),  # Extreme-based
        pl.col("avg_price").pct_change().alias("avg_returns"),  # Smoothed
        pl.col("min_price").pct_change().alias("min_returns")   # For downside
    ).with_columns(
        pl.col("max_returns").fill_nan(0.0),
        pl.col("avg_returns").fill_nan(0.0),
        pl.col("min_returns").fill_nan(0.0),
        pl.col("max_returns").alias("returns")  # Backward compatibility - use max_returns as primary returns
    )
    return resampled

def add_splits_and_labels(scaled: pl.DataFrame) -> pl.DataFrame:
    """Adds temporal split (80/20 per token) and pump_label (binary: next returns >0.5, fill last NaN). Post-split to prevent leakage."""
    start = time.time()
    split_df = scaled.with_columns(pl.col("datetime").rank("dense").over("token_id").alias("rank")).with_columns(
        pl.when(pl.col("rank") <= pl.col("rank").max().over("token_id") * 0.8).then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")
    ).drop("rank")
    
    # Changed from 0.5 (50%) to 0.10 (10%) for more realistic pump detection
    # 10% gains are much more tradeable and provide better class balance (~2% pump rate vs 0.5%)
    split_df = split_df.with_columns(
        pl.when(pl.col("returns").shift(-1) > 0.10).then(1).otherwise(0).alias("pump_label")
    ).with_columns(pl.col("pump_label").fill_nan(0).alias("pump_label"))  # Fill last row
    
    logger.info(f"Splits and labels took {time.time() - start:.2f}s")
    return split_df

def scale_per_token(with_volume: pl.DataFrame) -> pl.DataFrame:
    """Dual-track scaling: preserve raw values for EDA, scale volatility metrics for clustering. Uses RobustScaler for crypto data."""
    start = time.time()
    
    # Preserve raw values for plotting - create raw copies before any scaling
    with_volume = with_volume.with_columns([
        pl.col("returns").alias("raw_returns"),
        pl.col("max_returns").alias("raw_max_returns"),
        pl.col("min_returns").alias("raw_min_returns"),
        pl.col("avg_returns").alias("raw_avg_returns"),
        pl.col("intra_interval_max_return").alias("raw_intra_interval_max_return"),
        pl.col("volatility").alias("raw_volatility")
    ])
    
    def scale_group(group):
        if len(group) < 2:
            # Return with raw values preserved and minimal scaling for small groups
            return group.with_columns([
                pl.lit(0.0).alias("scaled_volatility"),
                pl.lit(0.0).alias("scaled_vol_metrics")
            ])
            
        group = group.with_columns(pl.col("datetime").rank("dense").alias("rank"))
        train_size = int(0.8 * len(group))
        if train_size < 1:
            return group.with_columns([
                pl.lit(0.0).alias("scaled_volatility"),
                pl.lit(0.0).alias("scaled_vol_metrics")
            ]).drop("rank")
        
        # Scale only volatility-related metrics for clustering, NOT returns
        # Use RobustScaler which handles outliers better than MinMaxScaler for crypto
        train_data = group.filter(pl.col("rank") <= train_size)
        
        # Scale volatility with NaN validation (important for clustering)
        train_vol = train_data['volatility'].to_numpy().reshape(-1, 1)
        
        # Check for valid data before scaling
        valid_vol_mask = ~(np.isnan(train_vol).all() or np.isinf(train_vol).all())
        if valid_vol_mask and len(train_vol[~np.isnan(train_vol)]) > 0:
            vol_scaler = RobustScaler().fit(train_vol)
            all_vol = group['volatility'].to_numpy().reshape(-1, 1)
            scaled_vol = vol_scaler.transform(all_vol).flatten()
        else:
            # Fallback: use simple normalization or zeros for all-NaN volatility
            all_vol = group['volatility'].to_numpy()
            if len(all_vol[~np.isnan(all_vol)]) > 0:
                vol_mean = np.nanmean(all_vol)
                vol_std = np.nanstd(all_vol)
                if vol_std > 0:
                    scaled_vol = (all_vol - vol_mean) / vol_std
                else:
                    scaled_vol = np.zeros_like(all_vol)
                # Fill NaN values with 0
                scaled_vol = np.nan_to_num(scaled_vol, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                scaled_vol = np.zeros(len(all_vol))
        
        # Create a combined volatility metric for clustering with NaN handling
        vol_metric_columns = ['volatility', 'vol_std_5', 'vol_return_ratio']
        available_columns = [col for col in vol_metric_columns if col in group.columns]
        
        if available_columns:
            vol_metrics = group.select(available_columns).to_numpy()
            train_vol_metrics = train_data.select(available_columns).to_numpy()
            
            # Check each column for valid data
            valid_data_exists = False
            for i in range(train_vol_metrics.shape[1]):
                col_data = train_vol_metrics[:, i]
                if len(col_data[~np.isnan(col_data)]) > 1:  # Need at least 2 valid points
                    valid_data_exists = True
                    break
            
            if valid_data_exists:
                # Clean training data: replace NaN/inf with column medians
                train_vol_metrics_clean = np.copy(train_vol_metrics)
                vol_metrics_clean = np.copy(vol_metrics)
                
                for i in range(train_vol_metrics_clean.shape[1]):
                    col_data = train_vol_metrics_clean[:, i]
                    if len(col_data[~np.isnan(col_data)]) > 0:
                        col_median = np.nanmedian(col_data)
                        train_vol_metrics_clean[np.isnan(train_vol_metrics_clean[:, i]), i] = col_median
                        vol_metrics_clean[np.isnan(vol_metrics_clean[:, i]), i] = col_median
                    else:
                        train_vol_metrics_clean[:, i] = 0.0
                        vol_metrics_clean[:, i] = 0.0
                
                # Replace inf values with finite values
                train_vol_metrics_clean = np.nan_to_num(train_vol_metrics_clean, nan=0.0, posinf=1.0, neginf=-1.0)
                vol_metrics_clean = np.nan_to_num(vol_metrics_clean, nan=0.0, posinf=1.0, neginf=-1.0)
                
                vol_metrics_scaler = RobustScaler().fit(train_vol_metrics_clean)
                scaled_vol_metrics = vol_metrics_scaler.transform(vol_metrics_clean)
                # Take mean of scaled volatility metrics as single feature for clustering
                scaled_vol_metrics_combined = np.mean(scaled_vol_metrics, axis=1)
            else:
                # All columns are NaN - use zeros
                scaled_vol_metrics_combined = np.zeros(len(group))
                if config.DEBUG_MODE:
                    logger.debug(f"Token {group['token_id'].unique()[0]}: All volatility metrics are NaN, using zeros")
        else:
            scaled_vol_metrics_combined = np.zeros(len(group))
        
        return group.with_columns([
            pl.Series("scaled_volatility", scaled_vol),
            pl.Series("scaled_vol_metrics", scaled_vol_metrics_combined)
        ]).drop("rank")
    
    scaled = with_volume.group_by("token_id").map_groups(scale_group)
    
    # Log data quality summary
    total_tokens = scaled.select(pl.col('token_id').n_unique()).item()
    tokens_with_zero_vol = scaled.filter(pl.col('scaled_volatility') == 0).select(pl.col('token_id').n_unique()).item()
    tokens_with_zero_metrics = scaled.filter(pl.col('scaled_vol_metrics') == 0).select(pl.col('token_id').n_unique()).item()
    
    logger.info(f"Dual-track scaling completed: {total_tokens} tokens processed")
    logger.info(f"Data quality: {tokens_with_zero_vol} tokens with zero volatility, {tokens_with_zero_metrics} tokens with zero vol_metrics")
    logger.info(f"Raw values preserved, volatility scaled with RobustScaler. Time: {time.time() - start:.2f}s")
    return scaled

def run_pipeline(output_base: str) -> dict:
    """Main pipeline: Run death check, load/clean/concat/join lazily, resample/analyze/trim, features/scale/split/label, cluster. Outputs dict of Parquets per interval."""
    start_total = time.time()
    
    # Pre-pipeline: Run backward death check
    death_summary = backward_raw_check(config.DATA_RAW_DIR)
    
    # Load and clean files lazily
    dfs = load_and_clean_files(config.DATA_RAW_DIR)
    
    # Lazy concat and sort
    start = time.time()
    df = pl.concat(dfs, how="vertical")
    logger.info(f"Concat took {time.time() - start:.2f}s")
    
    # Lazy join death_summary, add is_dead
    df = df.join(death_summary.lazy().select("token_id", "death_time"), on="token_id", how="left")
    df = df.with_columns(pl.when(pl.col("datetime") >= pl.col("death_time")).then(True).otherwise(False).alias("is_dead"))
    
    # Log dead/total (collect scalars only)
    total_tokens = df.select(pl.col('token_id').n_unique()).collect().item()
    dead_count = df.filter(pl.col('is_dead')).select(pl.col('token_id').n_unique()).collect().item()
    logger.info(f"Dead tokens post-join: {dead_count} / {total_tokens}")
    
    # Sort lazily
    start = time.time()
    sorted_data = df.sort(["token_id", "datetime"])
    logger.debug(f"Sort took {time.time() - start:.2f}s")
    
    # Memory-efficient processing: handle each interval separately with aggressive cleanup
    log_memory_usage("pipeline start")
    
    for interval_idx, interval in enumerate(config.RESAMPLE_INTERVALS):
        interval_start = time.time()
        logger.info(f"Processing interval {interval} ({interval_idx + 1}/{len(config.RESAMPLE_INTERVALS)})")
        log_memory_usage(f"start {interval}")
        
        # Pre-resample token count (lazy-safe)
        pre_resample_tokens = sorted_data.select(pl.col('token_id').n_unique()).collect().item()
        logger.debug(f"Pre-resample {interval} tokens: {pre_resample_tokens}")
        
        # Resample
        resampled = perform_resample(sorted_data, interval)
        log_memory_usage(f"after resample {interval}")
        
        # Post-resample count and shape (collect for log)
        post_resample_tokens = resampled.select(pl.col('token_id').n_unique()).collect().item()
        resampled_shape = resampled.collect().shape
        logger.debug(f"Post-resample {interval} tokens: {post_resample_tokens}")
        logger.info(f"Resample {interval} took {time.time() - interval_start:.2f}s | Shape: {resampled_shape}")
        
        # Analyze variability, detect/trim death (collect here as functions return eager)
        analyzed = analyze_variability(resampled.collect())  # Collect for non-lazy func
        # Clear resampled to free memory immediately
        del resampled
        log_memory_usage(f"after analyze {interval}")
        
        with_death = detect_death(analyzed, config.ZERO_THRESHOLD)
        # Clear analyzed to free memory
        del analyzed
        
        trimmed = trim_post_death(with_death)
        # Clear with_death to free memory
        del with_death
        log_memory_usage(f"after death detection {interval}")
        
        # Safeguard: Add placeholder if trim empties dead tokens (retain for archetypes)
        if trimmed.height == 0 or trimmed.select(pl.col('token_id').n_unique()).item() < post_resample_tokens:
            logger.warning(f"Adding placeholders for emptied tokens in {interval}")
            # Temporarily recreate token list for placeholders
            temp_resampled = perform_resample(sorted_data, interval)
            unique_tokens = temp_resampled.select('token_id').unique().collect()
            del temp_resampled  # Clean up immediately
            
            placeholder = pl.DataFrame({
                "token_id": unique_tokens['token_id'],
                "datetime": pl.lit(None).cast(pl.Datetime),
                "avg_price": 0.0,
                "volatility": 0.0,
                "returns": 0.0,
                "cv": 0.0,
                "zero_streak": 0,
                "is_dead": True
            })
            trimmed = pl.concat([trimmed, placeholder.filter(~placeholder['token_id'].is_in(trimmed['token_id']))])
            del unique_tokens, placeholder
        
        logger.info(f"Post-trim shape: {trimmed.shape}")
        log_memory_usage(f"after trim {interval}")
        
        # Features: Apply both full and early feature engineering
        featured = engineer_features(trimmed)
        del trimmed  # Free memory immediately
        log_memory_usage(f"after full features {interval}")
        
        with_early_features = engineer_early_features(featured)
        del featured  # Free memory immediately  
        log_memory_usage(f"after early features {interval}")
        
        with_volume = add_volume_placeholders(with_early_features)
        del with_early_features  # Free memory immediately
        log_memory_usage(f"after volume features {interval}")
        
        # Scale, split, label (eager for group_map/scikit)
        scaled = scale_per_token(with_volume)
        del with_volume  # Free memory immediately
        log_memory_usage(f"after scaling {interval}")
        
        split_df = add_splits_and_labels(scaled)
        del scaled  # Free memory immediately
        log_memory_usage(f"after splits/labels {interval}")
        
        # Clustering (early + full)
        clustering_start = time.time()
        early_clustered = fast_early_clustering(split_df)
        log_memory_usage(f"after early clustering {interval}")
        
        full_clustered = cluster_archetypes(early_clustered)
        del early_clustered  # Free memory immediately
        del split_df  # Free memory immediately
        log_memory_usage(f"after full clustering {interval}")
        
        logger.info(f"Clustering took {time.time() - clustering_start:.2f}s | Final shape: {full_clustered.shape}")
        
        # Save immediately and free memory
        output_path = f"{output_base}_{interval}.parquet"
        full_clustered.write_parquet(output_path)
        logger.info(f"Processed {interval}: Saved {output_path} | Shape {full_clustered.shape}")
        
        # Delete final dataframe to free memory
        del full_clustered
        
        # Aggressive memory cleanup after each interval
        aggressive_memory_cleanup()
        
        logger.info(f"Completed {interval} processing in {time.time() - interval_start:.2f}s")
        logger.info("-" * 50)  # Visual separator between intervals
    
    log_memory_usage("pipeline end")
    logger.info(f"Total pipeline time: {time.time() - start_total:.2f}s")
    logger.info("Pipeline completed successfully with memory-efficient processing!")
    
    # Return empty dict since we're not storing processed data in memory anymore
    # Files are saved directly to disk for memory efficiency
    return {}

if __name__ == "__main__":
    run_pipeline("processed_features")