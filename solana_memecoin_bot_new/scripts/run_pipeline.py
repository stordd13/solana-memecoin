# scripts/run_pipeline.py (Handled empty groups in scaling; conditional datetime cast per file; collect after resample for eager ops)
import polars as pl
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import glob
import time
import numpy as np

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
    start_total = time.time()
    
    # Dynamic multi-file loading
    data_dir = config.DATA_RAW_DIR
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    
    if not all_files:
        logger.error("No Parquet files found in data/raw/dataset/")
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
            
            # Skip metadata files like '_tokens_list*'
            if os.path.basename(file_path).startswith('_tokens_list'):
                logger.warning(f"Skipped metadata file {os.path.basename(file_path)}")
                continue
            
            if 'datetime' in schema and 'price' in schema and len(schema) == 2:
                token_id = os.path.splitext(os.path.basename(file_path))[0]
                # Conditional cast based on dtype
                if schema['datetime'] == pl.String:
                    lazy_df = lazy_df.with_columns(pl.col("datetime").str.to_datetime().alias("datetime"))
                elif schema['datetime'] != pl.Datetime:
                    logger.warning(f"Unexpected datetime dtype in {file_path}: {schema['datetime']}, attempting cast")
                    lazy_df = lazy_df.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))
                lazy_df = lazy_df.with_columns(pl.lit(token_id).alias("token_id"))
                dfs.append(lazy_df)
                logger.info(f"Loaded {os.path.basename(file_path)} in {time.time() - start:.2f}s")
            else:
                logger.warning(f"Skipped {os.path.basename(file_path)} (schema mismatch)")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Loaded {len(dfs)} token files")
    
    if not dfs:
        raise ValueError("No valid token files")
    
    start = time.time()
    df = pl.concat(dfs, how="vertical")
    logger.info(f"Concat took {time.time() - start:.2f}s")
    
    # Cleaning and sort
    start = time.time()
    cleaned = df.drop_nulls().filter(pl.col("price") >= 0).with_columns(pl.col("price").interpolate().alias("price"))
    sorted_data = cleaned.sort(["token_id", "datetime"])
    logger.debug(f"Cleaning/sort took {time.time() - start:.2f}s")
    
    outputs = {}
    for interval in config.RESAMPLE_INTERVALS:
        start = time.time()
        resampled = sorted_data.group_by_dynamic("datetime", every=interval, group_by="token_id").agg(
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").std().alias("volatility")
        ).with_columns(pl.col("avg_price").pct_change().alias("returns")).drop_nulls().collect()
        logger.info(f"Resample {interval} took {time.time() - start:.2f}s | Shape: {resampled.shape}")
        
        # Variability, death, trim
        analyzed = analyze_variability(resampled)
        with_death = detect_death(analyzed, config.ZERO_THRESHOLD)
        trimmed = trim_post_death(with_death)
        
        if trimmed.height == 0:
            logger.warning(f"Empty DF for {interval} after trim")
            continue
        
        logger.info(f"Death/trim shape: {trimmed.shape}")
        
        # Features and placeholders
        featured = engineer_features(trimmed)
        with_volume = add_volume_placeholders(featured)
        
        # Filter groups with at least 2 rows before scaling (avoid empty train)
        with_volume = with_volume.filter(pl.col("token_id").count().over("token_id") >= 2)
        
        # Per-token scaling (fit on train split only to prevent leakage)
        start = time.time()
        def scale_group(group):
            if len(group) < 2:
                logger.debug(f"Skipping small group for token {group['token_id'][0]} (len={len(group)})")
                return group.with_columns(pl.lit(0.0).alias("scaled_returns"))
            group = group.with_columns(pl.col("datetime").rank("dense").alias("rank"))
            train_size = int(0.8 * len(group))
            if train_size < 1:
                logger.debug(f"No train data for token {group['token_id'][0]}, setting scaled to 0")
                return group.with_columns(pl.lit(0.0).alias("scaled_returns")).drop("rank")
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
        split_df = split_df.with_columns(
            pl.when(pl.col("returns").shift(-1) > 0.5).then(1).otherwise(0).alias("pump_label")
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