import polars as pl
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import glob

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

def run_pipeline(output_base: str) -> dict:
    """
    Full Polars pipeline for 30k+ tokens (<1min run): List all Parquet files in data/raw/dataset/, check schema to exclude '_tokens_list.parquet' (different columns like 'signature'), load token files, extract 'token_id' from filenames, add as column, concat DFs (same schema: datetime/price), clean (nulls/gaps/negatives), dual resample (1m/5m for comparison), add variability (after 'returns' calculation), death detection with 30 min threshold and trim constants to prevent leakage, expanded features (for ML training, vectorized with no leakage), per-token scaling (fit on train only), temporal splits (80/20 per token, no future leakage), early + full dynamic clustering.
    Outputs dict of processed Parquets (1m/5m).
    """
    # Dynamic multi-file loading: List all Parquet files
    data_dir = config.DATA_RAW_DIR
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    
    if not all_files:
        raise ValueError("No Parquet files found in data/raw/dataset/")
    
    # Load each file if schema matches 'datetime'/'price' (exclude list file)
    dfs = []
    print(f"Found {len(all_files)} total parquet files")
    
    for file_path in all_files:
        lazy_df = pl.scan_parquet(file_path)
        schema = lazy_df.collect_schema()  # Use collect_schema() to avoid performance warning
        
        # Debug: Show schema for first few files
        if len(dfs) < 3:
            print(f"File: {os.path.basename(file_path)}")
            print(f"  Schema: {dict(schema)}")
            print(f"  Columns: {list(schema.keys())}")
        
        if 'datetime' in schema and 'price' in schema and len(schema) == 2:  # Exact match for token files
            token_id = os.path.splitext(os.path.basename(file_path))[0]  # e.g., 'token_address'
            lazy_df = lazy_df.with_columns(pl.lit(token_id).alias("token_id"))
            dfs.append(lazy_df)
        else:
            print(f"Skipped {os.path.basename(file_path)} (schema mismatch or extra columns)")
    
    print(f"Successfully loaded {len(dfs)} token files out of {len(all_files)} total files")
    
    if not dfs:
        raise ValueError("No valid token Parquet files found with 'datetime'/'price' schema")
    
    df = pl.concat(dfs, how="vertical")  # Lazy concat, same schema
    
    # Cleaning: Handle nulls/negatives/gaps, then sort for group_by_dynamic
    cleaned = df.drop_nulls().filter(pl.col("price") >= 0).with_columns(pl.col("price").interpolate().alias("price"))
    sorted_data = cleaned.sort(["token_id", "datetime"])  # Required for group_by_dynamic
    
    outputs = {}
    for interval in config.RESAMPLE_INTERVALS:
        resampled = sorted_data.lazy().group_by_dynamic("datetime", every=interval, group_by="token_id").agg(
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").std().alias("volatility")
        ).with_columns(pl.col("avg_price").pct_change().alias("returns")).drop_nulls()
        
        # Variability analysis after 'returns' calculation
        analyzed = analyze_variability(resampled.collect())
        
        with_death = detect_death(analyzed, config.ZERO_THRESHOLD)  # 30 min
        trimmed = trim_post_death(with_death)  # Trim constants post-death to avoid leakage/noise
        
        # Check for empty DataFrame before feature engineering
        if trimmed.height == 0:
            print(f"Warning: Empty DataFrame for {interval} interval after death trimming")
            continue
        
        print(f"Processing {interval}: {trimmed.height} rows, {trimmed.width} columns")
        print(f"Available columns: {trimmed.columns}")
        
        featured = engineer_features(trimmed)  # Expanded features, vectorized
        with_volume = add_volume_placeholders(featured)
        
        # Per-token scaling (fit on train split only to prevent leakage)
        def scale_group(group):
            group = group.with_columns(pl.col("datetime").rank("dense").alias("rank"))
            train_size = int(0.8 * len(group))
            train_returns = group.filter(pl.col("rank") <= train_size)['returns'].to_numpy().reshape(-1, 1)
            scaler = MinMaxScaler().fit(train_returns)
            scaled = scaler.transform(group['returns'].to_numpy().reshape(-1, 1)).flatten()
            return group.with_columns(pl.Series("scaled_returns", scaled)).drop("rank")
        scaled = with_volume.group_by("token_id").map_groups(scale_group)
        
        # Full temporal split (80/20 per token)
        split_df = scaled.with_columns(pl.col("datetime").rank("dense").over("token_id").alias("rank")).with_columns(
            pl.when(pl.col("rank") <= pl.col("rank").max().over("token_id") * 0.8).then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")
        ).drop("rank")
        
        # Early + full clustering
        early_clustered = fast_early_clustering(split_df, config.EARLY_MINUTES)
        full_clustered = cluster_archetypes(early_clustered)
        
        output_path = f"{output_base}_{interval}.parquet"
        full_clustered.write_parquet(output_path)
        outputs[interval] = full_clustered
        print(f"Processed {interval}: Shape {full_clustered.shape}")
    
    return outputs

if __name__ == "__main__":
    run_pipeline("processed_features")