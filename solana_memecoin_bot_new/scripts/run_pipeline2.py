import polars as pl
from sklearn.preprocessing import MinMaxScaler
import sys
import os

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
    Full Polars pipeline for 30k+ tokens (<1min run): Query from centralized 'memecoin.db' (table 'token_prices' with 'token_id', 'datetime', 'price'), clean (nulls/gaps/negatives), dual resample (1m/5m for comparison), add variability (after 'returns' calculation), death detection with 30 min threshold and trim constants to prevent leakage, expanded features (for ML training, vectorized with no leakage), per-token scaling (fit on train only), temporal splits (80/20 per token, no future leakage), early + full dynamic clustering.
    Outputs dict of processed Parquets (1m/5m).
    """
    import sqlite3
    conn = sqlite3.connect(os.path.join(bot_new_root, 'memecoin.db'))
    df = pl.read_database("SELECT * FROM token_prices", conn)  # Polars query from DB
    
    # Cleaning: Handle nulls/negatives/gaps
    cleaned = df.drop_nulls().filter(pl.col("price") >= 0).with_columns(pl.col("price").interpolate().alias("price"))
    
    outputs = {}
    for interval in config.RESAMPLE_INTERVALS:
        resampled = cleaned.group_by_dynamic("datetime", every=interval, group_by="token_id").agg(
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").std().alias("volatility")
        ).with_columns(pl.col("avg_price").pct_change().alias("returns")).drop_nulls()
        
        # Variability analysis after 'returns' calculation
        analyzed = analyze_variability(resampled)
        
        with_death = detect_death(analyzed, config.ZERO_THRESHOLD)  # 30 min
        trimmed = trim_post_death(with_death)  # Trim constants post-death to avoid leakage/noise
        
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
            pl.when(pl.col("rank") <= pl.col("rank").max().over("token_id") * 0.8).then("train").otherwise("test").alias("split")
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