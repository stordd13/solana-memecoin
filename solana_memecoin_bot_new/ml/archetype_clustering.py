import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import sys
import os

# Dynamic path fix: Add bot_new root to PYTHONPATH for config
ml_dir = os.path.dirname(__file__)
bot_new_root = os.path.abspath(os.path.join(ml_dir, '..'))
sys.path.append(bot_new_root)  # For config.py in root
import config

def fast_early_clustering(df: pl.DataFrame, min_minutes: int = config.EARLY_MINUTES) -> pl.DataFrame:
    """
    Lightweight clustering on first 5 min for early entry (features: returns std/mean/momentum/dump flag).
    Dynamic K; challenges noise in initial data.
    """
    early_df = df.group_by("token_id").head(min_minutes).with_columns(
        pl.col("returns").pct_change().alias("momentum")
    )
    features = early_df.group_by("token_id").agg(
        pl.col("returns").std().alias("ret_std"),
        pl.col("returns").mean().alias("ret_mean"),
        pl.col("momentum").mean().alias("avg_momentum"),
        pl.col("initial_dump_flag").any().alias("dump_flag")
    ).select(pl.exclude("token_id")).to_numpy()
    
    # Check for NaN/inf values in the early feature matrix
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("Warning: Found NaN or inf values in early feature matrix, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    best_k, best_score, best_labels = 2, -1, None
    max_clusters = min(config.N_ARCHETYPES_RANGE[1], len(features) - 1)  # Can't have more clusters than samples - 1
    for k in range(config.N_ARCHETYPES_RANGE[0], max_clusters + 1):
        if k >= len(features):  # Need at least k+1 samples for k clusters
            break
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        if len(set(labels)) > 1:  # Only calculate silhouette if we have multiple clusters
            score = silhouette_score(features, labels)
        else:
            score = -1  # Invalid score if all samples in same cluster
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    print(f"Early Clustering: Optimal K={best_k}, Score={best_score:.3f}")
    
    token_ids = early_df["token_id"].unique().to_list()
    return df.join(pl.DataFrame({"token_id": token_ids, "early_archetype": best_labels}), on="token_id", how="left")

def cluster_archetypes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full dynamic K-means on expanded features; uses early as optional seed.
    """
    feature_cols = ["scaled_returns", "ma_5", "vol_std_5", "momentum_lag1", "rsi_14", "acf_lag_1", "imbalance_ratio", "initial_dump_flag"]
    
    # Check which columns exist and filter accordingly
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns for clustering: {missing_cols}")
    
    if not available_cols:
        print("Error: No feature columns available for clustering")
        return df.with_columns(pl.lit(0).alias("archetype"))
    
    print(f"Using {len(available_cols)} features for clustering: {available_cols}")
    
    # Use only available columns and ensure we have the right number of rows
    features_df = df.select(pl.col(available_cols)).drop_nulls()
    features = features_df.to_numpy()
    
    if len(features) == 0:
        print("Error: No valid feature data for clustering")
        return df.with_columns(pl.lit(0).alias("archetype"))
    
    # Check for NaN/inf values in the feature matrix
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("Warning: Found NaN or inf values in feature matrix, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature matrix stats: min={np.min(features):.3f}, max={np.max(features):.3f}, mean={np.mean(features):.3f}")
    
    best_k, best_score, best_labels = 2, -1, None
    max_clusters = min(config.N_ARCHETYPES_RANGE[1], len(features) - 1)  # Can't have more clusters than samples - 1
    for k in range(config.N_ARCHETYPES_RANGE[0], max_clusters + 1):
        if k >= len(features):  # Need at least k+1 samples for k clusters
            break
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        if len(set(labels)) > 1:  # Only calculate silhouette if we have multiple clusters
            score = silhouette_score(features, labels)
        else:
            score = -1  # Invalid score if all samples in same cluster
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    print(f"Full Clustering: Optimal K={best_k}, Score={best_score:.3f}")
    
    # Create labels for the original dataframe
    # Since we dropped nulls, we need to map back to original indices
    original_indices = features_df.select(pl.int_range(pl.len()).alias("idx")).to_numpy().flatten()
    
    # Create a series with the same length as original df, filling with -1 for missing values
    archetype_series = pl.Series("archetype", [-1] * len(df))
    
    # Assign cluster labels to the valid indices
    for i, label in enumerate(best_labels):
        if i < len(original_indices):
            archetype_series = archetype_series.scatter(original_indices[i], label)
    
    return df.with_columns(archetype_series)