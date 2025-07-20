import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os
import time

# Dynamic path fix: Add bot_new root to PYTHONPATH for config
ml_dir = os.path.dirname(__file__)
bot_new_root = os.path.abspath(os.path.join(ml_dir, '..'))
sys.path.append(bot_new_root)  # For config.py in root
import config
from utils import setup_logger

logger = setup_logger(__name__)

def select_best_k(features, min_k=2, max_k=10):
    silhouette_scores = []
    calinski_scores = []
    inertia = []
    labels_list = []
    for k in range(min_k, max_k + 1):
        if k >= len(features): break
        k_start = time.time()
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Increased n_init for better init
        labels = kmeans.fit_predict(features)
        labels_list.append(labels)
        if len(set(labels)) > 1:
            sil = silhouette_score(features, labels, sample_size=10000)
            cal = calinski_harabasz_score(features, labels)
        else:
            sil = cal = -1
            logger.debug(f"Single cluster at k={k}, skipping scores")
        silhouette_scores.append(sil)
        calinski_scores.append(cal)
        inertia.append(kmeans.inertia_)
        logger.debug(f"k={k}: Silhouette={sil:.3f}, Calinski={cal:.3f}, Inertia={inertia[-1]:.3f}, Time={time.time() - k_start:.2f}s")
    
    # Select best K: Max silhouette/Calinski, min elbow change
    best_k_sil = np.argmax(silhouette_scores) + min_k if max(silhouette_scores) > 0 else min_k
    best_k_cal = np.argmax(calinski_scores) + min_k if max(calinski_scores) > 0 else min_k
    elbow_k = min_k + np.argmin(np.diff(np.diff(inertia))) + 1 if len(inertia) > 2 else min_k  # Second derivative min
    # Vote: Mode or average; fallback to elbow if ties
    votes = [best_k_sil, best_k_cal, elbow_k]
    best_k = max(set(votes), key=votes.count)
    logger.info(f"Best K: Silhouette={best_k_sil}, Calinski={best_k_cal}, Elbow={elbow_k}, Selected={best_k}")
    return best_k, silhouette_scores[best_k - min_k], labels_list[best_k - min_k]

def fast_early_clustering(df: pl.DataFrame, min_minutes: int = config.EARLY_MINUTES) -> pl.DataFrame:
    """
    Lightweight clustering on first 5 min for early entry (features: returns std/mean/momentum/dump flag).
    Dynamic K with multi-metric; challenges noise in initial data.
    """
    start = time.time()
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
        logger.warning("Found NaN or inf values in early feature matrix, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features to [0,1] to handle extremes
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    best_k, best_score, best_labels = select_best_k(features, config.N_ARCHETYPES_RANGE[0], config.N_ARCHETYPES_RANGE[1])
    logger.info(f"Early Clustering: Optimal K={best_k}, Score={best_score:.3f}, Time={time.time() - start:.2f}s")
    
    token_ids = early_df["token_id"].unique().to_list()
    return df.join(pl.DataFrame({"token_id": token_ids, "early_archetype": best_labels}), on="token_id", how="left")

def cluster_archetypes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full dynamic K-means on expanded features; uses early as optional seed.
    """
    start = time.time()
    feature_cols = ["scaled_returns", "ma_5", "vol_std_5", "momentum_lag1", "rsi_14", "acf_lag_1", "imbalance_ratio", "initial_dump_flag"]
    
    # Check which columns exist and filter accordingly
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns for clustering: {missing_cols}")
    
    if not available_cols:
        logger.error("No feature columns available for clustering")
        return df.with_columns(pl.lit(0).alias("archetype"))
    
    logger.info(f"Using {len(available_cols)} features for clustering: {available_cols}")
    
    # Use only available columns and ensure we have the right number of rows
    features_df = df.select(pl.col(available_cols)).drop_nulls()
    features = features_df.to_numpy()
    
    if len(features) == 0:
        logger.error("No valid feature data for clustering")
        return df.with_columns(pl.lit(0).alias("archetype"))
    
    # Check for NaN/inf values in the feature matrix
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Found NaN or inf values in feature matrix, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features to [0,1] to handle extremes
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Feature matrix stats: min={np.min(features):.3f}, max={np.max(features):.3f}, mean={np.mean(features):.3f}")
    
    best_k, best_score, best_labels = select_best_k(features, config.N_ARCHETYPES_RANGE[0], config.N_ARCHETYPES_RANGE[1])
    logger.info(f"Full Clustering: Optimal K={best_k}, Score={best_score:.3f}, Time={time.time() - start:.2f}s")
    
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