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
        kmeans = KMeans(n_clusters=k, random_state=42)
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
    start = time.time()
    
    # Take only first min_minutes (10 by default) of each token
    early_df = df.group_by("token_id").head(min_minutes)
    
    # Use early features that work well with 10-minute windows
    early_feature_cols = [
        "early_mean_returns", "early_max_return", "early_min_return", "early_return_volatility",
        "early_price_range", "early_avg_volatility", "early_max_volatility",
        "rolling_mean_6min", "rolling_std_6min", "volatility_trend_6min",
        "rolling_mean_7min", "momentum_7min", "early_dump_flag", "early_stability_ratio"
    ]
    
    # Check which early features are available in the dataframe
    available_early_cols = [col for col in early_feature_cols if col in early_df.columns]
    
    if not available_early_cols:
        # Fallback to basic features if early features not available
        logger.warning("Early features not found, using basic aggregations")
        agg_df = early_df.group_by("token_id").agg([
            pl.col("returns").std().alias("ret_std"),
            pl.col("returns").mean().alias("ret_mean"),
            pl.col("returns").pct_change().mean().alias("avg_momentum"),
            pl.col("initial_dump_flag").any().alias("dump_flag")
        ]).fill_nan(0.0)
        features = agg_df.select(pl.exclude("token_id")).to_numpy()
    else:
        # Use early features with aggregation (take last value per token for each feature)
        agg_df = early_df.group_by("token_id").agg([
            pl.col(col).last().alias(f"{col}_last") if col not in ["early_dump_flag"] 
            else pl.col(col).any().alias(f"{col}_any") for col in available_early_cols
        ]).fill_nan(0.0)
        features = agg_df.select(pl.exclude("token_id")).to_numpy()
    
    # Clean any remaining NaN/inf values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Found NaN or inf values in early feature matrix, cleaning...")
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    best_k, best_score, best_labels = select_best_k(features, config.N_ARCHETYPES_RANGE[0], config.N_ARCHETYPES_RANGE[1])
    logger.info(f"Early Clustering: Optimal K={best_k}, Score={best_score:.3f}, Time={time.time() - start:.2f}s")
    
    return df.join(agg_df.with_columns(pl.Series("early_archetype", best_labels)).select("token_id", "early_archetype"), on="token_id", how="left")

def cluster_archetypes(df: pl.DataFrame) -> pl.DataFrame:
    start = time.time()
    # Updated to use dual-track scaling features: volatility metrics for clustering, raw returns preserved for analysis
    feature_cols = ["scaled_volatility", "scaled_vol_metrics", "ma_5", "vol_std_5", "momentum_lag1", "rsi_14", "acf_lag_1", "imbalance_ratio", "initial_dump_flag"]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    if not available_cols:
        logger.error("No feature columns available for clustering")
        return df.with_columns(pl.lit(0).alias("archetype"))
    
    logger.info(f"Using {len(available_cols)} features for clustering: {available_cols}")
    
    # Per-token aggregation with fill_nan to keep short/dead tokens
    agg_df = df.group_by("token_id").agg([pl.col(col).mean().alias(col) for col in available_cols]).fill_nan(0.0)
    features = agg_df.select(available_cols).to_numpy()
    
    if len(features) == 0:
        logger.error("No valid feature data for clustering")
        return df.with_columns(pl.lit(0).alias("archetype"))
    
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Found NaN or inf values in feature matrix, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Feature matrix stats: min={np.min(features):.3f}, max={np.max(features):.3f}, mean={np.mean(features):.3f}")
    
    best_k, best_score, best_labels = select_best_k(features, config.N_ARCHETYPES_RANGE[0], config.N_ARCHETYPES_RANGE[1])
    logger.info(f"Full Clustering: Optimal K={best_k}, Score={best_score:.3f}, Time={time.time() - start:.2f}s")
    
    return df.join(agg_df.with_columns(pl.Series("archetype", best_labels)).select("token_id", "archetype"), on="token_id", how="left")