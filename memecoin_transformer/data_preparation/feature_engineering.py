"""
feature_engineering.py
Feature engineering pour les données de memecoins
Version corrigée avec protection contre les valeurs extrêmes
Modified to load from multiple parquet files in normal_behavior_tokens directory
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

class MemecoinsFeatureEngineer:
    """
    Feature engineering spécialisé pour les memecoins
    Transforme les données brutes (price, date_utc, token_address) 
    en features utilisables pour le ML
    """
    
    def __init__(self, df: pl.DataFrame):
        self.df = df.sort(["token_address", "date_utc"])
        # Nettoyer les prix dès le départ
        self.df = self.df.with_columns([
            pl.when(pl.col("price") <= 0).then(None).otherwise(pl.col("price")).alias("price")
        ])
        
    def add_core_features(self) -> pl.DataFrame:
        """Features essentielles calculables dès les premières minutes"""
        self.df = self.df.with_columns([
            # 1. Features temporelles basiques
            pl.col("date_utc").dt.minute().alias("minute_of_hour"),
            pl.col("date_utc").dt.hour().alias("hour_of_day"),
            
            # 2. Minutes depuis le lancement (crucial pour les memecoins)
            (pl.col("date_utc") - pl.col("date_utc").first().over("token_address"))
            .dt.total_minutes()
            .cast(pl.Int32)
            .alias("minutes_since_launch"),
            
            # 3. Prix transformé (plus stable) - avec protection
            pl.when(pl.col("price") > 0)
            .then(pl.col("price").log())
            .otherwise(None)
            .alias("log_price"),
            
            # 4. Returns instantanés - AVEC CLIPPING
            # Pct change avec protection contre division par 0 et clipping
            pl.col("price").pct_change().over("token_address")
            .clip(-0.99, 10.0)  # Max -99% ou +1000%
            .fill_null(0)
            .alias("returns"),
            
            # Log returns avec clipping
            pl.when(pl.col("price") > 0)
            .then(pl.col("price").log().diff().over("token_address"))
            .otherwise(0)
            .clip(-2.0, 2.0)  # Equivalent à prix x0.13 à x7.4
            .fill_null(0)
            .alias("log_returns"),
            
            # 5. Prix relatif au prix initial - AVEC PROTECTION
            pl.when(pl.col("price").first().over("token_address") > 0)
            .then(pl.col("price") / pl.col("price").first().over("token_address"))
            .otherwise(1.0)
            .clip(0.001, 1000.0)  # Entre 0.1% et 100,000% du prix initial
            .alias("price_multiple"),
        ])
        return self.df
    
    def add_fast_indicators(self) -> pl.DataFrame:
        """Indicateurs calculables avec peu de points"""
        self.df = self.df.with_columns([
            # Rolling windows courts (3, 5, 10 minutes max)
            pl.col("price").rolling_mean(3, min_samples=1).over("token_address").alias("ma_3"),
            pl.col("price").rolling_mean(5, min_samples=2).over("token_address").alias("ma_5"),
            pl.col("price").rolling_mean(10, min_samples=5).over("token_address").alias("ma_10"),
            
            # Écart au MA - AVEC PROTECTION
            pl.when(pl.col("price").rolling_mean(5, min_samples=2).over("token_address") > 0)
            .then(pl.col("price") / pl.col("price").rolling_mean(5, min_samples=2).over("token_address"))
            .otherwise(1.0)
            .clip(0.1, 10.0)  # Entre 10% et 1000% du MA
            .fill_null(1.0)
            .alias("price_to_ma5_ratio"),
            
            # Volatilité instantanée - TOUJOURS POSITIVE
            pl.col("log_returns").rolling_std(5, min_samples=2).over("token_address")
            .abs()  # Force positive
            .clip(0, 1.0)  # Max volatilité de 100%
            .fill_null(0)
            .alias("volatility_5m"),
            
            # Range de prix sur 5 minutes - AVEC PROTECTION
            pl.when(
                (pl.col("price").rolling_min(5, min_samples=2).over("token_address") > 0) &
                (pl.col("price").rolling_max(5, min_samples=2).over("token_address") > 0)
            )
            .then(
                pl.col("price").rolling_max(5, min_samples=2).over("token_address") / 
                pl.col("price").rolling_min(5, min_samples=2).over("token_address")
            )
            .otherwise(1.0)
            .clip(1.0, 10.0)  # Range entre 1x et 10x
            .fill_null(1.0)
            .alias("price_range_5m"),
            
            # Momentum simple - AVEC PROTECTION
            pl.when(pl.col("price").shift(5).over("token_address") > 0)
            .then(pl.col("price") / pl.col("price").shift(5).over("token_address"))
            .otherwise(1.0)
            .clip(0.01, 100.0)  # Entre 1% et 10,000%
            .fill_null(1.0)
            .alias("momentum_5m"),
        ])
        return self.df
    
    def add_pump_detection_features(self) -> pl.DataFrame:
        """Features spécifiques pour détecter les pumps/dumps"""
        self.df = self.df.with_columns([
            # Nombre de hausses consécutives
            (pl.col("returns") > 0).cast(pl.Int32)
            .rolling_sum(3, min_samples=1).over("token_address")
            .clip(0, 3)  # Max 3
            .alias("consecutive_ups_3m"),
            
            # Vitesse de changement - UTILISER LOG RETURNS DÉJÀ CLIPPÉS
            pl.col("log_returns").abs()
            .clip(0, 2.0)  # Déjà clippé mais on re-vérifie
            .rolling_mean(3, min_samples=1).over("token_address")
            .clip(0, 1.0)  # Max 100% de changement moyen
            .fill_null(0)
            .alias("avg_abs_change_3m"),
            
            # Est-ce qu'on est en pump ? (>5% en 3 minutes) - AVEC PROTECTION
            pl.when(pl.col("price").shift(3).over("token_address") > 0)
            .then(pl.col("price") / pl.col("price").shift(3).over("token_address") > 1.05)
            .otherwise(False)
            .fill_null(False)
            .cast(pl.Int32)
            .alias("is_pumping"),
            
            # Est-ce qu'on est en dump ? (<-5% en 3 minutes) - AVEC PROTECTION
            pl.when(pl.col("price").shift(3).over("token_address") > 0)
            .then(pl.col("price") / pl.col("price").shift(3).over("token_address") < 0.95)
            .otherwise(False)
            .fill_null(False)
            .cast(pl.Int32)
            .alias("is_dumping"),
        ])
        return self.df
    
    def add_multi_step_targets(self, forecast_steps: int = 5) -> pl.DataFrame:
        """
        Ajoute les targets pour prédire les N prochains points
        forecast_steps: nombre de minutes à prédire dans le futur
        """
        df = self.df
        
        # Pour chaque step futur
        for step in range(1, forecast_steps + 1):
            df = df.with_columns([
                # Prix futur à chaque step
                pl.col("price").shift(-step).over("token_address")
                .alias(f"target_price_t{step}"),
                
                # Log prix (plus stable pour la prédiction)
                pl.col("log_price").shift(-step).over("token_address")
                .alias(f"target_log_price_t{step}"),
                
                # Return cumulé depuis maintenant jusqu'à ce step - AVEC CLIPPING
                pl.when(
                    (pl.col("price") > 0) & 
                    (pl.col("price").shift(-step).over("token_address") > 0)
                )
                .then(
                    ((pl.col("price").shift(-step).over("token_address") / pl.col("price")) - 1)
                    .clip(-0.99, 10.0)  # Entre -99% et +1000%
                )
                .otherwise(0)
                .alias(f"target_return_t{step}"),
                
                # Direction binaire à chaque step
                (pl.col("price").shift(-step) > pl.col("price"))
                .over("token_address")
                .fill_null(False)
                .cast(pl.Int32)
                .alias(f"target_direction_t{step}"),
            ])
        
        # Ajouter des métriques agrégées sur la séquence future
        df = df.with_columns([
            # Prix max dans les N prochaines minutes
            pl.concat_list([
                pl.col(f"target_price_t{i}") for i in range(1, forecast_steps + 1)
            ]).list.max().alias(f"target_max_price_next_{forecast_steps}m"),
            
            # Prix min dans les N prochaines minutes
            pl.concat_list([
                pl.col(f"target_price_t{i}") for i in range(1, forecast_steps + 1)
            ]).list.min().alias(f"target_min_price_next_{forecast_steps}m"),
            
            # Y a-t-il un pump dans les N prochaines minutes? (>5%)
            pl.concat_list([
                pl.col(f"target_return_t{i}") > 0.05 for i in range(1, forecast_steps + 1)
            ]).list.any().cast(pl.Int32).alias(f"target_has_pump_next_{forecast_steps}m"),
            
            # Y a-t-il un dump dans les N prochaines minutes? (<-5%)
            pl.concat_list([
                pl.col(f"target_return_t{i}") < -0.05 for i in range(1, forecast_steps + 1)
            ]).list.any().cast(pl.Int32).alias(f"target_has_dump_next_{forecast_steps}m"),
        ])
        
        self.df = df
        return self.df
    
    def validate_features(self) -> None:
        """Valide que toutes les features ont des valeurs raisonnables"""
        print("\n🔍 Validation des features...")
        
        feature_cols = [
            "returns", "log_returns", "price_multiple", "price_to_ma5_ratio",
            "volatility_5m", "momentum_5m", "avg_abs_change_3m"
        ]
        
        for col in feature_cols:
            if col in self.df.columns:
                col_stats = self.df.select(
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                ).to_dicts()[0]
                
                print(f"{col:20s}: min={col_stats['min']:8.2f}, max={col_stats['max']:8.2f}, "
                      f"mean={col_stats['mean']:8.2f}, std={col_stats['std']:8.2f}")
                
                # Alertes
                if col == "volatility_5m" and col_stats['min'] < 0:
                    print(f"  ⚠️  ERREUR: Volatilité négative détectée!")
                elif col in ["returns", "log_returns"] and (col_stats['max'] > 100 or col_stats['min'] < -1):
                    print(f"  ⚠️  ERREUR: Returns extrêmes détectés!")
    
    def create_all_features(self, forecast_steps: int = 5) -> pl.DataFrame:
        """Pipeline complet de feature engineering"""
        print("1. Ajout des features de base...")
        self.add_core_features()
        
        print("2. Ajout des indicateurs techniques rapides...")
        self.add_fast_indicators()
        
        print("3. Ajout des features de détection pump/dump...")
        self.add_pump_detection_features()
        
        print("4. Ajout des targets multi-step...")
        self.add_multi_step_targets(forecast_steps)
        
        # Validation
        self.validate_features()
        
        print(f"\n✅ Feature engineering terminé: {len(self.df.columns)} colonnes")
        return self.df


def create_sequences_from_features(
    df: pl.DataFrame,
    sequence_length: int = 15,
    forecast_steps: int = 5,
    min_minutes_since_launch: int = 15
) -> Dict:
    """
    Crée des séquences à partir du DataFrame avec features
    Cette fonction est séparée de la classe pour plus de flexibilité
    """
    
    # Features pour l'input
    feature_cols = [
        "minutes_since_launch",
        "log_price",
        "returns",
        "price_multiple",
        "price_to_ma5_ratio",
        "volatility_5m",
        "momentum_5m",
        "is_pumping",
        "is_dumping",
        "avg_abs_change_3m"
    ]
    
    # Colonnes des targets
    target_price_cols = [f"target_log_price_t{i}" for i in range(1, forecast_steps + 1)]
    target_return_cols = [f"target_return_t{i}" for i in range(1, forecast_steps + 1)]
    target_direction_cols = [f"target_direction_t{i}" for i in range(1, forecast_steps + 1)]
    
    # S'assurer qu'on a pas de NaN dans les colonnes critiques
    check_cols = feature_cols + target_price_cols
    df_clean = df.drop_nulls(subset=check_cols)
    
    # Filtrer les valeurs aberrantes avant de créer les séquences
    df_clean = df_clean.filter(
        (pl.col("returns").abs() <= 10) &  # Max 1000% return
        (pl.col("volatility_5m") >= 0) &   # Volatilité positive
        (pl.col("volatility_5m") <= 1) &   # Max 100% volatilité
        (pl.col("price_multiple") > 0) &   # Ratio positif
        (pl.col("price_multiple") <= 1000) # Max 100,000x
    )
    
    sequences = []
    target_sequences = []
    target_returns = []
    target_directions = []
    metadata = []
    
    print("\nCréation des séquences par token...")
    n_tokens_processed = 0
    n_sequences_total = 0
    
    for token, group in df_clean.group_by("token_address"):
        group = group.sort("date_utc")
        
        # Ne pas utiliser les toutes premières minutes
        mask = group["minutes_since_launch"] >= min_minutes_since_launch
        group = group.filter(mask)
        
        if len(group) < sequence_length + forecast_steps:
            continue
        
        # Extraire toutes les données nécessaires
        features = group.select(feature_cols).to_numpy()
        
        # Vérifier encore une fois les valeurs
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"  ⚠️  Skipping token {token[0]} due to NaN/Inf values")
            continue
        
        # Targets : séquences futures complètes
        future_prices = group.select(target_price_cols).to_numpy()
        future_returns = group.select(target_return_cols).to_numpy()
        future_directions = group.select(target_direction_cols).to_numpy()
        
        # Métriques agrégées
        has_pump = group[f"target_has_pump_next_{forecast_steps}m"].to_numpy()
        has_dump = group[f"target_has_dump_next_{forecast_steps}m"].to_numpy()
        
        # Créer des séquences
        n_sequences_token = 0
        for i in range(sequence_length, len(features) - forecast_steps):
            # Vérifier que la séquence n'a pas de valeurs aberrantes
            seq = features[i-sequence_length:i]
            if np.any(np.abs(seq[:, 2]) > 10):  # returns column
                continue
                
            sequences.append(seq)
            target_sequences.append(future_prices[i])
            target_returns.append(future_returns[i])
            target_directions.append(future_directions[i])
            
            metadata.append({
                "token": token[0],
                "timestamp": group["date_utc"][i],
                "minutes_since_launch": int(group["minutes_since_launch"][i]),
                "current_price": float(group["price"][i]),
                "has_pump_next": int(has_pump[i]),
                "has_dump_next": int(has_dump[i])
            })
            n_sequences_token += 1
        
        n_tokens_processed += 1
        n_sequences_total += n_sequences_token
        
        if n_tokens_processed % 100 == 0:
            print(f"  Processed {n_tokens_processed} tokens, {n_sequences_total} sequences...")
    
    print(f"\n✅ Séquences créées: {n_sequences_total} séquences from {n_tokens_processed} tokens")
    
    # Validation finale
    sequences_array = np.array(sequences, dtype=np.float32)
    print(f"\n🔍 Validation finale des séquences:")
    for i, feat_name in enumerate(feature_cols):
        feat_values = sequences_array[:, :, i].flatten()
        print(f"{feat_name:20s}: min={feat_values.min():8.2f}, max={feat_values.max():8.2f}, "
              f"mean={feat_values.mean():8.2f}")
    
    return {
        "input_sequences": sequences_array,
        "target_sequences": np.array(target_sequences, dtype=np.float32),
        "target_returns": np.array(target_returns, dtype=np.float32),
        "target_directions": np.array(target_directions, dtype=np.int32),
        "metadata": metadata,
        "feature_names": feature_cols,
        "sequence_length": sequence_length,
        "forecast_steps": forecast_steps,
        "n_features": len(feature_cols)
    }


def load_tokens_from_directory(directory_path: Path, max_tokens: int = None) -> pl.DataFrame:
    """
    Load all parquet files from a directory and combine them into a single DataFrame
    Each file represents a single token's price data
    
    Args:
        directory_path: Path to directory containing parquet files
        max_tokens: Maximum number of tokens to load (None for all)
    
    Returns:
        Combined DataFrame with all tokens
    """
    print(f"📂 Loading tokens from: {directory_path}")
    
    # Get all parquet files
    parquet_files = list(directory_path.glob("*.parquet"))
    
    if max_tokens:
        parquet_files = parquet_files[:max_tokens]
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    all_dfs = []
    
    for file_path in tqdm(parquet_files, desc="Loading tokens"):
        # Extract token address from filename (remove .parquet extension)
        token_address = file_path.stem
        
        # Read the parquet file
        df = pl.read_parquet(file_path)
        
        # Add token_address column
        df = df.with_columns([
            pl.lit(token_address).alias("token_address")
        ])
        
        # Rename datetime to date_utc if needed
        if "datetime" in df.columns:
            df = df.rename({"datetime": "date_utc"})
        
        all_dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pl.concat(all_dfs)
    
    print(f"✅ Loaded {combined_df['token_address'].n_unique()} tokens with {len(combined_df)} total rows")
    
    return combined_df


if __name__ == "__main__":
    # New data loading from multiple parquet files
    data_dir = Path("/Users/stordd/doc/Solana/memecoin2/data/processed/normal_behavior_tokens")
    
    # Load all tokens from the directory
    print("🚀 Starting feature engineering pipeline...")
    print("=" * 60)
    
    # You can limit the number of tokens for testing by setting max_tokens
    # df = load_tokens_from_directory(data_dir, max_tokens=100)  # For testing
    df = load_tokens_from_directory(data_dir)  # For all tokens
    
    print(f"\n📊 Data summary:")
    print(f"  - Total rows: {len(df):,}")
    print(f"  - Unique tokens: {df['token_address'].n_unique():,}")
    print(f"  - Date range: {df['date_utc'].min()} to {df['date_utc'].max()}")
    print(f"  - Price range: {df['price'].min():.8f} to {df['price'].max():.8f}")
    
    # Feature engineering
    print("\n🔧 Starting feature engineering...")
    fe = MemecoinsFeatureEngineer(df)
    df_features = fe.create_all_features(forecast_steps=5)
    
    # Save features to new location
    output_path = Path("/Users/stordd/doc/Solana/memecoin2/data/processed/memecoin_features_from_normal_tokens.parquet")
    df_features.write_parquet(output_path)
    print(f"\n💾 Features saved to: {output_path}")
    
    # Create sequences
    print("\n📦 Creating sequences...")
    sequences_data = create_sequences_from_features(
        df_features,
        sequence_length=15,
        forecast_steps=5,
        min_minutes_since_launch=15
    )
    
    print(f"\n✅ Final summary:")
    print(f"  - Number of sequences: {len(sequences_data['input_sequences']):,}")
    print(f"  - Input shape: {sequences_data['input_sequences'].shape}")
    print(f"  - Target shape: {sequences_data['target_sequences'].shape}")
    
    # Save sequences to new location
    sequences_path = Path("/Users/stordd/doc/Solana/memecoin2/data/processed/sequences_from_normal_tokens.npz")
    np.savez_compressed(sequences_path, **sequences_data)
    print(f"💾 Sequences saved to: {sequences_path}")