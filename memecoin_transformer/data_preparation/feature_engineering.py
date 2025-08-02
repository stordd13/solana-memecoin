"""
feature_engineering.py
Feature engineering pour les données de memecoins
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class MemecoinsFeatureEngineer:
    """
    Feature engineering spécialisé pour les memecoins
    Transforme les données brutes (price, date_utc, token_address) 
    en features utilisables pour le ML
    """
    
    def __init__(self, df: pl.DataFrame):
        self.df = df.sort(["token_address", "date_utc"])
        
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
            
            # 3. Prix transformé (plus stable)
            pl.col("price").log().alias("log_price"),
            
            # 4. Returns instantanés
            pl.col("price").pct_change().over("token_address").fill_null(0).alias("returns"),
            pl.col("price").log().diff().over("token_address").fill_null(0).alias("log_returns"),
            
            # 5. Prix relatif au prix initial
            (pl.col("price") / pl.col("price").first().over("token_address")).alias("price_multiple"),
        ])
        return self.df
    
    def add_fast_indicators(self) -> pl.DataFrame:
        """Indicateurs calculables avec peu de points"""
        self.df = self.df.with_columns([
            # Rolling windows courts (3, 5, 10 minutes max)
            pl.col("price").rolling_mean(3, min_periods=1).over("token_address").alias("ma_3"),
            pl.col("price").rolling_mean(5, min_periods=2).over("token_address").alias("ma_5"),
            pl.col("price").rolling_mean(10, min_periods=5).over("token_address").alias("ma_10"),
            
            # Écart au MA
            (pl.col("price") / pl.col("price").rolling_mean(5, min_periods=2).over("token_address"))
            .fill_null(1.0).alias("price_to_ma5_ratio"),
            
            # Volatilité instantanée (rolling sur 5 minutes)
            pl.col("log_returns").rolling_std(5, min_periods=2).over("token_address")
            .fill_null(0).alias("volatility_5m"),
            
            # Range de prix sur 5 minutes
            (pl.col("price").rolling_max(5, min_periods=2).over("token_address") / 
             pl.col("price").rolling_min(5, min_periods=2).over("token_address"))
            .fill_null(1.0).alias("price_range_5m"),
            
            # Momentum simple
            (pl.col("price") / pl.col("price").shift(5).over("token_address"))
            .fill_null(1.0).alias("momentum_5m"),
        ])
        return self.df
    
    def add_pump_detection_features(self) -> pl.DataFrame:
        """Features spécifiques pour détecter les pumps/dumps"""
        self.df = self.df.with_columns([
            # Nombre de hausses consécutives
            (pl.col("returns") > 0).cast(pl.Int32)
            .rolling_sum(3, min_periods=1).over("token_address")
            .alias("consecutive_ups_3m"),
            
            # Vitesse de changement
            pl.col("log_returns").abs()
            .rolling_mean(3, min_periods=1).over("token_address")
            .alias("avg_abs_change_3m"),
            
            # Est-ce qu'on est en pump ? (>5% en 3 minutes)
            (pl.col("price") / pl.col("price").shift(3).over("token_address") > 1.05)
            .fill_null(False).cast(pl.Int32)
            .alias("is_pumping"),
            
            # Est-ce qu'on est en dump ? (<-5% en 3 minutes)
            (pl.col("price") / pl.col("price").shift(3).over("token_address") < 0.95)
            .fill_null(False).cast(pl.Int32)
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
                
                # Return cumulé depuis maintenant jusqu'à ce step
                ((pl.col("price").shift(-step) / pl.col("price")) - 1)
                .over("token_address")
                .alias(f"target_return_t{step}"),
                
                # Direction binaire à chaque step
                (pl.col("price").shift(-step) > pl.col("price"))
                .over("token_address")
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
        
        print(f"✅ Feature engineering terminé: {len(self.df.columns)} colonnes")
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
    
    sequences = []
    target_sequences = []
    target_returns = []
    target_directions = []
    metadata = []
    
    for token, group in df_clean.group_by("token_address"):
        group = group.sort("date_utc")
        
        # Ne pas utiliser les toutes premières minutes
        mask = group["minutes_since_launch"] >= min_minutes_since_launch
        group = group.filter(mask)
        
        if len(group) < sequence_length + forecast_steps:
            continue
        
        # Extraire toutes les données nécessaires
        features = group.select(feature_cols).to_numpy()
        
        # Targets : séquences futures complètes
        future_prices = group.select(target_price_cols).to_numpy()
        future_returns = group.select(target_return_cols).to_numpy()
        future_directions = group.select(target_direction_cols).to_numpy()
        
        # Métriques agrégées
        has_pump = group[f"target_has_pump_next_{forecast_steps}m"].to_numpy()
        has_dump = group[f"target_has_dump_next_{forecast_steps}m"].to_numpy()
        
        # Créer des séquences
        for i in range(sequence_length, len(features) - forecast_steps):
            sequences.append(features[i-sequence_length:i])
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
    
    return {
        "input_sequences": np.array(sequences, dtype=np.float32),
        "target_sequences": np.array(target_sequences, dtype=np.float32),
        "target_returns": np.array(target_returns, dtype=np.float32),
        "target_directions": np.array(target_directions, dtype=np.int32),
        "metadata": metadata,
        "feature_names": feature_cols,
        "sequence_length": sequence_length,
        "forecast_steps": forecast_steps,
        "n_features": len(feature_cols)
    }


if __name__ == "__main__":
    # Exemple d'utilisation
    data_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/data_onchain_filtered_high_score_tokens.parquet")
    df = pl.read_parquet(data_path)
    
    print(f"Données chargées : {len(df)} lignes, {df['token_address'].n_unique()} tokens")
    
    # Feature engineering
    fe = MemecoinsFeatureEngineer(df)
    df_features = fe.create_all_features(forecast_steps=5)
    
    # Sauvegarder le DataFrame avec features
    output_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/memecoin_features_complete.parquet")
    df_features.write_parquet(output_path)
    print(f"\n💾 Features sauvegardées : {output_path}")
    
    # Créer les séquences
    print("\nCréation des séquences...")
    sequences_data = create_sequences_from_features(
        df_features,
        sequence_length=15,
        forecast_steps=5,
        min_minutes_since_launch=15
    )
    
    print(f"\n✅ Séquences créées :")
    print(f"  - Nombre : {len(sequences_data['input_sequences'])}")
    print(f"  - Shape input : {sequences_data['input_sequences'].shape}")
    print(f"  - Shape target : {sequences_data['target_sequences'].shape}")
    
    # Sauvegarder les séquences
    sequences_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/sequences_raw.npz")
    np.savez_compressed(sequences_path, **sequences_data)
    print(f"💾 Séquences sauvegardées : {sequences_path}")