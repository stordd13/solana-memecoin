"""
data_scaling.py - Version corrigée
Scaling et préparation des données pour l'entraînement
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import json

class MemecoinSequenceScaler:
    """
    Scaler spécialisé pour les séquences de trading de memecoins
    Applique différentes stratégies de scaling selon le type de feature
    """
    
    def __init__(self, scaling_method: str = "robust"):
        self.scaling_method = scaling_method
        self.feature_names = None
        self.global_stats = {}
        self.feature_groups = {
            "price_features": ["log_price"],
            "return_features": ["returns", "log_returns", "volatility_5m", "avg_abs_change_3m"],
            "ratio_features": ["price_multiple", "price_to_ma5_ratio", "momentum_5m", "price_range_5m"],
            "temporal_features": ["minutes_since_launch", "hour_of_day", "minute_of_hour"],
            "binary_features": ["is_pumping", "is_dumping", "consecutive_ups_3m"]
        }
        
    def fit_global_stats(self, all_sequences: np.ndarray, feature_names: List[str]):
        """
        Calcule les statistiques globales pour certaines features
        """
        self.feature_names = feature_names
        
        # Pour chaque type de feature, calculer les stats appropriées
        for feat_type, feat_list in self.feature_groups.items():
            for feat_name in feat_list:
                if feat_name in feature_names:
                    idx = feature_names.index(feat_name)
                    all_values = all_sequences[:, :, idx].flatten()
                    
                    # Filtrer les NaN
                    valid_values = all_values[~np.isnan(all_values)]
                    
                    if len(valid_values) > 0:
                        self.global_stats[feat_name] = {
                            "min": float(np.min(valid_values)),
                            "max": float(np.max(valid_values)),
                            "mean": float(np.mean(valid_values)),
                            "std": float(np.std(valid_values)),
                            "median": float(np.median(valid_values)),
                            "q1": float(np.percentile(valid_values, 25)),
                            "q3": float(np.percentile(valid_values, 75))
                        }
        
        print(f"✅ Stats globales calculées pour {len(self.global_stats)} features")
    
    def scale_sequences(
        self, 
        sequences: np.ndarray, 
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Scale toutes les séquences en une fois (vectorisé)
        """
        n_samples, seq_length, n_features = sequences.shape
        scaled_sequences = np.zeros_like(sequences)
        scaled_targets = np.zeros_like(targets)
        
        # 1. Prix - Scaling par rapport au dernier point
        if "log_price" in self.feature_names:
            idx = self.feature_names.index("log_price")
            # Extraire tous les derniers prix en une fois
            last_prices = sequences[:, -1, idx]  # Shape: (n_samples,)
            # Broadcaster pour soustraire de chaque timestep
            scaled_sequences[:, :, idx] = sequences[:, :, idx] - last_prices[:, np.newaxis]
            # Scaler les targets aussi
            scaled_targets = targets - last_prices[:, np.newaxis]
        
        # 2. Returns et volatilité - Calcul vectorisé des stats
        for feat_name in self.feature_groups["return_features"]:
            if feat_name in self.feature_names:
                idx = self.feature_names.index(feat_name)
                feat_data = sequences[:, :, idx]  # Shape: (n_samples, seq_length)
                
                if self.scaling_method == "robust":
                    # Calcul vectorisé des percentiles par séquence
                    q1 = np.nanpercentile(feat_data, 25, axis=1, keepdims=True)
                    q3 = np.nanpercentile(feat_data, 75, axis=1, keepdims=True)
                    median = np.nanmedian(feat_data, axis=1, keepdims=True)
                    iqr = q3 - q1
                    
                    # Éviter division par zéro
                    iqr = np.where(iqr > 1e-8, iqr, self.global_stats.get(feat_name, {}).get("std", 1.0))
                    
                    scaled_sequences[:, :, idx] = (feat_data - median) / iqr
                else:
                    # Standard scaling vectorisé
                    mean = np.nanmean(feat_data, axis=1, keepdims=True)
                    std = np.nanstd(feat_data, axis=1, keepdims=True)
                    std = np.where(std > 1e-8, std, 1.0)
                    scaled_sequences[:, :, idx] = (feat_data - mean) / std
        
        # 3. Ratios - Log transform vectorisé
        for feat_name in self.feature_groups["ratio_features"]:
            if feat_name in self.feature_names:
                idx = self.feature_names.index(feat_name)
                scaled_sequences[:, :, idx] = np.log(sequences[:, :, idx] + 1e-8)
        
        # 4. Features temporelles - Normalisation globale vectorisée
        for feat_name in self.feature_groups["temporal_features"]:
            if feat_name in self.feature_names:
                idx = self.feature_names.index(feat_name)
                if feat_name == "minutes_since_launch" and feat_name in self.global_stats:
                    stats = self.global_stats[feat_name]
                    scaled_sequences[:, :, idx] = (sequences[:, :, idx] - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
                elif feat_name == "hour_of_day":
                    scaled_sequences[:, :, idx] = sequences[:, :, idx] / 24.0
                elif feat_name == "minute_of_hour":
                    scaled_sequences[:, :, idx] = sequences[:, :, idx] / 60.0
        
        # 5. Features binaires - Copie directe
        for feat_name in self.feature_groups["binary_features"]:
            if feat_name in self.feature_names:
                idx = self.feature_names.index(feat_name)
                scaled_sequences[:, :, idx] = sequences[:, :, idx]
        
        # Remplacer NaN
        scaled_sequences = np.nan_to_num(scaled_sequences, nan=0.0)
        scaled_targets = np.nan_to_num(scaled_targets, nan=0.0)
        
        # Créer les params (simplifié pour la version batch)
        scaling_params = [{
            "last_log_price": float(last_prices[i]) if "log_price" in self.feature_names else 0.0
        } for i in range(n_samples)]
        
        return scaled_sequences, scaled_targets, scaling_params
    
    def inverse_scale_predictions(
        self, 
        predictions: np.ndarray, 
        scaling_params: Dict
    ) -> np.ndarray:
        """
        Inverse le scaling pour les prédictions
        """
        if "last_log_price" in scaling_params:
            # Les prédictions sont des différences par rapport au dernier log_price
            return predictions + scaling_params["last_log_price"]
        return predictions
    
    def save_scaler(self, path: Path):
        """Sauvegarde les paramètres du scaler"""
        scaler_data = {
            "scaling_method": self.scaling_method,
            "feature_names": self.feature_names,
            "global_stats": self.global_stats,
            "feature_groups": self.feature_groups
        }
        with open(path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
    
    def load_scaler(self, path: Path):
        """Charge les paramètres du scaler"""
        with open(path, 'r') as f:
            scaler_data = json.load(f)
        self.scaling_method = scaler_data["scaling_method"]
        self.feature_names = scaler_data["feature_names"]
        self.global_stats = scaler_data["global_stats"]
        self.feature_groups = scaler_data["feature_groups"]


class DataPreparer:
    """
    Prépare les données pour l'entraînement avec train/val split et scaling
    """
    
    def __init__(self, sequences_path: Path):
        # Charger les séquences avec allow_pickle=True pour les metadata
        data = np.load(sequences_path, allow_pickle=True)
        
        # Convertir en dictionnaire normal
        self.sequences_data = {}
        for key in data.files:
            self.sequences_data[key] = data[key]
            
        # Convertir les types si nécessaire
        if "feature_names" in self.sequences_data and isinstance(self.sequences_data["feature_names"], np.ndarray):
            if self.sequences_data["feature_names"].ndim == 0:
                # Cas où c'est un scalar array contenant une liste
                self.sequences_data["feature_names"] = self.sequences_data["feature_names"].item()
            else:
                self.sequences_data["feature_names"] = self.sequences_data["feature_names"].tolist()
                
        # Convertir les autres champs scalaires si nécessaire
        for key in ["sequence_length", "forecast_steps", "n_features"]:
            if key in self.sequences_data and isinstance(self.sequences_data[key], np.ndarray):
                self.sequences_data[key] = int(self.sequences_data[key].item())
        
        self.scaler = MemecoinSequenceScaler(scaling_method="robust")
        
    def prepare_for_training(
        self, 
        validation_split: float = 0.2,
        random_seed: int = 42
    ) -> Dict:
        """
        Prépare les données complètes pour l'entraînement
        """
        np.random.seed(random_seed)
        
        # Extraire les données
        input_sequences = self.sequences_data["input_sequences"]
        target_sequences = self.sequences_data["target_sequences"]
        metadata = self.sequences_data["metadata"]
        
        # Gérer les feature_names selon le format
        if isinstance(self.sequences_data["feature_names"], list):
            feature_names = self.sequences_data["feature_names"]
        else:
            feature_names = self.sequences_data["feature_names"].tolist()
        
        print(f"📊 Données originales: {input_sequences.shape}")
        print(f"📊 Features: {feature_names}")
        
        # Fit le scaler sur toutes les données
        self.scaler.fit_global_stats(input_sequences, feature_names)
        
        # Scaler toutes les séquences
        print("🔄 Scaling des séquences...")
        scaled_inputs = []
        scaled_targets = []
        all_scaling_params = []
        
        # for i in range(len(input_sequences)):
        #     scaled_input, scaled_target, params = self.scaler.scale_sequence(
        #         input_sequences[i], 
        #         target_sequences[i]
        #     )
        #     scaled_inputs.append(scaled_input)
        #     scaled_targets.append(scaled_target)
        #     all_scaling_params.append(params)

        print("🔄 Scaling des séquences (version rapide)...")
        scaled_inputs, scaled_targets, all_scaling_params = self.scaler.scale_sequences(
            input_sequences, 
            target_sequences
        )
        print("✅ Scaling terminé!")

        scaled_inputs = np.array(scaled_inputs)
        scaled_targets = np.array(scaled_targets)
        
        # Split par token pour éviter le data leakage
        unique_tokens = list(set([m["token"] for m in metadata]))
        np.random.shuffle(unique_tokens)
        
        n_val_tokens = int(len(unique_tokens) * validation_split)
        val_tokens = set(unique_tokens[:n_val_tokens])
        
        print(f"📊 Split: {len(unique_tokens)-n_val_tokens} tokens train, {n_val_tokens} tokens validation")
        
        # Créer les indices
        train_idx = [i for i, m in enumerate(metadata) if m["token"] not in val_tokens]
        val_idx = [i for i, m in enumerate(metadata) if m["token"] in val_tokens]
        
        # Créer les datasets
        train_data = {
            "inputs": scaled_inputs[train_idx],
            "targets": scaled_targets[train_idx],
            "metadata": [metadata[i] for i in train_idx],
            "scaling_params": [all_scaling_params[i] for i in train_idx]
        }
        
        val_data = {
            "inputs": scaled_inputs[val_idx],
            "targets": scaled_targets[val_idx],
            "metadata": [metadata[i] for i in val_idx],
            "scaling_params": [all_scaling_params[i] for i in val_idx]
        }
        
        # Statistiques
        print(f"\n✅ Données préparées:")
        print(f"  - Train: {len(train_data['inputs'])} séquences")
        print(f"  - Validation: {len(val_data['inputs'])} séquences")
        print(f"  - Shape input: {train_data['inputs'].shape}")
        print(f"  - Shape target: {train_data['targets'].shape}")
        
        # Vérifier les distributions
        print(f"\n📊 Statistiques après scaling:")
        for i, feat in enumerate(feature_names[:5]):  # Top 5 features
            train_values = train_data["inputs"][:, :, i].flatten()
            print(f"  - {feat}: mean={np.mean(train_values):.3f}, std={np.std(train_values):.3f}")
        
        return {
            "train": train_data,
            "validation": val_data,
            "scaler": self.scaler,
            "feature_names": feature_names,
            "metadata": {
                "sequence_length": self.sequences_data.get("sequence_length", 15),
                "forecast_steps": self.sequences_data.get("forecast_steps", 5),
                "n_features": self.sequences_data.get("n_features", len(feature_names))
            }
        }


if __name__ == "__main__":
    # Chemins
    sequences_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/sequences_raw.npz")
    output_dir = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/")
    
    # Préparer les données
    preparer = DataPreparer(sequences_path)
    prepared_data = preparer.prepare_for_training(validation_split=0.2)
    
    # Sauvegarder les données scalées - SANS les metadata objects
    output_path = output_dir / "sequences_scaled.npz"
    np.savez_compressed(
        output_path,
        train_inputs=prepared_data["train"]["inputs"],
        train_targets=prepared_data["train"]["targets"],
        val_inputs=prepared_data["validation"]["inputs"],
        val_targets=prepared_data["validation"]["targets"],
        feature_names=prepared_data["feature_names"],
        **prepared_data["metadata"]
    )
    print(f"\n💾 Données scalées sauvegardées: {output_path}")
    
    # Sauvegarder les metadata séparément en JSON
    metadata_path = output_dir / "sequences_metadata.json"
    metadata_to_save = {
        "train_tokens": list(set([m["token"] for m in prepared_data["train"]["metadata"]])),
        "val_tokens": list(set([m["token"] for m in prepared_data["validation"]["metadata"]])),
        "n_train_sequences": len(prepared_data["train"]["inputs"]),
        "n_val_sequences": len(prepared_data["validation"]["inputs"]),
        "feature_names": prepared_data["feature_names"],
        **prepared_data["metadata"]
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    print(f"💾 Metadata sauvegardés: {metadata_path}")
    
    # Sauvegarder le scaler
    scaler_path = output_dir / "scaler_params.json"
    prepared_data["scaler"].save_scaler(scaler_path)
    print(f"💾 Paramètres du scaler sauvegardés: {scaler_path}")
    
    # Exemple d'utilisation
    print("\n📋 Exemple de données:")
    example_idx = 0
    print(f"  - Input shape: {prepared_data['train']['inputs'][example_idx].shape}")
    print(f"  - Target shape: {prepared_data['train']['targets'][example_idx].shape}")
    print(f"  - Scaling params: {list(prepared_data['train']['scaling_params'][example_idx].keys())}")