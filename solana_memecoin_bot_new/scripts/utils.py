# scripts/utils.py (Shared utility for logging; modular for all scripts)
import logging
import os
import sys
import glob
import joblib
import torch
from typing import Dict, Optional, Tuple

# Dynamic path fix for config.py in root
bot_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(bot_root)
sys.path.append(os.path.join(bot_root, 'ml'))
import config

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with file and console handlers. Levels based on config.DEBUG_MODE.
    """
    logger = logging.getLogger(name)
    level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    logger.setLevel(level)
    
    # File handler
    log_dir = os.path.join(config.BOT_NEW_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'bot.log'))
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_unified_models(interval: str = "5m", models_dir: Optional[str] = None) -> Dict:
    """
    Load unified models (transformer and baseline) for the specified interval.
    
    Args:
        interval: Data interval ('1m' or '5m')
        models_dir: Custom models directory path
        
    Returns:
        Dictionary containing loaded models and metadata
    """
    if models_dir is None:
        models_dir = os.path.join(bot_root, 'models')
    
    logger = setup_logger(__name__)
    loaded_models = {}
    
    # Load baseline XGBoost model
    baseline_pattern = os.path.join(models_dir, f'baseline_{interval}_unified*.pkl')
    baseline_files = glob.glob(baseline_pattern)
    
    if baseline_files:
        baseline_file = sorted(baseline_files)[-1]  # Get most recent
        try:
            loaded_models['baseline'] = joblib.load(baseline_file)
            logger.info(f"Loaded baseline model: {baseline_file}")
        except Exception as e:
            logger.warning(f"Failed to load baseline model {baseline_file}: {e}")
    else:
        logger.warning(f"No baseline model found for {interval}")
    
    # Load transformer model
    transformer_pattern = os.path.join(models_dir, f'transformer_{interval}_unified*.pth')
    transformer_files = glob.glob(transformer_pattern)
    
    if transformer_files:
        transformer_file = sorted(transformer_files)[-1]  # Get most recent
        try:
            # Import transformer class dynamically
            from transformer_forecast import UnifiedPumpTransformer
            
            # Load model state
            checkpoint = torch.load(transformer_file, map_location='cpu')
            
            # Get model parameters from checkpoint or use defaults
            input_dim = checkpoint.get('input_dim', 8)
            max_seq_len = checkpoint.get('max_seq_len', 60)
            
            # Create and load model
            model = UnifiedPumpTransformer(input_dim=input_dim, max_seq_len=max_seq_len)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            loaded_models['transformer'] = {
                'model': model,
                'features': checkpoint.get('features', ["scaled_returns", "ma_5", "rsi_14", "max_total_return", 
                                                      "max_returns", "vol_std_5", "momentum_lag1", "vol_return_ratio"]),
                'metrics': checkpoint.get('metrics', {}),
                'window_size': checkpoint.get('window_size', 20)
            }
            logger.info(f"Loaded transformer model: {transformer_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load transformer model {transformer_file}: {e}")
    else:
        logger.warning(f"No transformer model found for {interval}")
    
    return loaded_models

def get_unified_features() -> list:
    """
    Get the standard feature list for unified models (NO DATA LEAKAGE).
    
    Returns:
        List of legitimate feature column names
    """
    return ["scaled_returns", "ma_5", "rsi_14", "max_returns", "vol_std_5", 
            "momentum_lag1", "vol_return_ratio", "initial_price",
            "minutes_since_start", "current_total_return", "recent_avg_volatility", 
            "volatility_regime", "token_age_category"]

def validate_feature_compatibility(df_columns: list, required_features: list) -> Tuple[bool, list]:
    """
    Validate that the dataframe has all required features for model inference.
    
    Args:
        df_columns: List of dataframe column names
        required_features: List of required feature names
        
    Returns:
        Tuple of (is_compatible, missing_features)
    """
    missing_features = [feat for feat in required_features if feat not in df_columns]
    is_compatible = len(missing_features) == 0
    
    return is_compatible, missing_features