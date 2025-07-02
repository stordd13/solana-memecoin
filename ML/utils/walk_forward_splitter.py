"""
Walk-Forward Validation Splitter for Time Series Data

This module provides a unified walk-forward validation strategy that:
1. Handles varying token lengths (400-2000 minutes)
2. Works with all model types (regression, tree-based, LSTM)
3. Maintains temporal order and prevents data leakage
4. Provides both expanding and sliding window options
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    min_train_size: int  # Minimum training window size
    step_size: int  # How much to move forward each fold
    test_size: int  # Size of each test window
    strategy: str = 'expanding'  # 'expanding' or 'sliding'
    max_train_size: Optional[int] = None  # For sliding window
    
    def __post_init__(self):
        if self.strategy == 'sliding' and self.max_train_size is None:
            self.max_train_size = self.min_train_size * 2
            

class WalkForwardSplitter:
    """
    Unified walk-forward validation splitter for time series data.
    
    Handles:
    - Adaptive configuration based on data length
    - Both expanding and sliding window strategies
    - Per-token and cross-token splitting
    - Integration with existing scaling approaches
    """
    
    # Predefined configurations for different data lengths
    CONFIGS = {
        'short': WalkForwardConfig(
            min_train_size=240,  # 4 hours
            step_size=60,        # 1 hour step
            test_size=60,        # 1 hour test
            strategy='expanding'
        ),
        'medium': WalkForwardConfig(
            min_train_size=480,  # 8 hours
            step_size=120,       # 2 hour step
            test_size=120,       # 2 hour test
            strategy='expanding'
        ),
        'long': WalkForwardConfig(
            min_train_size=720,  # 12 hours
            step_size=180,       # 3 hour step
            test_size=180,       # 3 hour test
            strategy='expanding'
        ),
        'very_long': WalkForwardConfig(
            min_train_size=960,  # 16 hours
            step_size=240,       # 4 hour step
            test_size=240,       # 4 hour test
            strategy='expanding'
        ),
        'memecoin_micro': WalkForwardConfig(
            min_train_size=60,   # 1 hour - minimal but workable
            step_size=30,        # 30 minute step
            test_size=30,        # 30 minute test
            strategy='expanding'
        ),
        'memecoin_short': WalkForwardConfig(
            min_train_size=120,  # 2 hours - good balance
            step_size=30,        # 30 minute step
            test_size=30,        # 30 minute test
            strategy='expanding'
        ),
        'memecoin_medium': WalkForwardConfig(
            min_train_size=180,  # 3 hours - conservative but reasonable
            step_size=60,        # 1 hour step
            test_size=60,        # 1 hour test
            strategy='expanding'
        ),
        'lightgbm_short_term': WalkForwardConfig(
            min_train_size=120,  # REDUCED from 360 to 120 (2 hours)
            step_size=60,        # 1 hour step
            test_size=60,        # 1 hour test
            strategy='expanding'
        ),
        'lightgbm_medium_term': WalkForwardConfig(
            min_train_size=240,  # REDUCED from 720 to 240 (4 hours)
            step_size=120,       # 2 hour step
            test_size=120,       # 2 hour test
            strategy='expanding'
        )
    }
    
    def __init__(self, 
                 config: Optional[Union[str, WalkForwardConfig]] = None,
                 min_folds: int = 2,
                 max_folds: int = 10,
                 horizon_buffer: int = 0):
        """
        Initialize the splitter.
        
        Args:
            config: Either a string key ('short', 'medium', 'long', 'very_long') 
                   or a custom WalkForwardConfig
            min_folds: Minimum number of folds to generate
            max_folds: Maximum number of folds to generate
            horizon_buffer: Buffer to leave at the end for creating labels (e.g., 720 for 12h horizon)
        """
        if isinstance(config, str):
            self.config = self.CONFIGS[config]
        elif isinstance(config, WalkForwardConfig):
            self.config = config
        else:
            self.config = None  # Will be determined adaptively
            
        self.min_folds = min_folds
        self.max_folds = max_folds
        self.horizon_buffer = horizon_buffer
        
    def get_adaptive_config(self, data_length: int) -> WalkForwardConfig:
        """
        Automatically determine the best configuration based on data length.
        
        Args:
            data_length: Number of time steps in the data
            
        Returns:
            WalkForwardConfig appropriate for the data length
        """
        # NEW: Memecoin-aware adaptive configuration
        if data_length < 200:    # Less than 3.3 hours - very short memecoins
            config = self.CONFIGS['memecoin_micro']
        elif data_length < 400:  # Less than 6.7 hours - typical short memecoins  
            config = self.CONFIGS['memecoin_short']
        elif data_length < 600:  # Less than 10 hours - medium memecoins
            config = self.CONFIGS['memecoin_medium']
        elif data_length < 1200:  # Less than 20 hours
            config = self.CONFIGS['medium']
        elif data_length < 1800:  # Less than 30 hours
            config = self.CONFIGS['long']
        else:
            config = self.CONFIGS['very_long']
            
        # Adjust if needed to ensure minimum folds
        max_possible_folds = (data_length - config.min_train_size) // config.step_size
        if max_possible_folds < self.min_folds:
            # Reduce step size to get more folds
            new_step_size = max((data_length - config.min_train_size) // self.min_folds, 30)
            config = WalkForwardConfig(
                min_train_size=config.min_train_size,
                step_size=new_step_size,
                test_size=new_step_size,
                strategy=config.strategy
            )
            
        return config
        
    def split(self, 
              data: Union[pl.DataFrame, np.ndarray],
              time_column: Optional[str] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for walk-forward validation.
        
        Args:
            data: Either a Polars DataFrame or numpy array
            time_column: Column name for time ordering (if DataFrame)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        # Determine data length
        if isinstance(data, pl.DataFrame):
            n_samples = len(data)
            if time_column:
                # Ensure data is sorted by time
                data = data.sort(time_column)
        else:
            n_samples = len(data)
            
        # Get configuration
        config = self.config or self.get_adaptive_config(n_samples)
        
        # Adjust effective data length to account for horizon buffer
        effective_n_samples = n_samples - self.horizon_buffer
        
        # Generate folds
        fold_count = 0
        current_train_end = config.min_train_size
        
        while current_train_end + config.test_size <= effective_n_samples and fold_count < self.max_folds:
            if config.strategy == 'expanding':
                train_start = 0
            else:  # sliding
                train_start = max(0, current_train_end - config.max_train_size)
                
            train_indices = np.arange(train_start, current_train_end)
            test_indices = np.arange(current_train_end, 
                                   min(current_train_end + config.test_size, effective_n_samples))
            
            yield train_indices, test_indices
            
            current_train_end += config.step_size
            fold_count += 1
            
        logger.info(f"Generated {fold_count} folds for {n_samples} samples (with {self.horizon_buffer} buffer)")
        
    def split_by_token(self,
                       data: pl.DataFrame,
                       token_column: str = 'token_address',
                       time_column: str = 'minutes_since_launch',
                       min_token_length: int = 300) -> Dict[str, List[Tuple[pl.DataFrame, pl.DataFrame]]]:
        """
        Split data by token, applying walk-forward validation to each token separately.
        
        Args:
            data: Polars DataFrame with multiple tokens
            token_column: Column identifying different tokens
            time_column: Column for time ordering
            min_token_length: Minimum length required for a token to be included
            
        Returns:
            Dictionary mapping token addresses to list of (train, test) DataFrames
        """
        results = {}
        
        for token in data[token_column].unique():
            token_data = data.filter(pl.col(token_column) == token).sort(time_column)
            
            if len(token_data) < min_token_length:
                logger.warning(f"Token {token} has only {len(token_data)} samples, skipping")
                continue
                
            token_splits = []
            for train_idx, test_idx in self.split(token_data, time_column):
                train_data = token_data[train_idx]
                test_data = token_data[test_idx]
                token_splits.append((train_data, test_data))
                
            results[token] = token_splits
            
        logger.info(f"Split {len(results)} tokens with walk-forward validation")
        return results
        
    def get_global_splits(self,
                          data: pl.DataFrame,
                          time_column: str = 'minutes_since_launch') -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Get walk-forward splits treating all data as one continuous series.
        Useful for models that learn across tokens.
        
        Args:
            data: Polars DataFrame
            time_column: Column for time ordering
            
        Returns:
            List of (train, test) DataFrames
        """
        # Sort by time globally
        data = data.sort(time_column)
        
        splits = []
        for train_idx, test_idx in self.split(data, time_column):
            train_data = data[train_idx]
            test_data = data[test_idx]
            splits.append((train_data, test_data))
            
        return splits
        
    def get_fold_info(self, data_length: int) -> List[Dict[str, int]]:
        """
        Get information about the folds that would be generated for a given data length.
        
        Args:
            data_length: Number of time steps
            
        Returns:
            List of dictionaries with fold information
        """
        config = self.config or self.get_adaptive_config(data_length)
        
        fold_info = []
        fold_count = 0
        current_train_end = config.min_train_size
        
        while current_train_end + config.test_size <= data_length and fold_count < self.max_folds:
            if config.strategy == 'expanding':
                train_start = 0
            else:
                train_start = max(0, current_train_end - config.max_train_size)
                
            fold_info.append({
                'fold': fold_count,
                'train_start': train_start,
                'train_end': current_train_end,
                'test_start': current_train_end,
                'test_end': min(current_train_end + config.test_size, data_length),
                'train_size': current_train_end - train_start,
                'test_size': min(config.test_size, data_length - current_train_end)
            })
            
            current_train_end += config.step_size
            fold_count += 1
            
        return fold_info

    def get_adaptive_horizon_buffer(self, data_length: int, max_horizon: int = 720) -> int:
        """
        Get adaptive horizon buffer based on data length and max prediction horizon.
        
        For short tokens, we can't afford to reserve 12h (720min) at the end.
        Instead, we use a smaller buffer or only predict shorter horizons.
        
        Args:
            data_length: Number of time steps in the data
            max_horizon: Maximum prediction horizon in minutes
            
        Returns:
            Appropriate horizon buffer size
        """
        # For very short tokens, use minimal buffer
        if data_length < 300:  # Less than 5 hours
            return min(60, max_horizon)  # 1h buffer max
        elif data_length < 600:  # Less than 10 hours  
            return min(240, max_horizon)  # 4h buffer max
        elif data_length < 1200:  # Less than 20 hours
            return min(480, max_horizon)  # 8h buffer max
        else:
            return max_horizon  # Full buffer for long tokens
    
    def get_available_horizons(self, data_length: int, horizons: List[int]) -> List[int]:
        """
        Get horizons that are feasible given the data length.
        
        Args:
            data_length: Number of time steps in the data
            horizons: List of desired horizons in minutes
            
        Returns:
            List of feasible horizons
        """
        # Reserve 25% of data for horizon buffer as maximum
        max_feasible_horizon = int(data_length * 0.25)
        
        return [h for h in horizons if h <= max_feasible_horizon]

    def smart_split_for_memecoins(self,
                                  data: pl.DataFrame,
                                  horizons: List[int],
                                  time_column: str = 'datetime') -> Tuple[List[Tuple[pl.DataFrame, pl.DataFrame]], List[int]]:
        """
        Smart split that automatically adapts to memecoin data characteristics.
        
        - Uses adaptive horizon buffer based on data length
        - Filters horizons to only feasible ones
        - Uses memecoin-appropriate configurations
        
        Args:
            data: Polars DataFrame
            horizons: Desired prediction horizons in minutes
            time_column: Column for time ordering
            
        Returns:
            Tuple of (splits, feasible_horizons)
        """
        data_length = len(data)
        
        # Get adaptive horizon buffer
        adaptive_buffer = self.get_adaptive_horizon_buffer(data_length, max(horizons))
        
        # Get feasible horizons
        feasible_horizons = self.get_available_horizons(data_length, horizons)
        
        if not feasible_horizons:
            print(f"⚠️  No feasible horizons for {data_length}-minute token!")
            return [], []
        
        print(f"📊 Token length: {data_length} minutes")
        print(f"🎯 Feasible horizons: {feasible_horizons} (filtered from {horizons})")
        print(f"🛡️  Adaptive buffer: {adaptive_buffer} minutes (instead of {max(horizons)})")
        
        # Temporarily override horizon buffer
        original_buffer = self.horizon_buffer
        self.horizon_buffer = adaptive_buffer
        
        try:
            # Get splits with adaptive buffer
            splits = self.get_global_splits(data, time_column)
            return splits, feasible_horizons
        finally:
            # Restore original buffer
            self.horizon_buffer = original_buffer


def create_walk_forward_splits(data: Union[pl.DataFrame, np.ndarray],
                               config: Optional[Union[str, WalkForwardConfig]] = None,
                               **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to create walk-forward splits.
    
    Args:
        data: Data to split
        config: Configuration (string key or WalkForwardConfig)
        **kwargs: Additional arguments passed to WalkForwardSplitter
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splitter = WalkForwardSplitter(config=config)
    return list(splitter.split(data, **kwargs))


# Example usage patterns for different model types
def example_lightgbm_usage():
    """Example of using walk-forward validation with LightGBM"""
    # This would be in your training script
    splitter = WalkForwardSplitter(config='medium')
    
    # For cross-token learning (like current LightGBM approach)
    # data = pl.read_parquet('features.parquet')
    # folds = splitter.get_global_splits(data, time_column='minutes_since_launch')
    
    # for fold_idx, (train_df, test_df) in enumerate(folds):
    #     # Train model on train_df
    #     # Evaluate on test_df
    #     # Average metrics across folds
    

def example_lstm_usage():
    """Example of using walk-forward validation with LSTM"""
    # This would be in your training script
    splitter = WalkForwardSplitter()  # Adaptive config
    
    # For per-token LSTM training
    # data = pl.read_parquet('features.parquet')
    # token_splits = splitter.split_by_token(data)
    
    # for token, folds in token_splits.items():
    #     for fold_idx, (train_df, test_df) in enumerate(folds):
    #         # Create sequences from train_df
    #         # Train LSTM
    #         # Evaluate on test_df sequences
    

def example_integration_with_scaling():
    """Example of integrating with existing scaling approaches"""
    # splitter = WalkForwardSplitter(config='long')
    # data = pl.read_parquet('features.parquet')
    
    # # For each fold
    # for train_idx, test_idx in splitter.split(data):
    #     train_data = data[train_idx]
    #     test_data = data[test_idx]
    #     
    #     # Fit scaler on training data only
    #     scaler = Winsorizer(lower_quantile=0.005, upper_quantile=0.995)
    #     scaler.fit(train_data.select(feature_columns))
    #     
    #     # Transform both sets
    #     train_scaled = scaler.transform(train_data.select(feature_columns))
    #     test_scaled = scaler.transform(test_data.select(feature_columns)) 