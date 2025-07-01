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
        )
    }
    
    def __init__(self, 
                 config: Optional[Union[str, WalkForwardConfig]] = None,
                 min_folds: int = 2,
                 max_folds: int = 10):
        """
        Initialize the splitter.
        
        Args:
            config: Either a string key ('short', 'medium', 'long', 'very_long') 
                   or a custom WalkForwardConfig
            min_folds: Minimum number of folds to generate
            max_folds: Maximum number of folds to generate
        """
        if isinstance(config, str):
            self.config = self.CONFIGS[config]
        elif isinstance(config, WalkForwardConfig):
            self.config = config
        else:
            self.config = None  # Will be determined adaptively
            
        self.min_folds = min_folds
        self.max_folds = max_folds
        
    def get_adaptive_config(self, data_length: int) -> WalkForwardConfig:
        """
        Automatically determine the best configuration based on data length.
        
        Args:
            data_length: Number of time steps in the data
            
        Returns:
            WalkForwardConfig appropriate for the data length
        """
        if data_length < 600:  # Less than 10 hours
            config = self.CONFIGS['short']
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
        
        # Generate folds
        fold_count = 0
        current_train_end = config.min_train_size
        
        while current_train_end + config.test_size <= n_samples and fold_count < self.max_folds:
            if config.strategy == 'expanding':
                train_start = 0
            else:  # sliding
                train_start = max(0, current_train_end - config.max_train_size)
                
            train_indices = np.arange(train_start, current_train_end)
            test_indices = np.arange(current_train_end, 
                                   min(current_train_end + config.test_size, n_samples))
            
            yield train_indices, test_indices
            
            current_train_end += config.step_size
            fold_count += 1
            
        logger.info(f"Generated {fold_count} folds for {n_samples} samples")
        
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