# data_loader.py
"""
Data loading utilities for memecoin analysis.
Handles loading raw data from processed directories.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import warnings
warnings.filterwarnings('ignore')

from .death_detection import categorize_by_lifespan


def load_subsample_tokens(processed_dir: Path, n_tokens: int = 1000, 
                         categories: List[str] = None, seed: int = 42) -> Dict[str, pl.DataFrame]:
    """
    Load a random subsample of tokens from processed data directories.
    
    Args:
        processed_dir: Path to processed data directory
        n_tokens: Number of tokens to sample
        categories: List of categories to sample from (default: all available)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping token names to DataFrames
    """
    random.seed(seed)
    np.random.seed(seed)
    
    processed_path = Path(processed_dir)
    
    # Discover available categories if not specified
    if categories is None:
        categories = []
        for item in processed_path.iterdir():
            if item.is_dir() and item.name not in ['results', '__pycache__']:
                categories.append(item.name)
    
    # Collect all available token files
    all_token_files = []
    for category in categories:
        category_path = processed_path / category
        if category_path.exists():
            token_files = list(category_path.glob("*.parquet"))
            all_token_files.extend([(category, f) for f in token_files])
    
    if len(all_token_files) == 0:
        raise ValueError(f"No token files found in categories: {categories}")
    
    # Sample random tokens
    sampled_files = random.sample(all_token_files, min(n_tokens, len(all_token_files)))
    
    # Load sampled tokens
    token_data = {}
    for category, file_path in sampled_files:
        try:
            df = pl.read_parquet(file_path)
            token_name = file_path.stem
            token_data[token_name] = df
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    return token_data


def load_categorized_tokens(processed_dir: Path, 
                          max_tokens_per_category: Optional[int] = None,
                          sample_ratio: Optional[float] = None,
                          categories: List[str] = None) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Load tokens organized by category.
    
    Args:
        processed_dir: Path to processed data directory
        max_tokens_per_category: Maximum tokens per category
        sample_ratio: Fraction of tokens to sample (0.0 to 1.0)
        categories: List of categories to load (default: all available)
        
    Returns:
        Dictionary mapping category -> token_name -> DataFrame
    """
    processed_path = Path(processed_dir)
    
    # Discover available categories if not specified
    if categories is None:
        categories = []
        for item in processed_path.iterdir():
            if item.is_dir() and item.name not in ['results', '__pycache__']:
                categories.append(item.name)
    
    categorized_data = {}
    
    for category in categories:
        category_path = processed_path / category
        if not category_path.exists():
            print(f"Warning: Category {category} not found")
            continue
        
        # Get all token files in category
        token_files = list(category_path.glob("*.parquet"))
        
        # Apply sampling if specified
        if sample_ratio is not None:
            n_sample = int(len(token_files) * sample_ratio)
            token_files = random.sample(token_files, min(n_sample, len(token_files)))
        
        # Apply max tokens limit
        if max_tokens_per_category is not None:
            token_files = token_files[:max_tokens_per_category]
        
        # Load tokens
        category_data = {}
        for file_path in token_files:
            try:
                df = pl.read_parquet(file_path)
                token_name = file_path.stem
                category_data[token_name] = df
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        categorized_data[category] = category_data
        print(f"Loaded {len(category_data)} tokens from {category}")
    
    return categorized_data


def prepare_token_for_analysis(token_df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare token data for analysis by extracting prices and calculating returns.
    
    Args:
        token_df: DataFrame with price data
        
    Returns:
        Tuple of (prices, returns) arrays
    """
    if 'price' not in token_df.columns:
        raise ValueError("Token DataFrame must contain 'price' column")
    
    prices = token_df['price'].to_numpy()
    
    # Calculate raw returns
    if len(prices) > 1:
        returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-12)
    else:
        returns = np.array([0.0])
    
    return prices, returns


def categorize_tokens_by_lifespan(token_data: Dict[str, pl.DataFrame]) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Categorize tokens by lifespan (sprint/standard/marathon).
    
    Args:
        token_data: Dictionary mapping token names to DataFrames
        
    Returns:
        Dictionary mapping lifespan category to token data
    """
    categorized = {
        'sprint': {},
        'standard': {},
        'marathon': {}
    }
    
    for token_name, token_df in token_data.items():
        lifespan = len(token_df)  # Number of minutes
        category = categorize_by_lifespan(lifespan)
        categorized[category][token_name] = token_df
    
    return categorized


def validate_token_data(token_df: pl.DataFrame) -> bool:
    """
    Validate that token data meets minimum requirements.
    
    Args:
        token_df: Token DataFrame
        
    Returns:
        True if valid, False otherwise
    """
    if token_df is None or token_df.height == 0:
        return False
    
    if 'price' not in token_df.columns:
        return False
    
    # Check for reasonable data length
    if token_df.height < 5:  # Minimum 5 minutes
        return False
    
    # Check for valid prices
    prices = token_df['price'].to_numpy()
    if np.any(prices <= 0) or np.any(~np.isfinite(prices)):
        return False
    
    return True


def get_base_price_groups(token_data: Dict[str, pl.DataFrame]) -> Tuple[List[str], List[str]]:
    """
    Split tokens into low and high base price groups for scale invariance testing.
    
    Args:
        token_data: Dictionary mapping token names to DataFrames
        
    Returns:
        Tuple of (low_base_price_tokens, high_base_price_tokens) lists
    """
    base_prices = []
    token_names = []
    
    for token_name, token_df in token_data.items():
        if validate_token_data(token_df):
            base_price = token_df['price'][0]
            base_prices.append(base_price)
            token_names.append(token_name)
    
    if len(base_prices) == 0:
        return [], []
    
    # Calculate median base price
    median_base = np.median(base_prices)
    
    low_tokens = [name for name, price in zip(token_names, base_prices) if price < median_base]
    high_tokens = [name for name, price in zip(token_names, base_prices) if price >= median_base]
    
    return low_tokens, high_tokens