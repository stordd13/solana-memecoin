"""
Data loader for memecoin analysis
"""

import polars as pl
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import re
import os

class DataLoader:
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader
        
        Args:
            base_path: Base path to the data directory. If None, uses default path.
        """
        if base_path is None:
            # Try to find the data directory relative to the current file
            current_dir = Path(__file__).parent
            base_path = current_dir.parent / "data" / "raw"
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        self.logger = logging.getLogger(__name__)
        self.token_cache: Optional[List[Dict[str, str]]] = None
        
        # Log the data path
        self.logger.info(f"Using data path: {self.base_path}")
        if not self.base_path.exists():
            self.logger.warning(f"Data path does not exist: {self.base_path}")
            
    def get_dataset_folders(self) -> List[Path]:
        """Get list of all dataset folders under the base path, including nested subfolders. Also include the base path itself if it contains .parquet files."""
        try:
            if not self.base_path.exists():
                self.logger.error(f"Data path does not exist: {self.base_path}")
                return []
            subdirs = [p for p in self.base_path.rglob('*') if p.is_dir() and any(f.suffix == '.parquet' for f in p.glob('*.parquet'))]
            # Also include the base path itself if it contains .parquet files
            if any(f.suffix == '.parquet' for f in self.base_path.glob('*.parquet')):
                subdirs.append(self.base_path)
            if not subdirs:
                self.logger.warning(f"No subdirectories with parquet files found under {self.base_path}")
            # Return relative paths from base_path for display/selection
            return [p.relative_to(self.base_path) if p != self.base_path else Path('.') for p in subdirs]
        except Exception as e:
            self.logger.error(f"Error getting dataset folders: {e}")
            return []
            
    def get_available_tokens(self) -> List[Dict[str, str]]:
        if self.token_cache is not None:
            return self.token_cache

        tokens = []
        try:
            if not self.base_path.exists():
                self.logger.error(f"Data path does not exist: {self.base_path}")
                return []
            
            for file in self.base_path.rglob("*.parquet"):
                address = file.stem
                tokens.append({
                    'symbol': address,
                    'address': address,
                    'file': str(file)
                })

            if not tokens:
                self.logger.warning(f"No token files found in {self.base_path}")
            
            self.token_cache = tokens
            return tokens
        except Exception as e:
            self.logger.error(f"Error getting available tokens: {e}")
            return []
            
    def get_token_data(self, token_symbol: str) -> Optional[pl.DataFrame]:
        try:
            available_tokens = self.get_available_tokens()
            token_info = next((t for t in available_tokens if t['symbol'] == token_symbol), None)
            
            if not token_info:
                self.logger.warning(f"Token not found: {token_symbol}")
                return None
                
            file_path = token_info['file']
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                return None
                
            df = pl.read_parquet(file_path)
            
            required_cols = ['datetime', 'price']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns in {token_symbol}. Required: {required_cols}, Found: {df.columns}")
                return None
                
            if df['datetime'].dtype != pl.Datetime:
                df = df.with_columns(pl.col('datetime').cast(pl.Datetime))
                    
            if df['price'].dtype not in [pl.Float64, pl.Float32]:
                df = df.with_columns(pl.col('price').cast(pl.Float64))
                    
            df = df.sort('datetime')
            
            df = df.with_columns([
                pl.lit(token_info['symbol']).alias('token'),
                pl.lit(token_info['address']).alias('address'),
            ])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {token_symbol}: {e}")
            return None
        
    def load_data(self, 
                 selected_tokens: Optional[List[str]] = None,
                 selected_datasets: Optional[List[str]] = None) -> Optional[pl.DataFrame]:
        """
        Load data for selected tokens and datasets
        
        Args:
            selected_tokens: List of token symbols to load. If None, loads all tokens.
            selected_datasets: List of dataset names to load (not used in new structure)
            
        Returns:
            DataFrame with the loaded data or None if error
        """
        try:
            if not self.base_path.exists():
                self.logger.error(f"Data path does not exist: {self.base_path}")
                return None
                
            all_data = []
            tokens = self.get_available_tokens()
            
            if not tokens:
                self.logger.warning("No tokens found")
                return None
            
            # Filter tokens based on selection
            if selected_tokens:
                tokens = [t for t in tokens if t['symbol'] in selected_tokens]
                
            if not tokens:
                self.logger.warning("No tokens found matching selection criteria")
                return None
                
            # Load data for each token
            for token in tokens:
                try:
                    if not os.path.exists(token['file']):
                        self.logger.warning(f"File not found: {token['file']}")
                        continue
                        
                    df = pl.read_parquet(token['file'])
                    
                    # Add token info
                    df = df.with_columns([
                        pl.lit(token['symbol']).alias('token'),
                        pl.lit(token['address']).alias('address'),
                    ])
                    all_data.append(df)
                    
                except Exception as e:
                    self.logger.error(f"Error loading data for {token['symbol']}: {e}")
                    continue
                    
            if not all_data:
                self.logger.warning("No data loaded")
                return None
                
            # Combine all data
            return pl.concat(all_data)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None

    # -------------------------------------------------------------
    # Additional helper methods for compatibility with quant_app.py
    # -------------------------------------------------------------

    def get_subdirectories(self, dataset_name: str):
        """Return list of subdirectories for a given dataset.

        In the current flat structure, there are no subdirectories, so we
        return an empty list to signal that tokens are directly in the
        dataset folder.
        """
        dataset_folder = self.base_path / dataset_name
        if dataset_folder.exists() and dataset_folder.is_dir():
            subdirs = [p for p in dataset_folder.iterdir() if p.is_dir()]
            return subdirs
        return []

    def get_parquet_files(self, folder_path):
        """Return list of parquet files inside the supplied folder path."""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            self.logger.warning(f"Folder does not exist: {folder_path}")
            return []
        return sorted(folder_path.glob("*.parquet"))

    def get_token_info(self, parquet_path):
        """Return a basic token info dict given a parquet file path."""
        parquet_path = Path(parquet_path)
        symbol = parquet_path.stem
        return {"symbol": symbol, "file": str(parquet_path)}

    def load_token_data(self, parquet_path):
        """Load a single parquet file and return a polars DataFrame."""
        try:
            df = pl.scan_parquet(parquet_path).collect()
            # Ensure expected columns
            if set(df.columns) >= {"datetime", "price"}:
                # Ensure correct dtypes
                if df['datetime'].dtype != pl.Datetime:
                    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))
                if df['price'].dtype not in [pl.Float64, pl.Float32]:
                    df = df.with_columns(pl.col('price').cast(pl.Float64))
                return df
            else:
                self.logger.warning(f"Missing expected columns in {parquet_path}")
                return pl.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading parquet {parquet_path}: {e}")
            return pl.DataFrame()