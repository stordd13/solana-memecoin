#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read a sample parquet file to understand the structure
sample_files = [
    "../data/features/dead_tokens/121NRPjccxJgA7U1WkFuY7iM51oXPJWRKakaYKYkpump.parquet",
    "../data/features/normal_behavior_tokens/123evkMohRQw9X95RpyfM32m8MG6VCB72BhHBhmGpump.parquet"
]

for i, sample_file in enumerate(sample_files):
    print(f"\n=== File {i+1}: {sample_file.split('/')[-1]} ===")
    try:
        df = pd.read_parquet(sample_file)
        
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nIndex info:")
        print(f"Index type: {type(df.index)}")
        print(f"Index name: {df.index.name}")
        
        # Check if datetime column exists
        if 'datetime' in df.columns:
            print(f"Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Time span: {df['datetime'].max() - df['datetime'].min()}")
            
        # Check for time series characteristics
        print(f"Total rows: {len(df)}")
        if len(df) > 1:
            # Check for time intervals if datetime exists
            if 'datetime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['datetime']):
                time_diffs = df['datetime'].diff().dropna()
                if len(time_diffs) > 0:
                    print(f"Time intervals - min: {time_diffs.min()}, max: {time_diffs.max()}, median: {time_diffs.median()}")
        
    except Exception as e:
        print(f"Error reading file: {e}")