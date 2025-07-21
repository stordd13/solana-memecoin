import polars as pl
import matplotlib.pyplot as plt
import os
import time
import sys
import numpy as np
import re

# Dynamic path fix
bot_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(bot_root)
sys.path.append(os.path.join(bot_root, 'scripts'))
from utils import setup_logger

logger = setup_logger(__name__)

def detect_interval(filename: str) -> str:
    """Extract interval (1m, 5m, etc.) from filename"""
    # Look for patterns like processed_features_1m.parquet or processed_features_5m.parquet  
    match = re.search(r'_(\d+[a-z]+)\.parquet$', filename.lower())
    if match:
        return match.group(1)
    # Fallback: look for just the interval pattern anywhere in filename
    match = re.search(r'(\d+[a-z]+)', filename.lower())
    if match:
        return match.group(1)
    # Default fallback
    return "unknown"

def perform_eda(input_path: str) -> None:
    """
    EDA: Stats on dumps, distributions, correlations. Plots full + dump-flagged returns (incl. max/intra for extremes). Saves plots.
    """
    abs_path = os.path.join(bot_root, input_path)
    if not os.path.exists(abs_path):
        fallback_path = os.path.join(bot_root, os.path.basename(input_path))
        if os.path.exists(fallback_path):
            abs_path = fallback_path
        else:
            raise FileNotFoundError(f"Processed file missing: {abs_path} or {fallback_path}.")
    
    # Detect interval from filename for organized folder structure
    interval = detect_interval(os.path.basename(abs_path))
    logger.info(f"Detected interval: {interval}")
    
    start = time.time()
    df = pl.read_parquet(abs_path)
    logger.info(f"Loaded {abs_path} | Shape: {df.shape} | Time: {time.time() - start:.2f}s")
    
    # Use raw returns for statistics if available, otherwise fallback to scaled
    returns_col = "raw_returns" if "raw_returns" in df.columns else "returns"
    max_returns_col = "raw_max_returns" if "raw_max_returns" in df.columns else "max_returns"
    
    stats = df.select(
        pl.col("initial_dump_flag").mean().alias("dump_rate"),
        pl.col(returns_col).mean().alias("avg_returns"),
        pl.col(returns_col).max().alias("max_returns"),
        pl.col(returns_col).min().alias("min_returns"),
        pl.col(returns_col).std().alias("returns_std"),
        pl.corr("imbalance_ratio", returns_col).alias("imbalance_corr")
    )
    print("EDA Stats:", stats)
    logger.info(f"EDA Stats: {stats.to_dicts()}")
    
    # Helper to plot distributions with robust NaN handling
    def plot_dist(field: str, title_suffix: str, filter_expr=None):
        try:
            # Apply filter if provided
            if filter_expr is not None:
                filtered_data = df.filter(filter_expr)
                if filtered_data.height == 0:
                    logger.warning(f"Filter for {field} {title_suffix} returned empty dataset - skipping plot")
                    return
                data = filtered_data
            else:
                data = df
            
            # Extract values and handle NaN/inf
            values = data.select(field).to_numpy().flatten()
            
            # Check for valid data
            if len(values) == 0:
                logger.warning(f"No data for {field} {title_suffix} - empty array")
                return
                
            # Remove NaN and inf values  
            finite_values = values[np.isfinite(values)]
            
            if len(finite_values) == 0:
                logger.warning(f"No finite values for {field} {title_suffix} - all NaN/inf, skipping plot")
                return
                
            if len(finite_values) < 10:
                logger.warning(f"Only {len(finite_values)} finite values for {field} {title_suffix} - may not be meaningful")
            
            # Create histogram with finite values only
            plt.figure(figsize=(10, 6))
            plt.hist(finite_values, bins=min(50, len(finite_values)//2 or 1), alpha=0.7, edgecolor='black')
            plt.title(f"{field} Distribution {title_suffix} ({len(finite_values)} finite values)")
            plt.xlabel(field)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            # Create interval-specific folder and save
            figures_dir = os.path.join(bot_root, "analysis", "figures", interval)
            os.makedirs(figures_dir, exist_ok=True)
            plot_path = os.path.join(figures_dir, f"{field}_{title_suffix}.png")
            
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot: {plot_path} ({len(finite_values)} finite values)")
            
        except Exception as e:
            logger.error(f"Error plotting {field} {title_suffix}: {e}")
            plt.close()  # Ensure plot is closed even on error
    
    # Plot for all tokens + dump-flagged, using raw (unscaled) fields for meaningful plots
    raw_fields = []
    if 'raw_returns' in df.columns:
        raw_fields.extend(['raw_returns', 'raw_max_returns', 'raw_intra_interval_max_return', 'raw_volatility'])
    else:
        # Fallback to original fields if raw not available
        raw_fields.extend(['returns', 'max_returns', 'intra_interval_max_return'] if 'max_returns' in df.columns else ['returns'])
    
    for field in raw_fields:
        if field in df.columns:
            plot_dist(field, "All_Tokens")
            plot_dist(field, "Dump_Flagged_Tokens", pl.col("initial_dump_flag"))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EDA on processed features")
    parser.add_argument("--file", type=str, help="Specific file to analyze (e.g., processed_features_1m.parquet)")
    parser.add_argument("--interval", type=str, choices=["1m", "5m", "both"], default="both", 
                       help="Which interval(s) to analyze")
    
    args = parser.parse_args()
    
    if args.file:
        # Process specific file
        perform_eda(args.file)
    else:
        # Auto-detect and process available files based on interval choice
        files_to_process = []
        
        if args.interval in ["1m", "both"]:
            if os.path.exists("processed_features_1m.parquet"):
                files_to_process.append("processed_features_1m.parquet")
            else:
                print("Warning: processed_features_1m.parquet not found")
                
        if args.interval in ["5m", "both"]:
            if os.path.exists("processed_features_5m.parquet"):
                files_to_process.append("processed_features_5m.parquet")
            else:
                print("Warning: processed_features_5m.parquet not found")
        
        if not files_to_process:
            print("No processed feature files found. Please run the pipeline first.")
            sys.exit(1)
            
        # Process each file
        for file_path in files_to_process:
            print(f"\n{'='*50}")
            print(f"Processing: {file_path}")
            print(f"{'='*50}")
            try:
                perform_eda(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"\n🎉 EDA completed for {len(files_to_process)} file(s)")
        print("Figures organized in analysis/figures/<interval>/ folders")