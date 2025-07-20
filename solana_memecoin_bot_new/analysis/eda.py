import polars as pl
import matplotlib.pyplot as plt
import os
import time
import sys

# Dynamic path fix for root (config.py) and scripts/ (utils.py)
bot_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(bot_root)
sys.path.append(os.path.join(bot_root, 'scripts'))
from utils import setup_logger

logger = setup_logger(__name__)

def perform_eda(input_path: str) -> None:
    """
    EDA: Stats on dumps, distributions, correlations (e.g., imbalance vs pumps).
    Saves plots; interactive stub for Streamlit.
    """
    # Make input_path absolute and flexible (check root or data/processed/)
    abs_path = os.path.join(bot_root, input_path)
    if not os.path.exists(abs_path):
        # Fallback to root if in data/processed/
        fallback_path = os.path.join(bot_root, os.path.basename(input_path))
        if os.path.exists(fallback_path):
            abs_path = fallback_path
        else:
            logger.error(f"Processed file missing: {abs_path} or {fallback_path}. Run pipeline first to generate it.")
            raise FileNotFoundError(f"Processed file missing: {abs_path} or {fallback_path}. Run pipeline first.")
    
    start = time.time()
    df = pl.read_parquet(abs_path)
    logger.info(f"Loaded {abs_path} | Shape: {df.shape} | Time: {time.time() - start:.2f}s")
    
    stats = df.select(
        pl.col("initial_dump_flag").mean().alias("dump_rate"),
        pl.col("returns").mean().alias("avg_returns"),
        pl.corr("imbalance_ratio", "returns").alias("imbalance_corr")
    )
    print("EDA Stats:", stats)
    logger.info(f"EDA Stats: {stats.to_dicts()}")
    
    # Plot dump flags vs returns with matplotlib
    dump_returns = df.filter(pl.col("initial_dump_flag")).select("returns").to_numpy().flatten()
    if len(dump_returns) > 0:
        plt.hist(dump_returns, bins=50)
        plt.title("Returns Distribution for Dump-Flagged Tokens")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plot_path = os.path.join(bot_root, "analysis/figures/dump_returns.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # Create figures/ if missing
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
    else:
        logger.warning("No dump-flagged tokens for plot")

if __name__ == "__main__":
    perform_eda("processed_features_5m.parquet")  # Relative from root