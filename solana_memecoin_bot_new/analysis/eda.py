import polars as pl
import matplotlib.pyplot as plt

def perform_eda(input_path: str) -> None:
    """
    EDA: Stats on dumps, distributions, correlations (e.g., imbalance vs pumps).
    Saves plots; interactive stub for Streamlit.
    """
    df = pl.read_parquet(input_path)
    stats = df.select(
        pl.col("initial_dump_flag").mean().alias("dump_rate"),
        pl.col("returns").mean().alias("avg_returns"),
        pl.corr("imbalance_ratio", "returns").alias("imbalance_corr")
    )
    print("EDA Stats:", stats)
    
    # Plot dump flags vs returns
    df.filter(pl.col("initial_dump_flag")).select("returns").plot.hist(bins=50)
    plt.title("Returns Distribution for Dump-Flagged Tokens")
    plt.savefig("analysis/figures/dump_returns.png")

if __name__ == "__main__":
    perform_eda("data/processed/processed_features_5m.parquet")