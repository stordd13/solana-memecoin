import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis


def compute_log_returns(df: pl.DataFrame) -> pl.Series | None:
    """Calcule les log-returns et retourne une Series Polars."""
    if df.height < 2:
        return None
    log_returns = (
        df.select(
            pl.col("price")
            .log().diff()
            .alias("log_returns")
        )
        .drop_nulls()
        .get_column("log_returns")
    )
    return log_returns if not log_returns.is_empty() else None

def compute_volatility_log_return(df: pl.DataFrame) -> float | None:
    """Calcule la volatilité (écart-type) des log-returns."""
    log_returns = compute_log_returns(df)
    if log_returns is None:
        return None
    return float(log_returns.std())

def compute_kurtosis_log_return(df: pl.DataFrame) -> float | None:
    """Calcule le kurtosis des log-returns."""
    log_returns = compute_log_returns(df)
    if log_returns is None:
        return None
    kurtosis_value = float(kurtosis(log_returns.to_numpy().flatten(), fisher=True, bias=False))
    return max(1E-6, kurtosis_value)

def compute_price_ratio(df: pl.DataFrame) -> float | None:
    """Calcule le ratio prix max / prix min."""
    min_price, max_price = float(df["price"].min()), float(df["price"].max())
    return min(1E6, max_price / (min_price + 1E-9))

def compute_score(df: pl.DataFrame) -> float | None:
    """ Calcule un score normalisé entre 0 (mauvais) et 1 (excellent). """
    if df.is_empty():
        return None
    df = df.sort("date_utc")
    price_ratio = min(100, compute_price_ratio(df))
    if price_ratio == 1:
        return 0.0
    kurtosis_log_return = compute_kurtosis_log_return(df)
    volatility_log_return = compute_volatility_log_return(df)

    if kurtosis_log_return is None or volatility_log_return is None:
        return None

    score = (volatility_log_return / (kurtosis_log_return + 1E-9)) * (price_ratio - 1)
    score = (np.exp(score) - 1) * (price_ratio - 1)
    score = abs(score / (1 + score))

    return score


if __name__ == "__main__":
    data_file_path = Path(
        "/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/data_onchain_merged_tokens_list.parquet"
    )

    lf_raw = pl.scan_parquet(
        data_file_path,
        low_memory=False,
        cache=False,
        use_statistics=True,
        parallel="auto",
        rechunk=False,
    )

    lf = lf_raw.select("token_address", "date_utc", "price")
    
    # Get all unique token addresses
    print("Getting unique tokens...")
    tokens_address_df = (
        lf.select("token_address")
        .unique()
        .collect(engine="streaming")
    )
    tokens_list = tokens_address_df.get_column("token_address").to_list()
    
    print(f"Total tokens to process: {len(tokens_list)}")
    
    # Process tokens with progress tracking
    from tqdm import tqdm
    
    results = []
    for token in tqdm(tokens_list, desc="Computing scores"):
        token_df = lf.filter(pl.col("token_address") == token).collect(engine="streaming")
        
        score = compute_score(token_df)
        price_ratio = compute_price_ratio(token_df)
        volatility = compute_volatility_log_return(token_df)
        kurtosis_val = compute_kurtosis_log_return(token_df)
        
        results.append({
            "token_address": token,
            "score": score if score is not None else 0.0,
            "price_ratio": price_ratio if price_ratio is not None else 0.0,
            "volatility": volatility if volatility is not None else 0.0,
            "kurtosis": kurtosis_val if kurtosis_val is not None else 0.0,
            "has_valid_data": score is not None,
        })
    
    # Create results dataframe
    results_df = pl.DataFrame(results)
    
    print(f"\nTotal tokens processed: {len(results_df)}")
    
    # Filter tokens with score > 0.1
    high_score_tokens = results_df.filter(pl.col("score") > 0.2)
    
    # Save to Parquet
    output_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/scored_tokens_high_01.parquet")
    high_score_tokens.write_parquet(output_path, compression="snappy")
    
    print(f"✅ Found {len(high_score_tokens)} tokens with score > 0.2")
    print(f"💾 Saved to: {output_path}")
    
    # Show statistics
    print("\nScore distribution (valid data only):")
    print(results_df.filter(pl.col("has_valid_data")).select("score").describe())
    
    # Show top 10
    print("\nTop 10 tokens by score:")
    print(high_score_tokens.sort("score", descending=True).head(10))
    
    # Save all scores for reference
    all_scores_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/all_token_scores.parquet")
    results_df.write_parquet(all_scores_path, compression="snappy")
    print(f"\n💾 All scores saved to: {all_scores_path}")

    ## save jeff dataframe filtered on the high score
    addresses = high_score_tokens.select("token_address") \
        .unique()
        
    address_list = addresses.get_column("token_address").to_list()
    final = lf.filter(pl.col("token_address").is_in(address_list))
    final_path = Path("/Users/stordd/Documents/GitHub/Solana/memecoin2/data/jeff/data_onchain_merged_tokens_list_high_score.parquet")
    final.sink_parquet(final_path, compression="snappy")
    print(f"\n💾 All high scores saved to: {final_path}")

