import polars as pl
import config

def simulate_trades(df: pl.DataFrame, model_results: dict, archetype: str = 'Sprint', use_early: bool = True) -> pl.DataFrame:
    """
    Small-start sim for 'Sprint': Buy if imbalance > threshold and no dump flag in first 5 min, hold 10 min with trailing stop.
    Low-stake with Kelly, slippage/fees, <5% drawdown. Challenge: Compare with/without early clustering.
    """
    if archetype:
        df = df.filter(pl.col("archetype") == archetype & ~pl.col("initial_dump_flag"))
    if use_early:
        df = df.filter(pl.col("early_archetype") == 0)  # Assume 0 = 'Sprint' in early clusters
    
    # Strategy: Buy on signal, hold, sell on trail/drawdown
    df = df.with_columns(pl.lit(0.0).alias("position"), pl.lit(0.0).alias("net_profit"))
    for token in df["token_id"].unique():
        token_df = df.filter(pl.col("token_id") == token).sort("timestamp")
        if token_df["imbalance_ratio"][0] > config.IMBALANCE_BUY_THRESHOLD and not token_df["initial_dump_flag"][0]:
            stake = min(config.STAKE_RANGE[1], max(config.STAKE_RANGE[0], config.MAX_POSITION_FRACTION * 1000))  # Kelly stub
            entry_price = token_df["avg_price"][0]
            token_df = token_df.with_columns(pl.lit(stake / entry_price).alias("position"))
            
            # Hold sim with trail
            peak = entry_price
            for i in range(1, len(token_df)):
                price = token_df["avg_price"][i]
                peak = max(peak, price)
                if price < peak * (1 - config.TRAILING_STOP) or (price - entry_price) / entry_price < -config.MAX_DRAWDOWN:
                    profit = (price - entry_price) * token_df["position"][i] * (1 - config.SLIPPAGE_RATE) - stake * config.FEE_RATE
                    token_df = token_df.with_columns(pl.when(pl.col("row_nr") == i).then(profit).otherwise(pl.col("net_profit")).alias("net_profit"))
                    break  # Sell
                
    win_rate = df.filter(pl.col("net_profit") > 0).shape[0] / df.filter(pl.col("position") > 0).shape[0]
    print(f"'Sprint' Sim Win Rate: {win_rate:.2%}, Avg Profit: {df['net_profit'].mean():.2f}")
    df.write_csv("logs/sprint_trades.csv")
    return df