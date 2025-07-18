import polars as pl

def analyze_variability(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flags high variability/extremes but keeps them; adds CV for robustness (on 'returns').
    """
    return df.with_columns(
        (pl.col("returns").std().over("token_id") / (pl.col("returns").mean().over("token_id") + 1e-6)).alias("cv")
    )

def detect_death(df: pl.DataFrame, zero_threshold: int) -> pl.DataFrame:
    """
    Vectorized death detection (avg_price=0 for >threshold min).
    """
    return df.with_columns(
        pl.col("avg_price").eq(0).cum_sum().over("token_id").alias("zero_streak")
    ).with_columns(
        pl.when(pl.col("zero_streak") > zero_threshold).then(True).otherwise(False).alias("is_dead")
    )

def trim_post_death(df: pl.DataFrame) -> pl.DataFrame:
    """
    Trims constant avg_price=0 after death; keeps pre-death volatility.
    """
    return df.filter(~(pl.col("is_dead") & pl.col("avg_price").eq(0)))