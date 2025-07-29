import polars as pl
import os
import glob
import time
from utils import setup_logger
import config

logger = setup_logger(__name__)

def analyze_variability(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flags high variability/extremes but keeps them; adds CV for robustness (on 'returns').
    """
    return df.with_columns(
        (pl.col("returns").std().over("token_id") / (pl.col("returns").mean().over("token_id") + 1e-6)).alias("cv")
    )


logger = setup_logger(__name__)

def backward_raw_check(raw_dir: str, constant_threshold_min: int = config.ZERO_THRESHOLD) -> pl.DataFrame:
    """
    Backward death detection on raw Parquets: Loops files, detects constant price streak from end (>threshold min).
    Uses exact equality for constant (no tolerance, per spec). Falls back to row count if timestamps sparse.
    Outputs DF with token_id, death_time (start of constant), death_hours (from token start), max_streak (rows).
    Handles short/empty files as alive (death_hours=24.0).
    """
    start = time.time()
    all_files = glob.glob(os.path.join(raw_dir, '*.parquet'))
    death_summary = []
    
    for file_path in all_files:
        if os.path.basename(file_path).startswith('_tokens_list'):
            continue  # Skip metadata
        
        token_id = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pl.read_parquet(file_path)
            # Handle datetime: Cast if string, assume μs if Datetime
            if df.schema.get("datetime") == pl.Utf8:
                df = df.with_columns(pl.col("datetime").str.to_datetime().alias("datetime"))
            elif df.schema.get("datetime") != pl.Datetime:
                df = df.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))
            
            if df.height == 0:
                logger.debug(f"Token {token_id}: Empty file, assumed alive")
                death_summary.append({"token_id": token_id, "death_time": None, "death_hours": 24.0, "max_streak": 0})
                continue
            
            if df.height < 2:
                logger.debug(f"Token {token_id}: Too short (<2 rows), assumed alive")
                death_summary.append({"token_id": token_id, "death_time": None, "death_hours": 24.0, "max_streak": 0})
                continue
            
            # Sort ascending
            df = df.sort("datetime")
            
            # Get last price
            last_price = df["price"][-1]
            
            # Flag constant from end (exact ==, no eps)
            df = df.with_columns((pl.col("price") == last_price).alias("is_constant_from_end"))
            
            # Reverse and cum_sum changes (~constant) to find streak from end
            df_reversed = df.reverse()
            df_reversed = df_reversed.with_columns((~pl.col("is_constant_from_end")).cum_sum().alias("change_count"))
            constant_streak_mask = df_reversed["change_count"] == 0
            max_streak = constant_streak_mask.sum()
            
            if max_streak > 1:
                # Get streak subset
                df_reversed_constant = df_reversed.filter(constant_streak_mask)
                death_end_time = df_reversed_constant["datetime"][0]  # Latest (scalar)
                death_start_time = df_reversed_constant["datetime"][-1]  # Start of constant (scalar)
                
                # Duration: Prefer timestamp diff, fallback to rows-1 (assume ~1 min/row)
                streak_duration = death_end_time - death_start_time
                streak_duration_min = streak_duration.total_seconds() / 60 if streak_duration else (max_streak - 1)
                
                if streak_duration_min >= constant_threshold_min:
                    token_start_time = df["datetime"][0]  # scalar
                    death_duration = death_start_time - token_start_time
                    death_hours = death_duration.total_seconds() / 3600 if death_duration else 24.0
                    death_summary.append({
                        "token_id": token_id,
                        "death_time": death_start_time,
                        "death_hours": death_hours,
                        "max_streak": max_streak
                    })
                    logger.debug(f"Token {token_id}: Death at {death_hours:.2f} hours, streak={max_streak}, duration={streak_duration_min:.1f} min")
                else:
                    death_summary.append({"token_id": token_id, "death_time": None, "death_hours": 24.0, "max_streak": max_streak})
            else:
                death_summary.append({"token_id": token_id, "death_time": None, "death_hours": 24.0, "max_streak": 0})
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Create and save summary DF
    summary_df = pl.DataFrame(death_summary)
    summary_path = os.path.join(config.BOT_NEW_ROOT, 'data/processed/death_summary.parquet')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.write_parquet(summary_path)
    logger.info(f"Saved death summary: {summary_path} | Time: {time.time() - start:.2f}s | Dead tokens: {summary_df.filter(pl.col('death_time').is_not_null()).height}")
    return summary_df

def detect_death(df: pl.DataFrame, constant_threshold_min: int = 120) -> pl.DataFrame:
    """
    Vectorized death detection based on constant price (not zero) for >threshold min.
    """
    return df.with_columns([
        # Check if price is constant (not changing) - use small epsilon for comparison
        pl.col("avg_price").diff().abs().lt(1e-10).over("token_id").alias("is_constant")
    ]).with_columns([
        # Count consecutive constant prices
        pl.col("is_constant").cum_sum().over("token_id").alias("constant_streak")
    ]).with_columns([
        pl.when(pl.col("constant_streak") > constant_threshold_min).then(True).otherwise(False).alias("is_dead")
    ])

def trim_post_death(df: pl.DataFrame) -> pl.DataFrame:
    trimmed = df.filter(~(pl.col("is_dead") & pl.col("is_constant")))
    # Ensure min 1 row per dead token
    dead_tokens = df.filter(pl.col("is_dead")).select("token_id").unique()
    missing = dead_tokens.join(trimmed.select("token_id").unique(), on="token_id", how="anti")
    
    # Ensure we have a DataFrame (not LazyFrame) and check if there are missing tokens
    if isinstance(missing, pl.LazyFrame):
        missing = missing.collect()
    
    if missing.height > 0:
        # Get column names and types from the original dataframe
        cols = df.columns
        dtypes = df.dtypes
        
        # Create placeholders with proper columns
        placeholder_data = {}
        for col, dtype in zip(cols, dtypes):
            if col == "token_id":
                placeholder_data[col] = missing["token_id"]
            elif col == "datetime":
                placeholder_data[col] = [None] * missing.height
            elif dtype in [pl.Float32, pl.Float64]:
                placeholder_data[col] = [0.0] * missing.height
            elif dtype in [pl.Int32, pl.Int64]:
                placeholder_data[col] = [0] * missing.height
            elif dtype == pl.Boolean:
                placeholder_data[col] = [False] * missing.height
            else:
                placeholder_data[col] = [None] * missing.height
        
        placeholders = pl.DataFrame(placeholder_data)
        trimmed = pl.concat([trimmed, placeholders])
    return trimmed

# Standalone run for raw death detection
if __name__ == "__main__":
    backward_raw_check(config.DATA_RAW_DIR)