import polars as pl
import matplotlib.pyplot as plt
import os
import glob
import time
import sys

# Dynamic path fix for root and scripts/
bot_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(bot_root)
sys.path.append(os.path.join(bot_root, 'scripts'))
from utils import setup_logger
import config

logger = setup_logger(__name__)

def detect_death_raw(raw_dir: str, constant_threshold_min: int = 120) -> pl.DataFrame:
    """
    Backward death detection on raw Parquets: Loop files, sort by datetime, check for constant price streak from the end (>120 min).
    Outputs summary DF with death_time/hours/streak per token.
    """
    start = time.time()
    all_files = glob.glob(os.path.join(raw_dir, '*.parquet'))
    death_summary = []

    for file_path in all_files:
        if os.path.basename(file_path).startswith('_tokens_list'):
            continue  # Skip metadata

        token_id = os.path.splitext(os.path.basename(file_path))[0]
        try:
            # Read and ensure datetime is properly parsed
            df = pl.read_parquet(file_path)

            # Handle datetime conversion based on column type
            if df.schema["datetime"] == pl.Utf8:
                df = df.with_columns(pl.col("datetime").str.to_datetime().alias("datetime"))

            if df.height == 0:
                logger.debug(f"Token {token_id}: Empty file, skipping")
                continue

            # Sort by datetime ascending first
            df = df.sort('datetime')

            # Find the last price value
            last_price = df["price"][-1]

            # Create a boolean column for where price equals the last price
            df = df.with_columns(
                (pl.col("price") == last_price).alias("is_constant_from_end")
            )

            # Working backwards, find the first index where price differs from last price
            # We'll use reverse cumulative sum to find the constant streak from the end
            df_reversed = df.reverse()
            df_reversed = df_reversed.with_columns(
                (~pl.col("is_constant_from_end")).cum_sum().alias("change_count")
            )

            # Find where the constant streak starts (where change_count is still 0)
            constant_streak_mask = df_reversed["change_count"] == 0
            constant_streak_length = constant_streak_mask.sum()

            if constant_streak_length > 1:  # At least 2 rows for a streak
                # Get the datetime range of the constant streak
                df_reversed_constant = df_reversed.filter(constant_streak_mask)

                # Since we reversed, the last row in original is first in reversed
                death_end_time = df_reversed_constant["datetime"][0]  # End of data
                death_start_time = df_reversed_constant["datetime"][-1]  # Start of constant period

                # Calculate duration in minutes
                streak_duration_min = (death_end_time - death_start_time).total_seconds() / 60

                if streak_duration_min >= constant_threshold_min:
                    # Calculate hours from token start to death
                    token_start_time = df["datetime"][0]
                    death_hours = (death_start_time - token_start_time).total_seconds() / 3600

                    logger.debug(f"Token {token_id}: Died with {constant_streak_length} constant rows, "
                                 f"Duration={streak_duration_min:.1f} min, Death after {death_hours:.1f} hours")

                    death_summary.append({
                        "token_id": token_id,
                        "death_time": death_start_time,  # When constant period started
                        "death_hours": death_hours,
                        "constant_streak_length": constant_streak_length,
                        "streak_duration_min": streak_duration_min,
                        "final_price": last_price
                    })
                else:
                    logger.debug(f"Token {token_id}: Constant streak too short ({streak_duration_min:.1f} min < {constant_threshold_min} min)")
            else:
                logger.debug(f"Token {token_id}: No constant streak at end")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Create summary dataframe
    if death_summary:
        summary_df = pl.DataFrame(death_summary)
    else:
        # Empty dataframe with proper schema
        summary_df = pl.DataFrame({
            "token_id": [],
            "death_time": [],
            "death_hours": [],
            "constant_streak_length": [],
            "streak_duration_min": [],
            "final_price": []
        })

    # Save summary
    summary_path = os.path.join(bot_root, 'data/processed/death_summary.parquet')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.write_parquet(summary_path)

    logger.info(f"Death detection complete: Found {len(death_summary)} dead tokens out of {len(all_files)} files")
    logger.info(f"Saved death summary: {summary_path} | Time: {time.time() - start:.2f}s")

    return summary_df

def analyze_death_distribution(raw_dir: str, constant_threshold_min: int = 120) -> None:
    summary_df = detect_death_raw(raw_dir, constant_threshold_min)
    
    if summary_df.height == 0:
        logger.warning("No deaths detected—all tokens alive or short streaks.")
        return
    
    stats = summary_df.select(
        pl.col("death_hours").mean().alias("avg_death_hours"),
        pl.col("death_hours").median().alias("median_death_hours"),
        pl.col("death_hours").std().alias("std_death_hours")
    )
    print("Death Stats:", stats)
    logger.info(f"Death Stats: {stats.to_dicts()}")
    
    # Plot histogram
    death_hours = summary_df["death_hours"].to_numpy()
    plt.hist(death_hours, bins=50, range=(0, 24))
    plt.title("Token Death Distribution (Hours Till Death)")
    plt.xlabel("Hours")
    plt.ylabel("Number of Tokens")
    plot_path = os.path.join(bot_root, "analysis/figures/death_hist.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    analyze_death_distribution(config.DATA_RAW_DIR)  # Raw dataset path