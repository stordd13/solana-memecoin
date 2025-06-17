import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

RAW_BASE = Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_BASE = Path(__file__).parent.parent / 'data' / 'processed'
GROUPS = ['high_quality_tokens', 'dead_tokens', 'tokens_with_issues', 'tokens_with_gaps']

CLEANED_BASE = PROCESSED_BASE / 'cleaned'
CLEANED_BASE.mkdir(exist_ok=True)

# --- Cleaning Functions ---
def remove_initial_spikes(df: pd.DataFrame, n=2, threshold=10):
    """Remove first n minutes if price is >threshold x median of next 10 min and then drops >99%."""
    if len(df) < n+10:
        return df, []
    removed = []
    for i in range(n):
        window = df.iloc[i+1:i+11]['price']
        if window.median() == 0:
            continue
        if df.iloc[i]['price'] > threshold * window.median():
            # Check for >99% drop after spike
            if df.iloc[i+1]['price'] < 0.01 * df.iloc[i]['price']:
                removed.append(i)
    if removed:
        df = df.drop(df.index[removed]).reset_index(drop=True)
    return df, removed

def fill_gaps(df: pd.DataFrame):
    """Fill gaps according to README rules. Assumes datetime is sorted and in pandas.Timestamp."""
    df = df.sort_values('datetime').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    modifications = []
    # Drop or aggregate duplicate datetimes
    if df['datetime'].duplicated().any():
        dupes = df[df['datetime'].duplicated(keep=False)]['datetime'].tolist()
        modifications.append({'duplicate_datetimes_aggregated': dupes})
        df = df.groupby('datetime', as_index=False).agg({'price': 'mean'})
    full_range = pd.date_range(df['datetime'].iloc[0], df['datetime'].iloc[-1], freq='1min')
    df = df.set_index('datetime').reindex(full_range)
    # Find gaps
    is_gap = df['price'].isna()
    gap_starts = np.where((~is_gap[:-1]) & (is_gap[1:]))[0] + 1
    gap_ends = np.where((is_gap[:-1]) & (~is_gap[1:]))[0] + 1
    if is_gap.iloc[0]:
        gap_starts = np.insert(gap_starts, 0, 0)
    if is_gap.iloc[-1]:
        gap_ends = np.append(gap_ends, len(df))
    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start
        if gap_len == 1:
            df.iloc[start:end] = df.interpolate().iloc[start:end]
            modifications.append((start, end, 'linear'))
        elif 2 <= gap_len <= 5:
            df.iloc[start:end] = df.interpolate(method='polynomial', order=2).iloc[start:end]
            modifications.append((start, end, 'poly2'))
        elif 6 <= gap_len <= 10:
            df.iloc[start:end] = df.fillna(method='ffill').interpolate().iloc[start:end]
            modifications.append((start, end, 'ffill+linear'))
        else:
            modifications.append((start, end, 'flagged_long_gap'))
    df = df.reset_index().rename(columns={'index': 'datetime'})
    return df, modifications

def handle_extreme_jumps(df: pd.DataFrame, jump_threshold=100):
    """Replace single-minute returns > jump_threshold (10,000%) with local median if not part of a real pump/dump."""
    returns = df['price'].pct_change()
    modifications = []
    for i in range(1, len(df)):
        if abs(returns.iloc[i]) > jump_threshold:
            # Check if it's a 2-3 min anomaly or a real pump
            local = df['price'].iloc[max(0, i-5):min(len(df), i+6)]
            if (df['price'].iloc[i] > 10 * local.median()) or (df['price'].iloc[i] < 0.1 * local.median()):
                # Replace with median
                old = df['price'].iloc[i]
                df.at[i, 'price'] = local.median()
                modifications.append((i, old, 'extreme_jump_replaced'))
    return df, modifications

def handle_zero_negative_prices(df: pd.DataFrame):
    """Interpolate isolated zero/negative prices, mark as invalid if consecutive or >5%."""
    modifications = []
    zero_neg = df['price'] <= 0
    if zero_neg.sum() > 0.05 * len(df):
        modifications.append(('too_many_zero_neg', zero_neg.sum()))
        return None, modifications
    # Interpolate isolated
    df.loc[zero_neg, 'price'] = np.nan
    df['price'] = df['price'].interpolate()
    modifications.extend([(i, 'zero_neg_interpolated') for i in np.where(zero_neg)[0]])
    return df, modifications

def clean_token_file(parquet_path: Path, group: str) -> Dict:
    """Clean a single token file and return cleaning log."""
    df = pd.read_parquet(parquet_path)
    log = {'file': str(parquet_path), 'modifications': []}
    # Remove initial spikes
    df, removed = remove_initial_spikes(df)
    if removed:
        log['modifications'].append({'initial_spikes_removed': removed})
    # Fill gaps
    df, gap_mods = fill_gaps(df)
    if gap_mods:
        log['modifications'].append({'gaps_filled': gap_mods})
    # Handle extreme jumps
    df, jump_mods = handle_extreme_jumps(df)
    if jump_mods:
        log['modifications'].append({'extreme_jumps': jump_mods})
    # Handle zero/negative prices
    df, zero_mods = handle_zero_negative_prices(df)
    if zero_mods:
        log['modifications'].append({'zero_neg': zero_mods})
    if df is None:
        log['status'] = 'excluded_due_to_zero_neg'
        return log
    # Save cleaned file
    out_dir = CLEANED_BASE / group
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / parquet_path.name
    df.to_parquet(out_path, index=False)
    log['status'] = 'cleaned'
    log['output'] = str(out_path)
    return log

def clean_group(group: str):
    """Clean all tokens in a group folder."""
    in_dir = PROCESSED_BASE / group
    logs = []
    for file in in_dir.glob('*.parquet'):
        log = clean_token_file(file, group)
        logs.append(log)
    # Save cleaning log
    pd.DataFrame(logs).to_json(CLEANED_BASE / f'{group}_cleaning_log.json', orient='records', lines=True)

if __name__ == '__main__':
    # Example: clean high_quality_tokens group
    clean_group('tokens_with_gaps')
    print('Cleaning complete for high_quality_tokens. See logs in data/processed/cleaned/.') 