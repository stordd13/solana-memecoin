import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_BASE = PROJECT_ROOT / 'data' / 'raw' / 'dataset'
PROCESSED_BASE = PROJECT_ROOT / 'data' / 'processed'
CLEANED_BASE = PROJECT_ROOT / 'data' / 'cleaned'
CLEANED_BASE.mkdir(exist_ok=True)

# New category-aware folder structure
CATEGORIES = {
    'normal_behavior_tokens': 'gentle',     # Preserve natural volatility
    'dead_tokens': 'minimal',               # Basic cleaning only
    'tokens_with_extremes': 'preserve',     # Keep extreme movements
    'tokens_with_gaps': 'aggressive'        # Fill gaps aggressively
}

class CategoryAwareTokenCleaner:
    """
    Advanced token cleaner that applies different strategies based on token category
    and distinguishes between data artifacts and legitimate market movements
    """
    
    def __init__(self):
        self.cleaning_strategies = {
            'gentle': self._gentle_cleaning,
            'minimal': self._minimal_cleaning, 
            'preserve': self._preserve_extremes_cleaning,
            'aggressive': self._aggressive_cleaning
        }
        
        # Thresholds for different types of anomalies
        self.artifact_thresholds = {
            'listing_spike_multiplier': 20,     # 20x median for listing artifacts
            'listing_drop_threshold': 0.99,     # 99% drop after spike
            'data_error_threshold': 1000,       # 100,000% (obvious data errors)
            'flash_crash_recovery': 0.95,       # 95% recovery within 5 minutes
            'zero_volume_threshold': 0.001      # Minimum volume for valid trades
        }
        
        # Market behavior thresholds (preserve these!)
        self.market_thresholds = {
            'max_realistic_pump': 50,           # 5,000% pumps can be real
            'max_realistic_dump': 0.95,         # 95% dumps can be real
            'sustained_movement_minutes': 3,    # Real movements last >3 minutes
            'volume_confirmation_ratio': 0.1    # Volume should support price moves
        }

    def clean_token_file(self, token_path: Path, category: str) -> Dict:
        """
        Clean a single token file using category-appropriate strategy
        
        Args:
            token_path: Path to the token parquet file
            category: Token category (normal_behavior_tokens, dead_tokens, etc.)
            
        Returns:
            Dictionary with cleaning log and results
        """
        try:
            # Load data
            df = pd.read_parquet(token_path)
            token_name = token_path.stem
            
            log = {
                'token': token_name,
                'category': category,
                'original_rows': len(df),
                'modifications': [],
                'strategy_used': CATEGORIES.get(category, 'gentle'),
                'status': 'processing'
            }
            
            # Validate data structure
            if not self._validate_data_structure(df):
                log['status'] = 'invalid_data_structure'
                return log
            
            # Apply category-specific cleaning strategy
            strategy = CATEGORIES.get(category, 'gentle')
            df_cleaned, modifications = self.cleaning_strategies[strategy](df, token_name)
            
            if df_cleaned is None:
                log['status'] = 'excluded_due_to_severe_issues'
                log['modifications'] = modifications
                return log
            
            # Save cleaned file
            output_dir = CLEANED_BASE / category
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{token_name}.parquet"
            
            df_cleaned.to_parquet(output_path, index=False)
            
            log.update({
                'final_rows': len(df_cleaned),
                'rows_removed': len(df) - len(df_cleaned),
                'modifications': modifications,
                'output_path': str(output_path),
                'status': 'cleaned_successfully'
            })
            
            return log
            
        except Exception as e:
            logger.error(f"Error cleaning {token_path}: {e}")
            return {
                'token': token_path.stem,
                'category': category,
                'status': 'error',
                'error': str(e)
            }

    def _validate_data_structure(self, df: pd.DataFrame) -> bool:
        """Validate that the dataframe has required columns and structure"""
        required_columns = ['datetime', 'price']
        return all(col in df.columns for col in required_columns)

    def _gentle_cleaning(self, df: pd.DataFrame, token_name: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Gentle cleaning for normal behavior tokens
        - Remove obvious data artifacts only
        - Preserve natural volatility and legitimate pumps
        - Fill small gaps conservatively
        """
        modifications = []
        df = df.copy()
        
        # 1. Remove obvious listing artifacts (but preserve real pumps)
        df, listing_mods = self._remove_listing_artifacts(df)
        if listing_mods:
            modifications.extend(listing_mods)
        
        # 2. Handle severe data errors (>100,000% moves that are clearly wrong)
        df, error_mods = self._fix_data_errors(df)
        if error_mods:
            modifications.extend(error_mods)
        
        # 3. Fill small gaps only (1-2 minutes) with linear interpolation
        df, gap_mods = self._fill_small_gaps_only(df)
        if gap_mods:
            modifications.extend(gap_mods)
        
        # 4. Handle zero/negative prices conservatively
        df, price_mods = self._handle_invalid_prices_conservative(df)
        if price_mods:
            modifications.extend(price_mods)
        
        return df, modifications

    def _minimal_cleaning(self, df: pd.DataFrame, token_name: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Minimal cleaning for dead tokens
        - Only fix critical data errors
        - Don't bother with gaps or minor issues since token is inactive
        """
        modifications = []
        df = df.copy()
        
        # Only fix severe data errors and invalid prices
        df, error_mods = self._fix_data_errors(df)
        if error_mods:
            modifications.extend(error_mods)
            
        df, price_mods = self._handle_invalid_prices_conservative(df)
        if price_mods:
            modifications.extend(price_mods)
        
        return df, modifications

    def _preserve_extremes_cleaning(self, df: pd.DataFrame, token_name: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Preserve extremes cleaning for tokens with extreme movements
        - Keep ALL legitimate extreme movements (this is their defining characteristic!)
        - Only remove obvious data corruption
        - Be very conservative about what constitutes an "error"
        """
        modifications = []
        df = df.copy()
        
        # Only remove the most obvious data corruption (not extreme movements!)
        # 1. Remove only impossible values (negative prices, exact zeros from data errors)
        df, price_mods = self._handle_impossible_values_only(df)
        if price_mods:
            modifications.extend(price_mods)
        
        # 2. Fix only extreme data errors (>1,000,000% in one minute - clearly data corruption)
        df, corruption_mods = self._fix_extreme_data_corruption(df)
        if corruption_mods:
            modifications.extend(corruption_mods)
        
        # 3. Fill only critical gaps that break data continuity
        df, gap_mods = self._fill_critical_gaps_only(df)
        if gap_mods:
            modifications.extend(gap_mods)
        
        return df, modifications

    def _aggressive_cleaning(self, df: pd.DataFrame, token_name: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Aggressive cleaning for tokens with gaps
        - Fill all reasonable gaps
        - Clean data quality issues thoroughly
        - Still preserve legitimate market movements
        """
        modifications = []
        df = df.copy()
        
        # 1. Remove listing artifacts
        df, listing_mods = self._remove_listing_artifacts(df)
        if listing_mods:
            modifications.extend(listing_mods)
        
        # 2. Fill gaps aggressively (main issue for these tokens)
        df, gap_mods = self._fill_gaps_comprehensive(df)
        if gap_mods:
            modifications.extend(gap_mods)
        
        # 3. Fix data errors
        df, error_mods = self._fix_data_errors(df)
        if error_mods:
            modifications.extend(error_mods)
        
        # 4. Handle invalid prices
        df, price_mods = self._handle_invalid_prices_conservative(df)
        if price_mods:
            modifications.extend(price_mods)
        
        return df, modifications

    def _remove_listing_artifacts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Remove obvious listing artifacts while preserving real explosive launches
        
        Criteria for listing artifacts:
        - First few minutes only
        - Price >20x median of next 10 minutes  
        - Followed by >99% drop immediately
        - No volume or trading activity confirmation
        """
        if len(df) < 15:  # Need enough data to analyze
            return df, []
        
        modifications = []
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Check first 3 minutes for listing artifacts
        for i in range(min(3, len(df) - 10)):
            current_price = df.iloc[i]['price']
            
            # Get median of next 10 minutes
            window = df.iloc[i+1:i+11]['price']
            if len(window) == 0 or window.median() == 0:
                continue
                
            price_ratio = current_price / window.median()
            
            # Check if this looks like a listing artifact
            if price_ratio > self.artifact_thresholds['listing_spike_multiplier']:
                # Check for immediate crash (listing artifact signature)
                if i + 1 < len(df):
                    next_price = df.iloc[i + 1]['price']
                    drop_ratio = (current_price - next_price) / current_price
                    
                    if drop_ratio > self.artifact_thresholds['listing_drop_threshold']:
                        # This looks like a listing artifact - remove it
                        df = df.drop(df.index[i]).reset_index(drop=True)
                        modifications.append({
                            'type': 'listing_artifact_removed',
                            'position': i,
                            'price_ratio': price_ratio,
                            'drop_ratio': drop_ratio,
                            'reason': 'Obvious listing artifact: extreme spike followed by immediate crash'
                        })
                        break  # Only remove one at a time to avoid index issues
        
        return df, modifications

    def _fix_data_errors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Fix obvious data errors while preserving legitimate extreme movements
        
        Data errors vs legitimate movements:
        - Data error: Single point >100,000% different from surrounding prices
        - Legitimate: Sustained movement with volume/context
        """
        modifications = []
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        
        # Find potential data errors (>100,000% single-minute moves)
        error_threshold = self.artifact_thresholds['data_error_threshold']
        potential_errors = df[abs(df['returns']) > error_threshold].index
        
        for idx in potential_errors:
            if idx == 0 or idx == len(df) - 1:
                continue  # Skip first/last points
            
            current_price = df.loc[idx, 'price']
            prev_price = df.loc[idx - 1, 'price']
            next_price = df.loc[idx + 1, 'price']
            
            # Check if this is an isolated error (price reverts immediately)
            prev_ratio = abs((current_price - prev_price) / prev_price)
            next_ratio = abs((next_price - current_price) / current_price)
            
            # If both moves are extreme and price reverts, it's likely a data error
            if (prev_ratio > error_threshold and next_ratio > 0.9):
                # Replace with interpolated value
                df.loc[idx, 'price'] = (prev_price + next_price) / 2
                modifications.append({
                    'type': 'data_error_corrected',
                    'position': idx,
                    'original_price': current_price,
                    'corrected_price': df.loc[idx, 'price'],
                    'reason': 'Isolated extreme price point (likely data error)'
                })
        
        # Recalculate returns after corrections
        df['returns'] = df['price'].pct_change()
        df = df.drop('returns', axis=1)
        
        return df, modifications

    def _fix_extreme_data_corruption(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Fix only the most extreme data corruption (>1,000,000% moves)
        Used for extreme tokens to preserve their legitimate extreme movements
        """
        modifications = []
        df = df.copy()
        
        # Only fix moves >1,000,000% (10,000x) - these are definitely data errors
        corruption_threshold = 10000  # 1,000,000%
        df['returns'] = df['price'].pct_change()
        
        corruption_points = df[abs(df['returns']) > corruption_threshold].index
        
        for idx in corruption_points:
            if idx == 0 or idx == len(df) - 1:
                continue
            
            # Replace with local median (very conservative)
            local_window = df.iloc[max(0, idx-2):min(len(df), idx+3)]['price']
            df.loc[idx, 'price'] = local_window.median()
            
            modifications.append({
                'type': 'extreme_corruption_fixed',
                'position': idx,
                'reason': 'Data corruption >1,000,000% detected'
            })
        
        df = df.drop('returns', axis=1)
        return df, modifications

    def _fill_small_gaps_only(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fill only small gaps (1-2 minutes) with linear interpolation"""
        return self._fill_gaps_with_size_limit(df, max_gap_minutes=2)

    def _fill_critical_gaps_only(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fill only critical gaps that break data continuity"""
        return self._fill_gaps_with_size_limit(df, max_gap_minutes=1)

    def _fill_gaps_comprehensive(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fill gaps comprehensively for tokens where gaps are the main issue"""
        return self._fill_gaps_with_size_limit(df, max_gap_minutes=10)

    def _fill_gaps_with_size_limit(self, df: pd.DataFrame, max_gap_minutes: int) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Fill gaps up to specified size limit with appropriate interpolation
        """
        modifications = []
        df = df.copy().sort_values('datetime').reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Handle duplicates first
        if df['datetime'].duplicated().any():
            duplicates = df[df['datetime'].duplicated(keep=False)]['datetime'].tolist()
            df = df.groupby('datetime', as_index=False).agg({'price': 'mean'})
            modifications.append({
                'type': 'duplicates_aggregated',
                'count': len(duplicates),
                'method': 'mean_price'
            })
        
        # Create complete time range
        full_range = pd.date_range(
            df['datetime'].iloc[0], 
            df['datetime'].iloc[-1], 
            freq='1min'
        )
        
        df = df.set_index('datetime').reindex(full_range)
        
        # Find and fill gaps
        is_gap = df['price'].isna()
        if not is_gap.any():
            df = df.reset_index().rename(columns={'index': 'datetime'})
            return df, modifications
        
        # Identify gap segments
        gap_starts = np.where((~is_gap[:-1]) & (is_gap[1:]))[0] + 1
        gap_ends = np.where((is_gap[:-1]) & (~is_gap[1:]))[0] + 1
        
        if is_gap.iloc[0]:
            gap_starts = np.insert(gap_starts, 0, 0)
        if is_gap.iloc[-1]:
            gap_ends = np.append(gap_ends, len(df))
        
        # Fill gaps based on size
        for start, end in zip(gap_starts, gap_ends):
            gap_size = end - start
            
            if gap_size <= max_gap_minutes:
                if gap_size == 1:
                    # Linear interpolation for 1-minute gaps
                    df.iloc[start:end] = df.interpolate().iloc[start:end]
                    method = 'linear'
                elif gap_size <= 3:
                    # Linear interpolation for small gaps
                    df.iloc[start:end] = df.interpolate().iloc[start:end]
                    method = 'linear'
                elif gap_size <= 6:
                    # Polynomial for medium gaps
                    df.iloc[start:end] = df.interpolate(method='polynomial', order=2).iloc[start:end]
                    method = 'polynomial'
                else:
                    # Forward fill + linear for larger gaps
                    df.iloc[start:end] = df.fillna(method='ffill').interpolate().iloc[start:end]
                    method = 'ffill_linear'
                
                modifications.append({
                    'type': 'gap_filled',
                    'start': start,
                    'end': end,
                    'size_minutes': gap_size,
                    'method': method
                })
            else:
                modifications.append({
                    'type': 'gap_too_large',
                    'start': start,
                    'end': end,
                    'size_minutes': gap_size,
                    'action': 'left_unfilled'
                })
        
        df = df.reset_index().rename(columns={'index': 'datetime'})
        return df, modifications

    def _handle_invalid_prices_conservative(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Handle zero/negative prices conservatively"""
        modifications = []
        
        # Find invalid prices
        invalid_mask = df['price'] <= 0
        invalid_count = invalid_mask.sum()
        
        if invalid_count == 0:
            return df, modifications
        
        # If >5% of data is invalid, flag for exclusion
        if invalid_count / len(df) > 0.05:
            modifications.append({
                'type': 'too_many_invalid_prices',
                'invalid_count': invalid_count,
                'total_count': len(df),
                'percentage': (invalid_count / len(df)) * 100
            })
            return None, modifications
        
        # Interpolate isolated invalid prices
        df.loc[invalid_mask, 'price'] = np.nan
        df['price'] = df['price'].interpolate()
        
        modifications.append({
            'type': 'invalid_prices_interpolated',
            'count': invalid_count,
            'method': 'linear_interpolation'
        })
        
        return df, modifications

    def _handle_impossible_values_only(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Handle only impossible values (negative prices, exact zeros from data errors)"""
        modifications = []
        
        # Only fix impossible values, not just low prices
        impossible_mask = (df['price'] < 0) | (df['price'] == 0)
        impossible_count = impossible_mask.sum()
        
        if impossible_count == 0:
            return df, modifications
        
        # If >10% of data is impossible, flag for exclusion (higher threshold for extreme tokens)
        if impossible_count / len(df) > 0.10:
            modifications.append({
                'type': 'too_many_impossible_values',
                'impossible_count': impossible_count,
                'total_count': len(df)
            })
            return None, modifications
        
        # Interpolate impossible values only
        df.loc[impossible_mask, 'price'] = np.nan
        df['price'] = df['price'].interpolate()
        
        modifications.append({
            'type': 'impossible_values_fixed',
            'count': impossible_count
        })
        
        return df, modifications

def clean_category(category: str, limit: Optional[int] = None) -> Dict:
    """
    Clean all tokens in a specific category folder
    
    Args:
        category: Category name (e.g., 'normal_behavior_tokens')
        limit: Optional limit on number of files to process
        
    Returns:
        Summary statistics of cleaning process
    """
    cleaner = CategoryAwareTokenCleaner()
    category_folder = PROCESSED_BASE / category
    
    if not category_folder.exists():
        logger.error(f"Category folder {category_folder} does not exist")
        return {'error': f"Category folder {category} not found"}
    
    # Get all parquet files
    parquet_files = list(category_folder.glob('*.parquet'))
    if limit:
        parquet_files = parquet_files[:limit]
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {category_folder}")
        return {'warning': f"No files found in {category}"}
    
    # Process files
    results = []
    success_count = 0
    error_count = 0
    
    logger.info(f"Cleaning {len(parquet_files)} tokens from {category}")
    
    for i, file_path in enumerate(parquet_files):
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(parquet_files)} files")
        
        result = cleaner.clean_token_file(file_path, category)
        results.append(result)
        
        if result['status'] == 'cleaned_successfully':
            success_count += 1
        else:
            error_count += 1
    
    # Save cleaning log
    log_df = pd.DataFrame(results)
    log_path = CLEANED_BASE / f'{category}_cleaning_log.json'
    log_df.to_json(log_path, orient='records', lines=True)
    
    summary = {
        'category': category,
        'total_files': len(parquet_files),
        'successfully_cleaned': success_count,
        'errors': error_count,
        'success_rate': (success_count / len(parquet_files)) * 100,
        'log_saved_to': str(log_path),
        'strategy_used': CATEGORIES.get(category, 'gentle')
    }
    
    logger.info(f"Cleaning complete for {category}: {success_count}/{len(parquet_files)} successful")
    return summary

def clean_all_categories(limit_per_category: Optional[int] = None) -> Dict:
    """
    Clean all token categories with appropriate strategies
    
    Args:
        limit_per_category: Optional limit on files per category
        
    Returns:
        Summary of all cleaning operations
    """
    logger.info("Starting category-aware cleaning for all token categories")
    
    all_results = {}
    total_success = 0
    total_files = 0
    
    for category in CATEGORIES.keys():
        logger.info(f"\n{'='*50}")
        logger.info(f"Cleaning category: {category}")
        logger.info(f"Strategy: {CATEGORIES[category]}")
        logger.info(f"{'='*50}")
        
        result = clean_category(category, limit_per_category)
        all_results[category] = result
        
        if 'total_files' in result:
            total_files += result['total_files']
            total_success += result.get('successfully_cleaned', 0)
    
    # Overall summary
    overall_summary = {
        'total_categories': len(CATEGORIES),
        'total_files_processed': total_files,
        'total_successfully_cleaned': total_success,
        'overall_success_rate': (total_success / total_files * 100) if total_files > 0 else 0,
        'category_results': all_results,
        'cleaning_strategies_used': CATEGORIES
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL CLEANING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Successfully cleaned: {total_success}")
    logger.info(f"Overall success rate: {overall_summary['overall_success_rate']:.1f}%")
    logger.info(f"Cleaned files saved to: {CLEANED_BASE}")
    
    return overall_summary

if __name__ == '__main__':
    # Example usage - clean all categories
    summary = clean_all_categories(limit_per_category=None)
    
    # Save overall summary
    summary_path = CLEANED_BASE / 'overall_cleaning_summary.json'
    pd.Series(summary).to_json(summary_path, indent=2)
    logger.info(f"Overall summary saved to: {summary_path}") 