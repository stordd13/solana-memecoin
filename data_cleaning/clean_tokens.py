import os
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

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
            df = pl.read_parquet(token_path)
            token_name = token_path.stem
            
            log = {
                'token': token_name,
                'category': category,
                'original_rows': df.height,
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
            
            df_cleaned.write_parquet(output_path)
            
            log.update({
                'final_rows': df_cleaned.height,
                'rows_removed': df.height - df_cleaned.height,
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

    def _validate_data_structure(self, df: pl.DataFrame) -> bool:
        """Validate that the dataframe has required columns and structure"""
        required_columns = ['datetime', 'price']
        return all(col in df.columns for col in required_columns)

    def _gentle_cleaning(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Gentle cleaning for normal behavior tokens
        - Remove obvious data artifacts only
        - Preserve natural volatility and legitimate pumps
        - Fill small gaps conservatively
        """
        modifications = []
        df = df.clone()
        
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

    def _minimal_cleaning(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Minimal cleaning for dead tokens
        - Only fix critical data errors
        - Don't bother with gaps or minor issues since token is inactive
        """
        modifications = []
        df = df.clone()
        
        # Only fix severe data errors and invalid prices
        df, error_mods = self._fix_data_errors(df)
        if error_mods:
            modifications.extend(error_mods)
            
        df, price_mods = self._handle_invalid_prices_conservative(df)
        if price_mods:
            modifications.extend(price_mods)
        
        return df, modifications

    def _preserve_extremes_cleaning(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Preserve extremes cleaning for tokens with extreme movements
        - Keep ALL legitimate extreme movements (this is their defining characteristic!)
        - Only remove obvious data corruption
        - Be very conservative about what constitutes an "error"
        """
        modifications = []
        df = df.clone()
        
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

    def _aggressive_cleaning(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Aggressive cleaning for tokens with gaps
        - Fill all reasonable gaps
        - Clean data quality issues thoroughly
        - Still preserve legitimate market movements
        """
        modifications = []
        df = df.clone()
        
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

    def _remove_listing_artifacts(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Remove obvious listing artifacts while preserving legitimate pumps
        """
        modifications = []
        
        if df.height < 10:
            return df, modifications
        
        # Sort by datetime
        df = df.sort('datetime')
        
        # Calculate returns
        df = df.with_columns([
            pl.col('price').pct_change().alias('returns')
        ])
        
        # Detect potential listing artifacts
        # 1. Initial spike followed by immediate crash (classic listing artifact)
        median_price = df.select(pl.col('price').median()).item()
        
        # Look for initial spikes >20x median in first 10 minutes
        first_10_rows = df.head(10)
        spike_mask = first_10_rows['price'] > (median_price * self.artifact_thresholds['listing_spike_multiplier'])
        
        if spike_mask.any():
            # Check if followed by >99% drop within next few minutes
            spike_indices = first_10_rows.with_row_count().filter(spike_mask)['row_nr'].to_list()
            
            for spike_idx in spike_indices:
                if spike_idx < df.height - 3:  # Ensure we have data after spike
                    spike_price = df.row(spike_idx, named=True)['price']
                    next_few_prices = df.slice(spike_idx+1, 3)['price']
                    
                    # Check if price drops >99% after spike
                    min_after_spike = next_few_prices.min()
                    if min_after_spike < spike_price * (1 - self.artifact_thresholds['listing_drop_threshold']):
                        # This is likely a listing artifact
                        spike_datetime = df.row(spike_idx, named=True)['datetime']
                        df = df.filter(pl.col('datetime') != spike_datetime)
                        modifications.append({
                            'type': 'listing_artifact_removed',
                            'index': spike_idx,
                            'price': spike_price,
                            'reason': 'spike_followed_by_crash'
                        })
        
        return df, modifications

    def _fix_data_errors(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Fix obvious data errors (>100,000% moves in one minute)
        """
        modifications = []
        
        if df.height < 2:
            return df, modifications
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df = df.with_columns([
                pl.col('price').pct_change().alias('returns')
            ])
        
        # Find extreme data errors (>100,000% = 1000x in one minute)
        error_threshold = self.artifact_thresholds['data_error_threshold']
        extreme_returns = df.filter(pl.col('returns').abs() > error_threshold)
        
        if extreme_returns.height > 0:
            # Remove these extreme outliers by interpolating
            error_indices = extreme_returns.with_row_count()['row_nr'].to_list()
            
            # Replace extreme values with interpolated values
            df = df.with_row_count().with_columns([
                pl.when(pl.col('row_nr').is_in(error_indices))
                .then(None)
                .otherwise(pl.col('price'))
                .alias('price_clean')
            ])
            
            # Linear interpolation for missing values
            df = df.with_columns([
                pl.col('price_clean').interpolate().alias('price')
            ]).drop(['price_clean', 'row_nr'])
            
            modifications.append({
                'type': 'extreme_data_errors_fixed',
                'count': len(error_indices),
                'threshold_used': error_threshold,
                'method': 'linear_interpolation'
            })
        
        return df, modifications

    def _fix_extreme_data_corruption(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Fix only extreme data corruption (>1,000,000% moves)
        """
        modifications = []
        
        if df.height < 2:
            return df, modifications
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df = df.with_columns([
                pl.col('price').pct_change().alias('returns')
            ])
        
        # Only fix absolutely extreme corruption (>1,000,000%)
        corruption_threshold = 10000  # 1,000,000%
        extreme_corruption = df.filter(pl.col('returns').abs() > corruption_threshold)
        
        if extreme_corruption.height > 0:
            # Remove these corrupted points
            corruption_indices = extreme_corruption.with_row_count()['row_nr'].to_list()
            
            df = df.with_row_count().with_columns([
                pl.when(pl.col('row_nr').is_in(corruption_indices))
                .then(None)
                .otherwise(pl.col('price'))
                .alias('price_clean')
            ])
            
            df = df.with_columns([
                pl.col('price_clean').interpolate().alias('price')
            ]).drop(['price_clean', 'row_nr'])
            
            modifications.append({
                'type': 'extreme_data_corruption_fixed',
                'count': len(corruption_indices),
                'threshold_used': corruption_threshold
            })
        
        return df, modifications

    def _fill_small_gaps_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """Fill only small gaps (1-2 minutes) with linear interpolation"""
        return self._fill_gaps_with_size_limit(df, max_gap_minutes=2)

    def _fill_critical_gaps_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """Fill only critical gaps (1-3 minutes) for extreme tokens"""
        return self._fill_gaps_with_size_limit(df, max_gap_minutes=3)

    def _fill_gaps_comprehensive(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """Fill gaps comprehensively for tokens where gaps are the main issue"""
        return self._fill_gaps_with_size_limit(df, max_gap_minutes=10)

    def _fill_gaps_with_size_limit(self, df: pl.DataFrame, max_gap_minutes: int) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Fill gaps up to specified size limit with appropriate interpolation
        """
        modifications = []
        df = df.sort('datetime')
        
        # Ensure datetime is properly parsed
        if df['datetime'].dtype != pl.Datetime:
            df = df.with_columns([
                pl.col('datetime').str.strptime(pl.Datetime).alias('datetime')
            ])
        
        # Handle duplicates first
        duplicates_count = df.group_by('datetime').len().filter(pl.col('len') > 1).height
        if duplicates_count > 0:
            df = df.group_by('datetime').agg([
                pl.col('price').mean().alias('price')
            ]).sort('datetime')
            modifications.append({
                'type': 'duplicates_aggregated',
                'count': duplicates_count,
                'method': 'mean_price'
            })
        
        # Create complete time range using Polars
        start_time = df.select(pl.col('datetime').min()).item()
        end_time = df.select(pl.col('datetime').max()).item()
        
        # Generate minute-by-minute range
        time_range = pl.datetime_range(
            start_time, 
            end_time, 
            interval='1m',
            eager=True
        )
        
        # Create complete DataFrame and join
        complete_df = pl.DataFrame({'datetime': time_range})
        df = complete_df.join(df, on='datetime', how='left')
        
        # Find gaps
        null_mask = df['price'].is_null()
        if not null_mask.any():
            return df, modifications
        
        # Identify gap segments using numpy for efficiency
        null_array = null_mask.to_numpy()
        gap_starts = np.where((~null_array[:-1]) & (null_array[1:]))[0] + 1
        gap_ends = np.where((null_array[:-1]) & (~null_array[1:]))[0] + 1
        
        if null_array[0]:
            gap_starts = np.insert(gap_starts, 0, 0)
        if null_array[-1]:
            gap_ends = np.append(gap_ends, len(df))
        
        # Fill gaps based on size
        for start, end in zip(gap_starts, gap_ends):
            gap_size = end - start
            
            if gap_size <= max_gap_minutes:
                # Use linear interpolation for all gap sizes
                df = df.with_columns([
                    pl.col('price').interpolate().alias('price')
                ])
                
                modifications.append({
                    'type': 'gap_filled',
                    'start': int(start),
                    'end': int(end),
                    'size_minutes': int(gap_size),
                    'method': 'linear'
                })
            else:
                modifications.append({
                    'type': 'gap_too_large',
                    'start': int(start),
                    'end': int(end),
                    'size_minutes': int(gap_size),
                    'action': 'left_unfilled'
                })
        
        return df, modifications

    def _handle_invalid_prices_conservative(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """Handle zero/negative prices conservatively"""
        modifications = []
        
        # Find invalid prices
        invalid_mask = df['price'] <= 0
        invalid_count = invalid_mask.sum()
        
        if invalid_count == 0:
            return df, modifications
        
        # If >5% of data is invalid, flag for exclusion
        if invalid_count / df.height > 0.05:
            modifications.append({
                'type': 'too_many_invalid_prices',
                'invalid_count': invalid_count,
                'total_count': df.height,
                'percentage': (invalid_count / df.height) * 100
            })
            return None, modifications
        
        # Interpolate isolated invalid prices
        df = df.with_columns([
            pl.when(pl.col('price') <= 0)
            .then(None)
            .otherwise(pl.col('price'))
            .alias('price')
        ]).with_columns([
            pl.col('price').interpolate().alias('price')
        ])
        
        modifications.append({
            'type': 'invalid_prices_interpolated',
            'count': invalid_count,
            'method': 'linear_interpolation'
        })
        
        return df, modifications

    def _handle_impossible_values_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """Handle only impossible values (negative prices, exact zeros from data errors)"""
        modifications = []
        
        # Only fix impossible values, not just low prices
        impossible_mask = (df['price'] < 0) | (df['price'] == 0)
        impossible_count = impossible_mask.sum()
        
        if impossible_count == 0:
            return df, modifications
        
        # If >10% of data is impossible, flag for exclusion (higher threshold for extreme tokens)
        if impossible_count / df.height > 0.10:
            modifications.append({
                'type': 'too_many_impossible_values',
                'impossible_count': impossible_count,
                'total_count': df.height
            })
            return None, modifications
        
        # Interpolate impossible values only
        df = df.with_columns([
            pl.when((pl.col('price') < 0) | (pl.col('price') == 0))
            .then(None)
            .otherwise(pl.col('price'))
            .alias('price')
        ]).with_columns([
            pl.col('price').interpolate().alias('price')
        ])
        
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
    
    # Save cleaning log using Polars
    log_df = pl.DataFrame(results)
    log_path = CLEANED_BASE / f'{category}_cleaning_log.json'
    log_df.write_json(log_path)
    
    summary = {
        'category': category,
        'total_files_processed': len(parquet_files),
        'successfully_cleaned': success_count,
        'errors': error_count,
        'success_rate': (success_count / len(parquet_files)) * 100 if parquet_files else 0,
        'cleaning_log_path': str(log_path)
    }
    
    logger.info(f"Category {category} cleaning complete: {success_count}/{len(parquet_files)} files cleaned successfully")
    return summary

def clean_all_categories(limit_per_category: Optional[int] = None) -> Dict:
    """
    Clean all categories with their appropriate strategies
    
    Args:
        limit_per_category: Optional limit on number of files to process per category
        
    Returns:
        Summary statistics of entire cleaning process
    """
    logger.info("Starting comprehensive cleaning of all categories")
    
    results = {}
    total_files = 0
    total_success = 0
    total_errors = 0
    
    for category in CATEGORIES.keys():
        logger.info(f"\n--- Cleaning category: {category} ---")
        result = clean_category(category, limit_per_category)
        results[category] = result
        
        if 'total_files_processed' in result:
            total_files += result['total_files_processed']
            total_success += result['successfully_cleaned']
            total_errors += result['errors']
    
    # Overall summary
    overall_summary = {
        'total_files_processed': total_files,
        'total_successfully_cleaned': total_success,
        'total_errors': total_errors,
        'overall_success_rate': (total_success / total_files) * 100 if total_files > 0 else 0,
        'category_results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save overall summary using Polars
    summary_path = CLEANED_BASE / 'cleaning_summary.json'
    pl.DataFrame([overall_summary]).write_json(summary_path)
    
    logger.info(f"\n=== CLEANING COMPLETE ===")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Successfully cleaned: {total_success}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Success rate: {overall_summary['overall_success_rate']:.1f}%")
    logger.info(f"Summary saved to: {summary_path}")
    
    return overall_summary

def clean_category_with_gap_investigation(category: str, 
                                        investigation_results: Optional[Dict] = None,
                                        limit: Optional[int] = None) -> Dict:
    """
    Clean a category with intelligent gap handling based on investigation results
    
    Args:
        category: Category name
        investigation_results: Results from gap investigation to guide cleaning decisions
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
    
    # Determine enhanced cleaning strategy based on investigation
    if investigation_results and 'recommendations' in investigation_results:
        recommendations = investigation_results['recommendations']
        
        # Override cleaning strategy based on gap investigation
        enhanced_strategies = {}
        
        # Tokens recommended for removal - skip cleaning
        skip_tokens = set()
        if 'remove_completely' in recommendations:
            skip_tokens.update(recommendations['remove_completely'])
        
        # Tokens that need manual review - use gentle cleaning
        gentle_tokens = set()
        if 'needs_manual_review' in recommendations:
            gentle_tokens.update(recommendations['needs_manual_review'])
        
        # Tokens that can be kept and cleaned - use appropriate strategy
        clean_tokens = set()
        if 'keep_and_clean' in recommendations:
            clean_tokens.update(recommendations['keep_and_clean'])
        
        logger.info(f"Gap investigation guided strategy:")
        logger.info(f"  Skip cleaning: {len(skip_tokens)} tokens")
        logger.info(f"  Gentle clean: {len(gentle_tokens)} tokens")
        logger.info(f"  Standard clean: {len(clean_tokens)} tokens")
    else:
        skip_tokens = set()
        gentle_tokens = set()
        clean_tokens = set()
    
    # Process files with enhanced strategy
    results = []
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    logger.info(f"Cleaning {len(parquet_files)} tokens from {category} with gap investigation guidance")
    
    for i, file_path in enumerate(parquet_files):
        token_name = file_path.stem
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(parquet_files)} files")
        
        # Check if token should be skipped based on investigation
        if token_name in skip_tokens:
            results.append({
                'token': token_name,
                'category': category,
                'status': 'skipped_per_investigation',
                'reason': 'recommended_for_removal'
            })
            skipped_count += 1
            continue
        
        # Apply appropriate cleaning strategy
        if token_name in gentle_tokens:
            # Force gentle cleaning regardless of category
            original_strategy = cleaner.cleaning_strategies[CATEGORIES.get(category, 'gentle')]
            cleaner.cleaning_strategies[CATEGORIES.get(category, 'gentle')] = cleaner._gentle_cleaning
            
        result = cleaner.clean_token_file(file_path, category)
        results.append(result)
        
        # Restore original strategy
        if token_name in gentle_tokens:
            cleaner.cleaning_strategies[CATEGORIES.get(category, 'gentle')] = original_strategy
        
        if result['status'] == 'cleaned_successfully':
            success_count += 1
        else:
            error_count += 1
    
    # Save enhanced cleaning log
    log_df = pl.DataFrame(results)
    log_path = CLEANED_BASE / f'{category}_enhanced_cleaning_log.json'
    log_df.write_json(log_path)
    
    summary = {
        'category': category,
        'total_files_processed': len(parquet_files),
        'successfully_cleaned': success_count,
        'errors': error_count,
        'skipped': skipped_count,
        'success_rate': (success_count / (len(parquet_files) - skipped_count)) * 100 if (len(parquet_files) - skipped_count) > 0 else 0,
        'cleaning_log_path': str(log_path),
        'investigation_guided': investigation_results is not None
    }
    
    logger.info(f"Enhanced category {category} cleaning complete:")
    logger.info(f"  Cleaned: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Skipped: {skipped_count}")
    
    return summary

def clean_all_categories_with_investigation(investigation_results: Optional[Dict] = None,
                                          limit_per_category: Optional[int] = None) -> Dict:
    """
    Clean all categories with intelligent gap handling based on investigation results
    
    Args:
        investigation_results: Results from comprehensive gap investigation
        limit_per_category: Optional limit on number of files to process per category
        
    Returns:
        Summary statistics of entire enhanced cleaning process
    """
    logger.info("Starting enhanced cleaning of all categories with gap investigation guidance")
    
    if investigation_results:
        logger.info("Gap investigation results will guide cleaning decisions")
        logger.info(f"Tokens to skip: {len(investigation_results.get('recommendations', {}).get('remove_completely', []))}")
        logger.info(f"Tokens for manual review: {len(investigation_results.get('recommendations', {}).get('needs_manual_review', []))}")
        logger.info(f"Tokens to clean: {len(investigation_results.get('recommendations', {}).get('keep_and_clean', []))}")
    else:
        logger.info("No gap investigation results provided - using standard category cleaning")
    
    results = {}
    total_files = 0
    total_success = 0
    total_errors = 0
    total_skipped = 0
    
    for category in CATEGORIES.keys():
        logger.info(f"\n--- Enhanced cleaning category: {category} ---")
        result = clean_category_with_gap_investigation(
            category, 
            investigation_results, 
            limit_per_category
        )
        results[category] = result
        
        if 'total_files_processed' in result:
            total_files += result['total_files_processed']
            total_success += result['successfully_cleaned']
            total_errors += result['errors']
            total_skipped += result.get('skipped', 0)
    
    # Overall enhanced summary
    overall_summary = {
        'total_files_processed': total_files,
        'total_successfully_cleaned': total_success,
        'total_errors': total_errors,
        'total_skipped': total_skipped,
        'overall_success_rate': (total_success / (total_files - total_skipped)) * 100 if (total_files - total_skipped) > 0 else 0,
        'category_results': results,
        'investigation_guided': investigation_results is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save enhanced summary
    summary_path = CLEANED_BASE / 'enhanced_cleaning_summary.json'
    pl.DataFrame([overall_summary]).write_json(summary_path)
    
    logger.info(f"\n=== ENHANCED CLEANING COMPLETE ===")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Successfully cleaned: {total_success}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Skipped (per investigation): {total_skipped}")
    logger.info(f"Success rate: {overall_summary['overall_success_rate']:.1f}%")
    logger.info(f"Enhanced summary saved to: {summary_path}")
    
    return overall_summary

if __name__ == "__main__":
    # Example usage
    summary = clean_all_categories(limit_per_category=None)
    print("Cleaning complete!")
    print(f"Overall success rate: {summary['overall_success_rate']:.1f}%") 