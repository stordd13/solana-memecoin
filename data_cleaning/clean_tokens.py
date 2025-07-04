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
    Advanced token cleaner that applies universal data quality checks to all tokens,
    then category-specific refinements. This ensures consistent data quality across
    all categories while preserving legitimate market movements.
    """
    
    def __init__(self):
        self.cleaning_strategies = {
            'gentle': self._gentle_cleaning,
            'minimal': self._minimal_cleaning, 
            'preserve': self._preserve_extremes_cleaning,
            'aggressive': self._aggressive_cleaning
        }
        
        # REFACTORED: More lenient thresholds for memecoin data
        # Memecoins can have extreme but legitimate movements
        self.artifact_thresholds = {
            'listing_spike_multiplier': 100,    # 100x median for listing artifacts (was 20x)
            'listing_drop_threshold': 0.99,     # 99% drop after spike (unchanged)
            'data_error_threshold': 10000,      # 1,000,000% - only truly impossible moves (was 1000)
            'flash_crash_recovery': 0.95,       # 95% recovery within 5 minutes (unchanged)
            'zero_volume_threshold': 0.001      # Minimum volume for valid trades (unchanged)
        }
        
        # Market behavior thresholds - even more permissive for memecoins
        self.market_thresholds = {
            'max_realistic_pump': 10000,        # 1,000,000% pumps can be real (was 50)
            'max_realistic_dump': 0.99,         # 99% dumps can be real (was 0.95)
            'sustained_movement_minutes': 3,    # Real movements last >3 minutes (unchanged)
            'volume_confirmation_ratio': 0.1    # Volume should support price moves (unchanged)
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
            
            # Initialize modifications list
            modifications = []
            
            # Calculate returns if not present
            if 'returns' not in df.columns:
                df = self._calculate_returns(df)
                modifications.append({
                    'type': 'returns_calculated',
                    'description': 'Calculated returns from price data'
                })
            
            # UNIVERSAL DATA QUALITY CHECKS (apply to ALL tokens regardless of category)
            df, universal_mods = self._universal_data_quality_checks(df, token_name)
            if universal_mods:
                modifications.extend(universal_mods)
            
            # If token was deemed completely invalid by universal checks, exclude it
            if df is None:
                log['status'] = 'excluded_by_universal_quality_checks'
                log['modifications'] = modifications
                return log
            
            # Apply category-specific cleaning strategy
            strategy = CATEGORIES.get(category, 'gentle')
            df_cleaned, category_modifications = self.cleaning_strategies[strategy](df, token_name)
            if category_modifications:
                modifications.extend(category_modifications)
            
            if df_cleaned is None:
                log['status'] = 'excluded_due_to_severe_issues'
                log['modifications'] = modifications
                return log
            
            # Save cleaned file
            output_dir = CLEANED_BASE / category
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{token_name}.parquet"
            
            df_cleaned.write_parquet(output_path, 
                                     compression="zstd", 
                                     compression_level=3)
            
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

    def _calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate returns from price data.
        Returns = (price_t - price_t-1) / price_t-1
        """
        # Sort by datetime to ensure proper order
        df = df.sort('datetime')
        
        # Calculate returns using polars expression
        df = df.with_columns([
            (pl.col('price').pct_change()).alias('returns')
        ])
        
        # Replace the first NaN with 0 (no return for first observation)
        df = df.with_columns([
            pl.col('returns').fill_null(0.0)
        ])
        
        return df

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
        - Remove constant price periods at the end (CRITICAL: prevents data leakage!)
        - Don't bother with gaps or minor issues since token is inactive
        """
        modifications = []
        df = df.clone()
        
        # CRITICAL FIX: Remove constant price periods at the end to prevent data leakage
        df, death_mods = self._remove_death_period(df, token_name)
        if death_mods:
            modifications.extend(death_mods)
        
        # Only fix severe data errors and invalid prices
        df, error_mods = self._fix_data_errors(df)
        if error_mods:
            modifications.extend(error_mods)
            
        df, price_mods = self._handle_invalid_prices_conservative(df)
        if price_mods:
            modifications.extend(price_mods)
        
        return df, modifications

    def _remove_death_period(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Remove constant price periods at the end of dead tokens to prevent data leakage.
        
        This is CRITICAL for ML model integrity - without this, models will learn to predict
        constant prices when they see a pattern of constant prices, leading to artificially
        high accuracy metrics.
        
        Args:
            df: Token dataframe sorted by datetime
            token_name: Name of token for logging
            
        Returns:
            Tuple of (cleaned_df, modifications_list)
        """
        modifications = []
        
        if df.height < 60:  # Need at least 1 hour of data
            return df, modifications
        
        # Sort by datetime to ensure proper order
        df = df.sort('datetime')
        
        # Find the longest constant price period at the end
        prices = df['price'].to_list()
        
        if not prices:
            return df, modifications
        
        # Work backwards from the end to find where constant period starts
        last_price = prices[-1]
        constant_count = 0
        
        for i in range(len(prices) - 1, -1, -1):
            if abs(prices[i] - last_price) < (last_price * 0.0001):  # Allow for tiny rounding differences
                constant_count += 1
            else:
                break
        
        # Only remove if constant period is >= 60 minutes (1 hour) to prevent data leakage
        min_constant_minutes = 60
        
        if constant_count >= min_constant_minutes:
            # Keep minimal constant data for context, remove the bulk to prevent leakage
            # Keep only 2 minutes of constant price - just enough to show the death transition
            keep_constant_minutes = 2
            remove_count = constant_count - keep_constant_minutes
            
            if remove_count > 0:
                # Keep everything except the last 'remove_count' rows
                df_cleaned = df.head(df.height - remove_count)
                
                modifications.append({
                    'type': 'death_period_removed',
                    'constant_minutes_total': constant_count,
                    'constant_minutes_removed': remove_count,
                    'constant_minutes_kept': keep_constant_minutes,
                    'constant_price': last_price,
                    'rows_before': df.height,
                    'rows_after': df_cleaned.height,
                    'reason': 'prevent_data_leakage_in_forecasting_models'
                })
                
                print(f"🛡️  ANTI-LEAKAGE: {token_name} - Removed {remove_count} minutes of constant price (kept {keep_constant_minutes} for context)")
                
                return df_cleaned, modifications
        
        return df, modifications

    def _universal_data_quality_checks(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Universal data quality checks that apply to ALL tokens regardless of category.
        These catch fundamental data corruption issues that affect model training.
        
        Checks:
        1. Remove constant death periods (ALL categories need this)
        2. Detect and remove staircase artifacts (vertical + horizontal patterns)
        3. Remove tokens with extreme price ratios (>50,000x likely data errors)
        4. Handle NaN values and infinite prices
        5. Remove tokens that are too short after cleaning
        """
        modifications = []
        original_length = df.height
        
        if original_length < 60:  # Less than 1 hour of data
            return None, [{'type': 'excluded_too_short', 'original_length': original_length}]
        
        # 1. Remove NaN and infinite values FIRST
        df, nan_mods = self._handle_nan_and_infinite_values(df)
        if nan_mods:
            modifications.extend(nan_mods)
        
        if df is None or df.height < 30:
            return None, modifications + [{'type': 'excluded_too_many_invalid_values'}]
        
        # 2. UNIVERSAL: Remove death periods (applies to ALL categories!)
        df, death_mods = self._remove_death_period(df, token_name)
        if death_mods:
            modifications.extend(death_mods)
        
        if df.height < 30:
            return None, modifications + [{'type': 'excluded_mostly_death_period'}]
        
        # 3. Check for extreme price ratios (likely data corruption)
        prices = df['price'].to_numpy()
        if len(prices) > 0:
            finite_prices = prices[np.isfinite(prices)]
            if len(finite_prices) > 0:
                min_price = np.min(finite_prices)
                max_price = np.max(finite_prices)
                if min_price > 0:
                    price_ratio = max_price / min_price
                    # REFACTORED: Allow extreme ratios for memecoins (up to 10,000,000x)
                    # Only flag if it's clearly impossible (>10,000,000x is suspicious)
                    if price_ratio > 10000000:  # 10,000,000x price ratio threshold
                        return None, modifications + [{
                            'type': 'excluded_extreme_price_ratio',
                            'price_ratio': price_ratio,
                            'min_price': min_price,
                            'max_price': max_price,
                            'reason': 'Price ratio >10,000,000x suggests data corruption'
                        }]
                    elif price_ratio > 1000000:
                        # Log but don't exclude - these can be legitimate
                        print(f"🚀 EXTREME RATIO: {token_name} - {price_ratio:.0f}x (min: {min_price:.10f}, max: {max_price:.2f})")
                        modifications.append({
                            'type': 'extreme_ratio_noted',
                            'price_ratio': price_ratio,
                            'action': 'preserved'
                        })
        
        # 4. Detect and remove staircase artifacts
        df, staircase_mods = self._detect_and_remove_staircase_artifacts(df, token_name)
        if staircase_mods:
            modifications.extend(staircase_mods)
        
        if df is None or df.height < 30:
            return None, modifications + [{'type': 'excluded_staircase_artifact'}]
        
        # 5. Additional check for extreme returns in remaining data
        if df.height > 0:
            returns = df['returns'].to_numpy()
            finite_returns = returns[np.isfinite(returns)]
            if len(finite_returns) > 0:
                max_abs_return = np.max(np.abs(finite_returns))
                if max_abs_return > 1000:  # >100,000% return is unrealistic
                    return None, modifications + [{
                        'type': 'excluded_extreme_returns',
                        'max_return': max_abs_return
                    }]

        # 6. Check price variability to filter "straight line" tokens
        df, variability_mods = self._check_price_variability(df, token_name)
        if variability_mods:
            modifications.extend(variability_mods)
        
        if df is None:
            return None, modifications + [{'type': 'excluded_low_variability'}]

        # 7. Final length check
        if df.height < 60:  # After all cleaning, ensure minimum viable length
            return None, modifications + [{'type': 'excluded_too_short_after_cleaning', 'final_length': df.height}]
        
        if len(modifications) > 0:
            print(f"🧹 UNIVERSAL: {token_name} - Applied {len(modifications)} universal quality fixes")
        
        return df, modifications

    def _handle_nan_and_infinite_values(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """Handle NaN and infinite values in price data"""
        modifications = []
        original_length = df.height
        
        # Remove rows with NaN or infinite prices
        df_clean = df.filter(pl.col('price').is_finite())
        
        removed_count = original_length - df_clean.height
        if removed_count > 0:
            modifications.append({
                'type': 'nan_infinite_values_removed',
                'count': removed_count,
                'percentage': (removed_count / original_length) * 100
            })
        
        # If more than 50% of data was NaN/infinite, reject the token
        if removed_count > (original_length * 0.5):
            return None, modifications
        
        return df_clean, modifications

    def _check_price_variability(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        REFACTORED: Focus on detecting truly flat "straight line" tokens.
        Real tokens have natural price variations - preserve those!
        """
        modifications = []
        prices = df['price'].to_numpy()
        
        if len(prices) < 30:
            return df, modifications
        
        # Key insight: Real tokens have price ticks/changes, dead lines don't
        
        # 1. Count actual price changes (not percentage-based)
        price_changes = np.diff(prices)
        non_zero_changes = np.sum(price_changes != 0)
        change_ratio = non_zero_changes / len(price_changes) if len(price_changes) > 0 else 0
        
        # 2. Count unique prices (straight lines have very few)
        unique_prices = len(np.unique(prices))
        unique_ratio = unique_prices / len(prices)
        
        # 3. Look for long periods of identical prices
        max_consecutive_same = 0
        current_consecutive = 1
        for i in range(1, len(prices)):
            if prices[i] == prices[i-1]:
                current_consecutive += 1
                max_consecutive_same = max(max_consecutive_same, current_consecutive)
            else:
                current_consecutive = 1
        
        # 4. Calculate tick density in different time windows
        # Real tokens show activity across different timeframes
        window_sizes = [10, 30, 60] if len(prices) >= 60 else [10, 30] if len(prices) >= 30 else [10]
        active_windows = 0
        total_windows = 0
        
        for window_size in window_sizes:
            if window_size <= len(prices):
                for i in range(0, len(prices) - window_size + 1, window_size // 2):
                    window_prices = prices[i:i + window_size]
                    window_unique = len(np.unique(window_prices))
                    total_windows += 1
                    if window_unique > 2:  # More than 2 unique prices in window
                        active_windows += 1
        
        activity_ratio = active_windows / total_windows if total_windows > 0 else 0
        
        # 5. Simple entropy check - how "random" are the price movements
        if len(price_changes) > 0:
            # Normalize price changes to detect patterns
            normalized_changes = np.sign(price_changes)  # -1, 0, 1
            unique_patterns = len(np.unique(normalized_changes))
        else:
            unique_patterns = 0
        
        # REFACTORED CRITERIA: Only filter truly dead/straight line tokens
        is_straight_line = (
            unique_ratio < 0.01 and                    # <1% unique prices (very strict)
            change_ratio < 0.05 and                    # <5% of time prices change
            max_consecutive_same > len(prices) * 0.8   # >80% consecutive same prices
        ) or (
            unique_prices <= 3 and                     # 3 or fewer unique prices total
            activity_ratio < 0.1                       # <10% of windows show activity
        )
        
        # Calculate additional metrics expected by the app
        # Price coefficient of variation (CV)
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        price_cv = price_std / price_mean if price_mean > 0 else 0
        
        # Log price coefficient of variation
        log_prices = np.log(prices + 1e-10)  # Add small value to avoid log(0)
        log_price_cv = np.std(log_prices) / np.mean(log_prices) if np.mean(log_prices) != 0 else 0
        
        # Flat periods fraction (fraction of time with no price changes)
        flat_periods_fraction = 1 - change_ratio
        
        # Range efficiency (actual range vs theoretical max range)
        price_range = np.max(prices) - np.min(prices)
        theoretical_max_range = np.max(prices) + np.std(prices)
        range_efficiency = price_range / theoretical_max_range if theoretical_max_range > 0 else 0
        
        # Normalized entropy (measure of price distribution randomness)
        # Simple entropy calculation based on price bins
        if len(prices) > 1:
            hist, _ = np.histogram(prices, bins=min(10, unique_prices))
            hist = hist[hist > 0]  # Remove zero bins
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 1 else 0
            normalized_entropy = entropy / np.log2(len(probs)) if len(probs) > 1 else 0
        else:
            normalized_entropy = 0
        
        # Log the analysis
        variability_metrics = {
            'unique_prices': unique_prices,
            'unique_ratio': unique_ratio,
            'change_ratio': change_ratio,
            'max_consecutive_same': max_consecutive_same,
            'activity_ratio': activity_ratio,
            'unique_patterns': unique_patterns,
            'is_straight_line': is_straight_line,
            # Additional metrics for app compatibility
            'price_cv': price_cv,
            'log_price_cv': abs(log_price_cv),  # Use absolute value to avoid negative CV
            'flat_periods_fraction': flat_periods_fraction,
            'range_efficiency': range_efficiency,
            'normalized_entropy': normalized_entropy
        }
        
        if is_straight_line:
            print(f"📏 STRAIGHT LINE: {token_name} - {unique_prices} unique prices, "
                  f"{change_ratio:.1%} changes, max consecutive: {max_consecutive_same}")
            return None, modifications + [{
                'type': 'excluded_straight_line',
                'metrics': variability_metrics,
                'reason': 'Token shows no meaningful price variation (straight line)'
            }]
        else:
            # Token has acceptable variability
            if unique_ratio < 0.1:  # Low but not too low
                print(f"⚠️  LOW VAR (kept): {token_name} - {unique_prices} unique prices, "
                      f"but {change_ratio:.1%} changes")
            
            modifications.append({
                'type': 'variability_check_passed',
                'metrics': variability_metrics
            })
        
        return df, modifications

    def _detect_and_remove_staircase_artifacts(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        REFACTORED: Detect TRUE staircase artifacts only.
        Uses the same enhanced logic as _fix_extreme_data_corruption for consistency.
        Now checks BOTH pre-jump and post-jump patterns.
        """
        modifications = []
        prices = df['price'].to_numpy()
        
        if len(prices) < 60:  # Need 60 for 30 before + 30 after analysis
            return df, modifications
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Find jump points (>100% in one minute)
        jump_indices = np.where(returns > 1.0)[0]  # 100% threshold
        
        artifacts_found = 0
        earliest_staircase = None
        staircase_details = []
        
        for jump_idx in jump_indices:
            # Need 30 minutes before AND after for enhanced analysis
            if jump_idx >= 30 and jump_idx < len(prices) - 30:
                
                # Analyze both pre-jump and post-jump behavior
                pre_jump_prices = prices[jump_idx-30:jump_idx]
                post_jump_prices = prices[jump_idx+1:jump_idx+31]
                
                # Use the same analysis functions as the main cleaning method
                pre_metrics = self._analyze_staircase_pattern(pre_jump_prices, "PRE")
                post_metrics = self._analyze_staircase_pattern(post_jump_prices, "POST")
                
                # Check if either side shows staircase pattern
                is_pre_staircase = self._is_staircase_pattern(pre_metrics)
                is_post_staircase = self._is_staircase_pattern(post_metrics)
                
                is_definitive_staircase = is_pre_staircase or is_post_staircase
                
                if is_definitive_staircase:
                    artifacts_found += 1
                    
                    # Determine staircase type
                    staircase_type = []
                    if is_pre_staircase:
                        staircase_type.append("flat→jump")
                    if is_post_staircase:
                        staircase_type.append("jump→flat")
                    
                    staircase_details.append({
                        'jump_idx': jump_idx,
                        'staircase_type': " + ".join(staircase_type),
                        'pre_metrics': pre_metrics,
                        'post_metrics': post_metrics,
                        'is_high_confidence': (
                            (pre_metrics['unique_prices'] <= 3 and pre_metrics['tick_movements'] <= 1) or
                            (post_metrics['unique_prices'] <= 3 and post_metrics['tick_movements'] <= 1)
                        )
                    })
                    
                    # Update earliest staircase with smart cut point selection
                    if earliest_staircase is None or jump_idx < earliest_staircase:
                        earliest_staircase = jump_idx
                    
                    print(f"🪜 ENHANCED STAIRCASE: {' + '.join(staircase_type)} at minute {jump_idx} "
                          f"(Pre: {pre_metrics['unique_prices']}u/{pre_metrics['tick_movements']}t, "
                          f"Post: {post_metrics['unique_prices']}u/{post_metrics['tick_movements']}t)")
        
        if earliest_staircase is not None:
            # Find the details of the earliest staircase for smart cutting
            earliest_details = next(d for d in staircase_details if d['jump_idx'] == earliest_staircase)
            
            # Smart cut point selection based on staircase type
            is_pre_type = "flat→jump" in earliest_details['staircase_type']
            
            if is_pre_type:
                # For flat→jump pattern, cut at start of flat period
                cut_point = max(0, earliest_staircase - 30)
                cut_type = "pre_jump_flat_removal"
            else:
                # For jump→flat pattern, cut at jump point  
                cut_point = earliest_staircase
                cut_type = "post_jump_flat_removal"
            
            # Remove from the calculated cut point onwards
            df_clean = df.head(cut_point + 1)
            
            # Count high confidence staircases
            high_confidence_count = sum(1 for d in staircase_details if d['is_high_confidence'])
            
            modifications.append({
                'type': 'enhanced_staircase_artifacts_removed_universal',
                'total_artifacts': artifacts_found,
                'high_confidence_artifacts': high_confidence_count,
                'cut_at_minute': cut_point,
                'cut_type': cut_type,
                'earliest_staircase_type': earliest_details['staircase_type'],
                'original_length': len(prices),
                'final_length': df_clean.height,
                'reason': 'Enhanced pre+post jump staircase pattern detected'
            })
            
            print(f"✂️ ENHANCED UNIVERSAL CUT: {token_name} - Removed from minute {cut_point} "
                  f"({earliest_details['staircase_type']} pattern)")
            
            return df_clean, modifications
        
        return df, modifications

    def _preserve_extremes_cleaning(self, df: pl.DataFrame, token_name: str) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Preserve extremes cleaning for tokens with extreme movements
        - Keep ALL legitimate extreme movements (this is their defining characteristic!)
        - Only remove obvious data corruption
        - Be very conservative about what constitutes an "error"
        - BUT still remove constant death periods to prevent data leakage
        """
        modifications = []
        df = df.clone()
        
        # CRITICAL: Remove constant price periods at the end (even for extreme tokens!)
        # This prevents data leakage in ML models
        df, death_mods = self._remove_death_period(df, token_name)
        if death_mods:
            modifications.extend(death_mods)
        
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
        REFACTORED: Only fix truly impossible data errors.
        Memecoins can have 10,000%+ moves in one minute legitimately!
        """
        modifications = []
        
        if df.height < 2:
            return df, modifications
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df = df.with_columns([
                pl.col('price').pct_change().alias('returns')
            ])
        
        # REFACTORED: Much higher threshold for "impossible" moves
        # Only flag if it's clearly a data error (>1,000,000% in one minute)
        error_threshold = 10000  # 1,000,000% = 10,000x in one minute
        extreme_returns = df.filter(pl.col('returns').abs() > error_threshold)
        
        if extreme_returns.height > 0:
            # These are likely data errors - but let's double-check
            error_indices = extreme_returns.with_row_count()['row_nr'].to_list()
            
            # Log the extreme values for monitoring
            for idx in error_indices:
                return_val = df.row(idx, named=True)['returns']
                price_before = df.row(max(0, idx-1), named=True)['price'] if idx > 0 else None
                price_after = df.row(idx, named=True)['price']
                print(f"🚨 EXTREME: {return_val*100:.1f}% move at minute {idx} "
                      f"(${price_before:.10f} → ${price_after:.10f})")
            
            # Only interpolate if it's truly impossible
            # For now, accept even these extreme moves - memecoins are wild!
            modifications.append({
                'type': 'extreme_returns_detected_but_kept',
                'count': len(error_indices),
                'threshold_used': error_threshold,
                'action': 'preserved',
                'reason': 'Memecoins can have extreme legitimate moves'
            })
            
            # Don't remove anything - just flag it
            print(f"⚠️  EXTREME MOVES: Found {len(error_indices)} moves >{error_threshold*100}% but keeping them")
        
        return df, modifications

    def _fix_extreme_data_corruption(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        REFACTORED: Smart detection that distinguishes between:
        1. Staircase artifacts: 
           - Type A: vertical jump + ZERO variation horizontal line (REMOVE)
           - Type B: horizontal flat line + vertical jump (REMOVE) ← NEW!
        2. Legitimate extreme pumps: can be 1,000,000x+ but show continued trading (KEEP)
        
        Key insight: Real tokens have price variations before AND after pumps, 
        staircases are perfectly flat on one or both sides.
        """
        modifications = []
        
        if df.height < 60:  # Need more data for robust pattern analysis (30 before + 30 after)
            return df, modifications
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df = df.with_columns([
                pl.col('price').pct_change().alias('returns')
            ])
        
        # Convert to numpy for analysis
        prices = df['price'].to_numpy()
        returns = df['returns'].to_numpy()
        
        # Find potential jump points (>100% in one minute)
        jump_threshold = 1.0  # 100% - lower threshold to catch more patterns
        jump_indices = np.where(returns > jump_threshold)[0]
        
        staircase_artifacts = []
        legitimate_pumps = []
        
        for idx in jump_indices:
            # Need 30 minutes before AND after for robust analysis
            if idx >= 30 and idx < len(returns) - 30:
                
                # Analyze the 30 minutes BEFORE the jump (NEW!)
                pre_jump_prices = prices[idx-30:idx]
                pre_jump_returns = returns[idx-30:idx]
                
                # Analyze the 30 minutes AFTER the jump (existing)
                post_jump_prices = prices[idx+1:idx+31]
                post_jump_returns = returns[idx+1:idx+31]
                
                # Check BOTH pre-jump and post-jump for staircase patterns
                pre_staircase_metrics = self._analyze_staircase_pattern(pre_jump_prices, "PRE")
                post_staircase_metrics = self._analyze_staircase_pattern(post_jump_prices, "POST")
                
                # Determine if this is a staircase artifact
                is_pre_staircase = self._is_staircase_pattern(pre_staircase_metrics)
                is_post_staircase = self._is_staircase_pattern(post_staircase_metrics)
                
                # If EITHER side shows staircase pattern → artifact
                is_staircase_artifact = is_pre_staircase or is_post_staircase
                
                if is_staircase_artifact:
                    staircase_type = []
                    if is_pre_staircase:
                        staircase_type.append("flat→jump")
                    if is_post_staircase:
                        staircase_type.append("jump→flat")
                    
                    staircase_artifacts.append({
                        'idx': idx,
                        'jump_size': returns[idx],
                        'staircase_type': " + ".join(staircase_type),
                        'pre_metrics': pre_staircase_metrics,
                        'post_metrics': post_staircase_metrics
                    })
                    
                    print(f"🪜 STAIRCASE: {' + '.join(staircase_type)} at minute {idx} "
                          f"({returns[idx]*100:.1f}%) - "
                          f"Pre: {pre_staircase_metrics['unique_prices']}u/{pre_staircase_metrics['tick_movements']}t, "
                          f"Post: {post_staircase_metrics['unique_prices']}u/{post_staircase_metrics['tick_movements']}t")
                else:
                    # This is a legitimate pump - has variation on both sides
                    post_volatility = np.std(post_jump_returns[np.isfinite(post_jump_returns)])
                    pre_volatility = np.std(pre_jump_returns[np.isfinite(pre_jump_returns)])
                    
                    legitimate_pumps.append({
                        'idx': idx,
                        'jump_size': returns[idx],
                        'pre_volatility': pre_volatility,
                        'post_volatility': post_volatility,
                        'pre_unique': pre_staircase_metrics['unique_prices'],
                        'post_unique': post_staircase_metrics['unique_prices']
                    })
                    
                    if returns[idx] > 10:  # Log only massive pumps
                        print(f"🚀 REAL PUMP: Jump at minute {idx} ({returns[idx]*100:.1f}%) - "
                              f"Pre: {pre_staircase_metrics['unique_prices']}u/{pre_staircase_metrics['tick_movements']}t, "
                              f"Post: {post_staircase_metrics['unique_prices']}u/{post_staircase_metrics['tick_movements']}t")
        
        # Remove staircase artifacts but be conservative
        if staircase_artifacts:
            # Only remove if we're confident it's a staircase
            # Find the earliest definitive staircase
            earliest_staircase = None
            staircase_confidence = {}
            
            for artifact in staircase_artifacts:
                # Check confidence level - both pre and post metrics
                pre_confident = (artifact['pre_metrics']['unique_prices'] <= 3 and 
                               artifact['pre_metrics']['tick_movements'] <= 1)
                post_confident = (artifact['post_metrics']['unique_prices'] <= 3 and 
                                artifact['post_metrics']['tick_movements'] <= 1)
                
                # High confidence if either side is definitely flat
                is_high_confidence = pre_confident or post_confident
                staircase_confidence[artifact['idx']] = is_high_confidence
                
                if is_high_confidence:
                    if earliest_staircase is None or artifact['idx'] < earliest_staircase:
                        earliest_staircase = artifact['idx']
            
            if earliest_staircase is not None:
                # For pre-jump staircases, we might want to cut earlier
                # Find if the earliest staircase is a pre-jump type
                earliest_artifact = next(a for a in staircase_artifacts if a['idx'] == earliest_staircase)
                is_pre_type = "flat→jump" in earliest_artifact['staircase_type']
                
                if is_pre_type:
                    # For flat→jump, cut at the start of the flat period (30 minutes before jump)
                    cut_point = max(0, earliest_staircase - 30)
                else:
                    # For jump→flat, cut at the jump point
                    cut_point = earliest_staircase
                
                df_clean = df.head(cut_point + 1)
                
                high_confidence_count = sum(1 for conf in staircase_confidence.values() if conf)
                
                modifications.append({
                    'type': 'enhanced_staircase_artifacts_removed',
                    'total_staircases': len(staircase_artifacts),
                    'high_confidence_staircases': high_confidence_count,
                    'legitimate_pumps_preserved': len(legitimate_pumps),
                    'cut_at_minute': cut_point,
                    'cut_type': 'pre_jump_flat' if is_pre_type else 'post_jump_flat',
                    'original_length': len(prices),
                    'final_length': df_clean.height,
                    'earliest_staircase_type': earliest_artifact['staircase_type']
                })
                
                print(f"✂️ ENHANCED CUT: Removed from minute {cut_point} "
                      f"({earliest_artifact['staircase_type']} staircase)")
                
                return df_clean, modifications
        
        # No staircases found - preserve all data
        if legitimate_pumps:
            max_pump = max(legitimate_pumps, key=lambda x: x['jump_size'])
            modifications.append({
                'type': 'all_pumps_preserved_enhanced',
                'pump_count': len(legitimate_pumps),
                'max_pump_size': max_pump['jump_size'],
                'max_pump_minute': max_pump['idx'],
                'analysis_type': 'pre_and_post_jump_checked'
            })
            print(f"✅ ENHANCED PRESERVED: All {len(legitimate_pumps)} pumps passed pre+post analysis "
                  f"(max: {max_pump['jump_size']*100:.1f}%)")
        
        return df, modifications

    def _analyze_staircase_pattern(self, prices: np.ndarray, period_type: str) -> Dict:
        """
        Analyze a price sequence for staircase pattern characteristics
        
        Args:
            prices: Array of prices to analyze
            period_type: "PRE" or "POST" for logging
            
        Returns:
            Dictionary with staircase pattern metrics
        """
        if len(prices) == 0:
            return {
                'unique_prices': 0,
                'unique_price_ratio': 0,
                'variation_ratio': 0,
                'relative_std': 0,
                'max_deviation': 0,
                'tick_movements': 0,
                'consecutive_same_max': 0
            }
        
        # 1. Count unique prices
        unique_prices = len(np.unique(prices))
        unique_price_ratio = unique_prices / len(prices)
        
        # 2. Calculate price variations
        price_diffs = np.diff(prices)
        non_zero_diffs = np.sum(np.abs(price_diffs) > 0)
        variation_ratio = non_zero_diffs / len(price_diffs) if len(price_diffs) > 0 else 0
        
        # 3. Standard deviation relative to mean
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        relative_std = price_std / price_mean if price_mean > 0 else 0
        
        # 4. Max deviation from mean
        max_deviation = np.max(np.abs(prices - price_mean)) / price_mean if price_mean > 0 else 0
        
        # 5. Count of "tick" movements (meaningful price changes)
        tick_threshold = price_mean * 0.0001 if price_mean > 0 else 0  # 0.01% threshold
        tick_movements = np.sum(np.abs(price_diffs) > tick_threshold)
        
        # 6. Longest consecutive sequence of identical prices
        consecutive_same_max = 0
        current_consecutive = 1
        for i in range(1, len(prices)):
            if prices[i] == prices[i-1]:
                current_consecutive += 1
                consecutive_same_max = max(consecutive_same_max, current_consecutive)
            else:
                current_consecutive = 1
        
        return {
            'unique_prices': unique_prices,
            'unique_price_ratio': unique_price_ratio,
            'variation_ratio': variation_ratio,
            'relative_std': relative_std,
            'max_deviation': max_deviation,
            'tick_movements': tick_movements,
            'consecutive_same_max': consecutive_same_max,
            'period_type': period_type,
            'length': len(prices)
        }

    def _is_staircase_pattern(self, metrics: Dict) -> bool:
        """
        Determine if metrics indicate a staircase pattern
        
        Uses the same strict criteria as before but now applied to both
        pre-jump and post-jump periods
        """
        return (
            metrics['unique_price_ratio'] < 0.1 and          # <10% unique prices
            metrics['variation_ratio'] < 0.1 and             # <10% intervals have changes
            metrics['relative_std'] < 0.001 and              # <0.1% standard deviation  
            metrics['max_deviation'] < 0.01 and              # <1% max deviation
            metrics['tick_movements'] < 3                     # <3 meaningful changes in 30 min
        ) or (
            metrics['consecutive_same_max'] >= 20             # 20+ consecutive identical prices
        )

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