#!/usr/bin/env python3
"""
Short-Term Feature Engineering for 15m-60m Predictions

This module creates specialized features for short-term memecoin prediction:
- Micro-pattern detection (order flow, bid-ask dynamics)
- High-frequency momentum indicators
- Volatility clustering patterns
- Market microstructure features
- Noise vs signal separation

USAGE:
    python feature_engineering/short_term_features.py --input_dir data/cleaned_tokens_short_term --output_dir data/features_short_term
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ShortTermFeatureEngineer:
    """
    Specialized feature engineering for short-term memecoin predictions
    
    Focuses on micro-patterns that are meaningful for 15m-60m predictions:
    - Order flow patterns
    - Momentum persistence
    - Volatility clustering
    - Market microstructure signals
    """
    
    def __init__(self):
        self.feature_windows = {
            'micro': [3, 5, 10],      # 3-10 minute micro patterns
            'short': [15, 30],        # 15-30 minute short patterns
            'context': [60, 120]      # 60-120 minute context
        }
    
    def create_short_term_features(self, df: pl.DataFrame, token_name: str = "token") -> pl.DataFrame:
        """
        Create comprehensive short-term features
        
        Args:
            df: DataFrame with 'datetime' and 'price' columns (short-term cleaned)
            token_name: Token identifier for logging
            
        Returns:
            DataFrame with short-term features added
        """
        if len(df) < 120:  # Need at least 2 hours for context
            print(f"⚠️ {token_name}: Insufficient data for short-term features ({len(df)} rows)")
            return pl.DataFrame()
        
        # Ensure proper sorting
        df = df.sort('datetime')
        
        try:
            # Start with basic price features
            df_features = df.with_columns([
                (pl.col('price').log() - pl.col('price').shift(1).log()).alias('log_returns'),
                pl.col('price').pct_change().alias('returns')
            ]).drop_nulls()
            
            # 1. Micro-momentum features (3-10 minute patterns)
            df_features = self._add_micro_momentum_features(df_features)
            
            # 2. High-frequency volatility clustering
            df_features = self._add_volatility_clustering_features(df_features)
            
            # 3. Order flow approximation features
            df_features = self._add_order_flow_features(df_features)
            
            # 4. Market microstructure patterns
            df_features = self._add_microstructure_features(df_features)
            
            # 5. Noise vs signal features
            df_features = self._add_noise_signal_features(df_features)
            
            # 6. Short-term momentum persistence
            df_features = self._add_momentum_persistence_features(df_features)
            
            # 7. Tick-by-tick patterns
            df_features = self._add_tick_patterns(df_features)
            
            # Drop nulls and ensure finite values
            df_features = df_features.drop_nulls()
            
            # Ensure all numeric columns are finite
            numeric_cols = [c for c in df_features.columns if c not in ['datetime']]
            for col in numeric_cols:
                df_features = df_features.with_columns([
                    pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(0).alias(col)
                ])
            
            print(f"✅ {token_name}: Created {len(df_features.columns)-2} short-term features")
            return df_features
            
        except Exception as e:
            print(f"❌ {token_name}: Error creating short-term features: {e}")
            return pl.DataFrame()
    
    def _add_micro_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add micro-momentum features for very short-term patterns"""
        
        # 3-minute momentum
        df = df.with_columns([
            # Price momentum over 3 minutes
            ((pl.col('price') / pl.col('price').shift(3) - 1).fill_null(0)).alias('momentum_3m'),
            # Returns momentum (acceleration)
            (pl.col('returns').rolling_mean(3).fill_null(0)).alias('returns_momentum_3m'),
            # Volatility momentum
            (pl.col('returns').abs().rolling_mean(3).fill_null(0)).alias('volatility_momentum_3m')
        ])
        
        # 5-minute momentum
        df = df.with_columns([
            ((pl.col('price') / pl.col('price').shift(5) - 1).fill_null(0)).alias('momentum_5m'),
            (pl.col('returns').rolling_mean(5).fill_null(0)).alias('returns_momentum_5m'),
            (pl.col('returns').abs().rolling_mean(5).fill_null(0)).alias('volatility_momentum_5m')
        ])
        
        # 10-minute momentum
        df = df.with_columns([
            ((pl.col('price') / pl.col('price').shift(10) - 1).fill_null(0)).alias('momentum_10m'),
            (pl.col('returns').rolling_mean(10).fill_null(0)).alias('returns_momentum_10m'),
            (pl.col('returns').abs().rolling_mean(10).fill_null(0)).alias('volatility_momentum_10m')
        ])
        
        # Momentum acceleration (change in momentum)
        df = df.with_columns([
            (pl.col('momentum_3m') - pl.col('momentum_3m').shift(3)).alias('momentum_accel_3m'),
            (pl.col('momentum_5m') - pl.col('momentum_5m').shift(5)).alias('momentum_accel_5m'),
            (pl.col('momentum_10m') - pl.col('momentum_10m').shift(10)).alias('momentum_accel_10m')
        ])
        
        return df
    
    def _add_volatility_clustering_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility clustering features for short-term risk assessment"""
        
        # Rolling volatility at different windows
        df = df.with_columns([
            pl.col('returns').abs().rolling_mean(5).alias('volatility_5m'),
            pl.col('returns').abs().rolling_mean(10).alias('volatility_10m'),
            pl.col('returns').abs().rolling_mean(15).alias('volatility_15m'),
            pl.col('returns').abs().rolling_mean(30).alias('volatility_30m')
        ])
        
        # Volatility ratios (clustering indicators)
        df = df.with_columns([
            (pl.col('volatility_5m') / pl.col('volatility_15m')).alias('volatility_ratio_5_15'),
            (pl.col('volatility_10m') / pl.col('volatility_30m')).alias('volatility_ratio_10_30'),
            (pl.col('volatility_5m') / pl.col('volatility_10m')).alias('volatility_ratio_5_10')
        ])
        
        # Volatility persistence (autocorrelation approximation)
        df = df.with_columns([
            # High volatility periods
            (pl.col('volatility_5m') > pl.col('volatility_15m') * 1.5).alias('high_vol_period'),
            # Volatility spikes
            (pl.col('volatility_5m') > pl.col('volatility_30m') * 2.0).alias('volatility_spike'),
            # Volatility trend
            (pl.col('volatility_5m') - pl.col('volatility_5m').shift(5)).alias('volatility_trend')
        ])
        
        return df
    
    def _add_order_flow_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add order flow approximation features"""
        
        # Price change patterns (proxy for order flow)
        df = df.with_columns([
            # Up/down tick counts
            (pl.col('returns') > 0).cast(pl.Int32).rolling_sum(5).alias('up_ticks_5m'),
            (pl.col('returns') < 0).cast(pl.Int32).rolling_sum(5).alias('down_ticks_5m'),
            (pl.col('returns') > 0).cast(pl.Int32).rolling_sum(10).alias('up_ticks_10m'),
            (pl.col('returns') < 0).cast(pl.Int32).rolling_sum(10).alias('down_ticks_10m')
        ])
        
        # Order flow imbalance
        df = df.with_columns([
            (pl.col('up_ticks_5m') - pl.col('down_ticks_5m')).alias('order_imbalance_5m'),
            (pl.col('up_ticks_10m') - pl.col('down_ticks_10m')).alias('order_imbalance_10m'),
            # Normalized order flow
            ((pl.col('up_ticks_5m') - pl.col('down_ticks_5m')) / 
             (pl.col('up_ticks_5m') + pl.col('down_ticks_5m') + 1)).alias('order_flow_ratio_5m'),
            ((pl.col('up_ticks_10m') - pl.col('down_ticks_10m')) / 
             (pl.col('up_ticks_10m') + pl.col('down_ticks_10m') + 1)).alias('order_flow_ratio_10m')
        ])
        
        # Large move detection (proxy for large orders)
        df = df.with_columns([
            # Large positive moves
            (pl.col('returns') > pl.col('returns').rolling_std(30) * 2).cast(pl.Int32).alias('large_buy_signal'),
            # Large negative moves
            (pl.col('returns') < -pl.col('returns').rolling_std(30) * 2).cast(pl.Int32).alias('large_sell_signal'),
            # Recent large moves
            (pl.col('returns') > pl.col('returns').rolling_std(30) * 2).cast(pl.Int32).rolling_sum(10).alias('recent_large_buys'),
            (pl.col('returns') < -pl.col('returns').rolling_std(30) * 2).cast(pl.Int32).rolling_sum(10).alias('recent_large_sells')
        ])
        
        return df
    
    def _add_microstructure_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add market microstructure features"""
        
        # Price level clustering (support/resistance approximation)
        df = df.with_columns([
            # Price distance from recent levels
            (pl.col('price') - pl.col('price').rolling_min(30)).alias('distance_from_low_30m'),
            (pl.col('price').rolling_max(30) - pl.col('price')).alias('distance_from_high_30m'),
            # Relative position in recent range
            ((pl.col('price') - pl.col('price').rolling_min(30)) / 
             (pl.col('price').rolling_max(30) - pl.col('price').rolling_min(30) + 1e-10)).alias('position_in_range_30m')
        ])
        
        # Round number effects (psychological levels)
        df = df.with_columns([
            # Distance from round numbers (approximation)
            (pl.col('price') % 1.0).alias('distance_from_round'),
            # Recent round number tests
            ((pl.col('price') % 1.0) < 0.1).cast(pl.Int32).rolling_sum(10).alias('round_number_tests')
        ])
        
        # Bid-ask spread approximation (using price volatility)
        df = df.with_columns([
            # Spread proxy
            pl.col('returns').abs().rolling_mean(3).alias('spread_proxy'),
            # Spread widening
            (pl.col('returns').abs().rolling_mean(3) > 
             pl.col('returns').abs().rolling_mean(15) * 1.5).alias('spread_widening')
        ])
        
        return df
    
    def _add_noise_signal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add noise vs signal separation features"""
        
        # Signal strength indicators
        df = df.with_columns([
            # Trend strength
            (pl.col('returns').rolling_mean(5).abs() / 
             (pl.col('returns').rolling_std(5) + 1e-10)).alias('trend_strength_5m'),
            (pl.col('returns').rolling_mean(10).abs() / 
             (pl.col('returns').rolling_std(10) + 1e-10)).alias('trend_strength_10m'),
            # Signal-to-noise ratio
            (pl.col('returns').rolling_mean(5).abs() / 
             pl.col('returns').abs().rolling_mean(5)).alias('signal_noise_ratio_5m')
        ])
        
        # Noise characteristics
        df = df.with_columns([
            # Return reversals (noise indicator)
            ((pl.col('returns') > 0) & (pl.col('returns').shift(1) < 0)).cast(pl.Int32).rolling_sum(10).alias('reversals_10m'),
            # Consecutive moves (signal indicator)
            ((pl.col('returns') > 0) & (pl.col('returns').shift(1) > 0)).cast(pl.Int32).rolling_sum(10).alias('consecutive_up_10m'),
            ((pl.col('returns') < 0) & (pl.col('returns').shift(1) < 0)).cast(pl.Int32).rolling_sum(10).alias('consecutive_down_10m')
        ])
        
        # Price efficiency measures
        df = df.with_columns([
            # Variance ratio (random walk test approximation)
            (pl.col('returns').rolling_var(10) / 
             (pl.col('returns').rolling_var(5) * 2 + 1e-10)).alias('variance_ratio_10_5'),
            # Autocorrelation approximation
            (pl.col('returns') * pl.col('returns').shift(1)).rolling_mean(10).alias('autocorr_1_approx')
        ])
        
        return df
    
    def _add_momentum_persistence_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum persistence features"""
        
        # Momentum consistency
        df = df.with_columns([
            # Consistent directional moves
            (pl.col('returns').sign().rolling_sum(5).abs() / 5).alias('momentum_consistency_5m'),
            (pl.col('returns').sign().rolling_sum(10).abs() / 10).alias('momentum_consistency_10m'),
            # Momentum strength
            (pl.col('returns').rolling_sum(5).abs() / 
             pl.col('returns').abs().rolling_sum(5)).alias('momentum_strength_5m'),
            (pl.col('returns').rolling_sum(10).abs() / 
             pl.col('returns').abs().rolling_sum(10)).alias('momentum_strength_10m')
        ])
        
        # Momentum decay/acceleration
        df = df.with_columns([
            # Recent vs older momentum
            (pl.col('returns').rolling_sum(3) / 
             (pl.col('returns').rolling_sum(10) - pl.col('returns').rolling_sum(3) + 1e-10)).alias('momentum_recent_vs_older'),
            # Momentum acceleration
            (pl.col('returns').rolling_sum(3) - pl.col('returns').rolling_sum(3).shift(3)).alias('momentum_acceleration')
        ])
        
        return df
    
    def _add_tick_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add tick-by-tick pattern features"""
        
        # Price change patterns
        df = df.with_columns([
            # Zero returns (no change)
            (pl.col('returns') == 0).cast(pl.Int32).rolling_sum(10).alias('zero_returns_10m'),
            # Small moves
            (pl.col('returns').abs() < pl.col('returns').abs().rolling_mean(30) * 0.5).cast(pl.Int32).rolling_sum(10).alias('small_moves_10m'),
            # Large moves
            (pl.col('returns').abs() > pl.col('returns').abs().rolling_mean(30) * 2).cast(pl.Int32).rolling_sum(10).alias('large_moves_10m')
        ])
        
        # Price level patterns
        df = df.with_columns([
            # Price clustering (similar price levels)
            (pl.col('price').round(6) == pl.col('price').shift(1).round(6)).cast(pl.Int32).rolling_sum(10).alias('price_clustering_10m'),
            # New highs/lows
            (pl.col('price') == pl.col('price').rolling_max(30)).cast(pl.Int32).rolling_sum(10).alias('new_highs_10m'),
            (pl.col('price') == pl.col('price').rolling_min(30)).cast(pl.Int32).rolling_sum(10).alias('new_lows_10m')
        ])
        
        return df


def process_short_term_features(input_dir: Path, output_dir: Path, limit: Optional[int] = None) -> Dict:
    """
    Process all tokens to create short-term features
    
    Args:
        input_dir: Directory containing short-term cleaned tokens
        output_dir: Directory to save features
        limit: Maximum number of tokens to process
        
    Returns:
        Processing results summary
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(exist_ok=True)
    
    # Get all token files
    token_files = list(input_dir.glob('*.parquet'))
    if limit:
        token_files = token_files[:limit]
    
    print(f"🎯 Processing {len(token_files)} tokens for short-term features")
    
    # Initialize feature engineer
    feature_engineer = ShortTermFeatureEngineer()
    
    results = {
        'processed_successfully': 0,
        'failed_processing': 0,
        'total_files': len(token_files),
        'feature_counts': []
    }
    
    for token_file in tqdm(token_files, desc="Creating short-term features"):
        try:
            token_name = token_file.stem
            
            # Load token data
            df = pl.read_parquet(token_file)
            
            # Create features
            df_features = feature_engineer.create_short_term_features(df, token_name)
            
            if len(df_features) > 0:
                # Save features
                output_file = output_dir / f"{token_name}.parquet"
                df_features.write_parquet(output_file)
                
                results['processed_successfully'] += 1
                results['feature_counts'].append(len(df_features.columns))
            else:
                results['failed_processing'] += 1
                
        except Exception as e:
            print(f"❌ Error processing {token_file.name}: {e}")
            results['failed_processing'] += 1
            continue
    
    # Calculate summary stats
    if results['feature_counts']:
        results['avg_features'] = sum(results['feature_counts']) / len(results['feature_counts'])
        results['min_features'] = min(results['feature_counts'])
        results['max_features'] = max(results['feature_counts'])
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Create short-term features for memecoin prediction')
    parser.add_argument('--input_dir', type=Path, default='data/cleaned_tokens_short_term',
                       help='Directory containing short-term cleaned tokens')
    parser.add_argument('--output_dir', type=Path, default='data/features_short_term',
                       help='Directory to save short-term features')
    parser.add_argument('--limit', type=int, help='Limit number of tokens to process')
    
    args = parser.parse_args()
    
    try:
        results = process_short_term_features(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            limit=args.limit
        )
        
        print(f"\n📊 SHORT-TERM FEATURE ENGINEERING RESULTS:")
        print(f"   ✅ Successfully processed: {results['processed_successfully']:,}")
        print(f"   ❌ Failed processing: {results['failed_processing']:,}")
        print(f"   📈 Success rate: {results['processed_successfully']/results['total_files']:.1%}")
        
        if 'avg_features' in results:
            print(f"   🔧 Average features per token: {results['avg_features']:.0f}")
            print(f"   📏 Feature range: {results['min_features']}-{results['max_features']}")
        
        print(f"\n🎯 SPECIALIZED SHORT-TERM FEATURES CREATED:")
        print(f"   • Micro-momentum (3-10 min patterns)")
        print(f"   • Volatility clustering indicators")
        print(f"   • Order flow approximation")
        print(f"   • Market microstructure signals")
        print(f"   • Noise vs signal separation")
        print(f"   • Momentum persistence measures")
        print(f"   • Tick-by-tick patterns")
        
        print(f"\n🚀 Ready for short-term model training!")
        print(f"   📁 Features saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 