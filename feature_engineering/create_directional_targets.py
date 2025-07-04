"""
Create Directional Targets for Hyperparameter Tuning

This script adds directional targets (label_Xm columns) to pre-engineered features.
It should be run AFTER feature_engineering/advanced_feature_engineering.py.

USAGE:
    python feature_engineering/create_directional_targets.py --input_dir data/features --output_dir data/features_with_targets

The script processes all feature files and adds:
- label_15m, label_30m, etc. (1 if price goes up, 0 if down)
- return_15m, return_30m, etc. (percentage return)
"""

import polars as pl
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def create_directional_targets(df: pl.DataFrame, horizons: List[int]) -> pl.DataFrame:
    """
    Create directional labels and returns for multiple horizons
    
    Args:
        df: DataFrame with 'price' column
        horizons: List of prediction horizons in minutes
        
    Returns:
        DataFrame with added label_Xm and return_Xm columns
    """
    if 'price' not in df.columns:
        print("⚠️ Warning: No 'price' column found")
        return df
        
    # Add labels and returns for each horizon
    for h in horizons:
        df = df.with_columns([
            # Directional target: 1 if price goes up, 0 if down
            (pl.col('price').shift(-h) > pl.col('price')).cast(pl.Int32).alias(f'label_{h}m'),
            # Percentage return
            ((pl.col('price').shift(-h) - pl.col('price')) / pl.col('price')).alias(f'return_{h}m')
        ])
    
    return df


def process_features_directory(input_dir: Path, output_dir: Path, horizons: List[int]):
    """Process all feature files and add directional targets"""
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each category
    categories = ["normal_behavior_tokens", "tokens_with_extremes", "dead_tokens"]
    
    for category in categories:
        input_cat_dir = input_dir / category
        output_cat_dir = output_dir / category
        
        if not input_cat_dir.exists():
            print(f"⚠️ Category directory not found: {input_cat_dir}")
            continue
            
        output_cat_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all parquet files
        parquet_files = list(input_cat_dir.glob("*.parquet"))
        print(f"\n📁 Processing {category}: {len(parquet_files)} files")
        
        for file_path in tqdm(parquet_files, desc=f"Processing {category}"):
            try:
                # Load features
                df = pl.read_parquet(file_path)
                
                # Skip if too short
                if len(df) < max(horizons) + 10:
                    print(f"⚠️ Skipping {file_path.name}: too short ({len(df)} rows)")
                    continue
                
                # Add directional targets
                df_with_targets = create_directional_targets(df, horizons)
                
                # Save to output directory
                output_path = output_cat_dir / file_path.name
                df_with_targets.write_parquet(output_path)
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                continue
    
    print(f"\n✅ Target creation complete!")
    print(f"📁 Output saved to: {output_dir}")


def main():
    """Main function for creating directional targets"""
    
    parser = argparse.ArgumentParser(description='Create Directional Targets for Features')
    parser.add_argument('--input_dir', type=str, default='data/features',
                       help='Input features directory (default: data/features)')
    parser.add_argument('--output_dir', type=str, default='data/features_with_targets',
                       help='Output directory (default: data/features_with_targets)')
    parser.add_argument('--horizons', type=int, nargs='+', 
                       default=[15, 30, 60, 120, 240, 360, 720],
                       help='Prediction horizons in minutes (default: 15 30 60 120 240 360 720)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    horizons = args.horizons
    
    print(f"🎯 CREATING DIRECTIONAL TARGETS")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🎯 Horizons: {horizons} minutes")
    print(f"{'='*60}")
    
    try:
        process_features_directory(input_dir, output_dir, horizons)
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()