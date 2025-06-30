"""
Complete Pipeline Runner for Memecoin Data Processing
Runs data analysis, cleaning, and feature engineering without Streamlit
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import required modules
from data_analysis.data_loader import DataLoader
from data_analysis.data_quality import DataQualityAnalyzer
from data_cleaning.clean_tokens import clean_all_categories
from feature_engineering.advanced_feature_engineering import main as run_feature_engineering

# Data folders you want to include under data/raw/
RAW_SUBFOLDERS = [
    "raw/dataset",       # original dataset
    "raw/dataset-fresh"  # newly added dataset with 10k extra tokens
]

def run_data_quality_analysis(raw_subfolders: list[str] = None):
    """Run data quality analysis on one or more raw subfolders and export categories"""

    if raw_subfolders is None:
        raw_subfolders = RAW_SUBFOLDERS

    print("\n" + "="*60)
    print("STEP 1: DATA QUALITY ANALYSIS & CATEGORIZATION")
    print("="*60)
    
    analyzer = DataQualityAnalyzer()
    
    # Accumulate quality reports across all subfolders
    quality_reports = {}
    processed_symbols = set()

    for subfolder in raw_subfolders:
        # Skip folders that don't exist
        data_loader = DataLoader(subfolder=subfolder)
        if not data_loader.base_path.exists():
            print(f"⚠️  Skipping missing folder: data/{subfolder}")
            continue

        print(f"\n📂 Processing {subfolder} ...")

        available_tokens = data_loader.get_available_tokens()
        print(f"Found {len(available_tokens)} tokens in data/{subfolder}")

        # Analyze each token
        for i, token_info in enumerate(available_tokens):
            symbol = token_info['symbol']

            # Skip duplicates across folders (keep first occurrence)
            if symbol in processed_symbols:
                continue

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(available_tokens)} tokens in {subfolder} …")

            try:
                df = data_loader.get_token_data(symbol)
                if df is not None and not df.is_empty():
                    report = analyzer.analyze_single_file(df, symbol)
                    quality_reports[symbol] = report
                    processed_symbols.add(symbol)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} ({subfolder}): {e}")
                continue

    print(f"\nSuccessfully analyzed {len(quality_reports)} unique tokens across all folders")
    
    # Export all categories with mutual exclusivity
    print("\n📊 Exporting tokens to mutually exclusive categories...")
    try:
        exported_results = analyzer.export_all_categories_mutually_exclusive(quality_reports)
        
        # Display results
        print("\n✅ Export Summary:")
        total_exported = sum(len(tokens) for tokens in exported_results.values())
        for category, tokens in exported_results.items():
            if tokens:
                print(f"  {category}: {len(tokens)} tokens")
        print(f"\nTotal tokens exported: {total_exported}")
        
        return True, quality_reports
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False, quality_reports


def run_data_cleaning():
    """Run data cleaning on all categories"""
    print("\n" + "="*60)
    print("STEP 2: DATA CLEANING")
    print("="*60)
    
    try:
        # Run category-aware cleaning
        summary = clean_all_categories(limit_per_category=None)
        
        # Display summary
        print("\n✅ Cleaning Summary:")
        print(f"  Total files processed: {summary.get('total_files_processed', 0)}")
        print(f"  Successfully cleaned: {summary.get('total_successfully_cleaned', 0)}")
        print(f"  Success rate: {summary.get('overall_success_rate', 0):.1f}%")
        
        # Show per-category results
        if 'category_results' in summary:
            print("\n📁 Per-category results:")
            for category, result in summary['category_results'].items():
                if isinstance(result, dict) and 'successfully_cleaned' in result:
                    total_processed = result.get('total_files_processed', 0)
                    print(f"  {category}: {result['successfully_cleaned']}/{total_processed} cleaned")
                elif isinstance(result, dict) and 'error' in result:
                    print(f"  {category}: {result['error']}")
                elif isinstance(result, dict) and 'warning' in result:
                    print(f"  {category}: {result['warning']}")
                else:
                    print(f"  {category}: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        return False


def run_feature_engineering_step(fast_mode: bool = False):
    """Run feature engineering on cleaned data
    Parameters
    ----------
    fast_mode : bool
        If True, invoke the feature-engineering module in fast mode (rolling features only).
    """
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    try:
        # Run feature engineering (propagate fast_mode flag)
        run_feature_engineering(fast_mode=fast_mode)
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return False


def main(fast_mode: bool = False):
    """Run the complete pipeline
    Parameters
    ----------
    fast_mode : bool
        If True, use fast mode for feature engineering (rolling features only) to save time.
    """
    start_time = datetime.now()
    
    print("\n" + "🚀"*20)
    print("MEMECOIN DATA PROCESSING PIPELINE")
    print("🚀"*20)
    print(f"\nStarting pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Raw data path: data/raw/dataset")
    
    # Track success of each step
    pipeline_status = {
        'data_quality': False,
        'data_cleaning': False,
        'feature_engineering': False
    }
    
    # Step 1: Data Quality Analysis & Export
    success, quality_reports = run_data_quality_analysis(RAW_SUBFOLDERS)
    pipeline_status['data_quality'] = success
    
    if not success:
        print("\n❌ Data quality analysis failed. Stopping pipeline.")
        return
    
    # Step 2: Data Cleaning
    print("\n⏳ Proceeding to data cleaning...")
    success = run_data_cleaning()
    pipeline_status['data_cleaning'] = success
    
    if not success:
        print("\n❌ Data cleaning failed. Stopping pipeline.")
        return
    
    # Step 3: Feature Engineering
    print("\n⏳ Proceeding to feature engineering...")
    success = run_feature_engineering_step(fast_mode=fast_mode)
    pipeline_status['feature_engineering'] = success
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Duration: {duration}")
    print("\nStep Status:")
    for step, status in pipeline_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {step.replace('_', ' ').title()}")
    
    if all(pipeline_status.values()):
        print("\n🎉 Pipeline completed successfully!")
        print("\n📁 Output locations:")
        print("  - Categorized tokens: data/processed/")
        print("  - Cleaned data: data/cleaned/")
        print("  - Rolling features (ML-safe): data/features/")
        print("\n🧠 Clean Architecture Benefits:")
        print("  - Rolling features: Saved to data/features/ (no data leakage)")
        print("  - Global features: Computed on-demand in Streamlit")
        print("  - Clean separation: Impossible to accidentally use global features in ML")
        print("\n🚀 Ready to train ML models!")
        print("  py ML/directional_models/train_lightgbm_model.py")
        print("  py ML/directional_models/train_lightgbm_model_medium_term.py")
        print("  py ML/directional_models/train_unified_lstm_model.py")
    else:
        print("\n⚠️ Pipeline completed with errors. Please check the logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memecoin data-processing pipeline")
    parser.add_argument('--fast', '--fast_mode', action='store_true', dest='fast_mode',
                        help='Enable fast mode (rolling features only)')
    args = parser.parse_args()

    main(fast_mode=args.fast_mode) 