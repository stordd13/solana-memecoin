import polars as pl
from pathlib import Path
from tqdm import tqdm

FEATURE_DIR = Path('../data/features')
CATEGORIES = [
    'normal_behavior_tokens',
    'tokens_with_extremes',
    'dead_tokens'
]

def investigate_files():
    print("="*60)
    print("🔬 Investigating Feature Files for Non-Finite Values 🔬")
    print("="*60)
    print(f"Searching in: {FEATURE_DIR.resolve()}")

    all_files = []
    for category in CATEGORIES:
        cat_dir = FEATURE_DIR / category
        if not cat_dir.exists():
            print(f"⚠️  Warning: Category directory not found: {cat_dir}")
            continue
        files_in_cat = list(cat_dir.glob('*.parquet'))
        print(f"Found {len(files_in_cat)} files in '{category}'")
        all_files.extend(files_in_cat)

    if not all_files:
        print("\\n❌ ERROR: No feature files found to investigate. Did the pipeline run?")
        return

    print(f"\\nTotal files to investigate: {len(all_files)}")

    files_with_issues = {}
    total_non_finite_count = 0

    for file_path in tqdm(all_files, desc="Analyzing files"):
        try:
            df = pl.read_parquet(file_path)
            
            numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64]]
            
            issues_in_file = []
            
            for col in numeric_cols:
                # Using Polars expressions to check for non-finite values
                non_finite_count = df.select(
                    (~pl.col(col).is_finite()).sum()
                ).item()

                if non_finite_count > 0:
                    issues_in_file.append(f"Column '{col}' has {non_finite_count} non-finite (NaN/inf) values.")
                    total_non_finite_count += non_finite_count
            
            if issues_in_file:
                files_with_issues[str(file_path)] = issues_in_file

        except Exception as e:
            print(f"\\n❌ Error reading or processing file {file_path}: {e}")

    print("\\n" + "="*60)
    print("📊 Investigation Summary 📊")
    print("="*60)

    if not files_with_issues:
        print("✅ SUCCESS! All feature files are clean. No NaN or infinity values were found.")
        print("\\nConclusion: The issue is likely within the ML training script itself, not the feature files.")
    else:
        print(f"❌ FOUND ISSUES in {len(files_with_issues)} out of {len(all_files)} files.")
        print(f"   Total non-finite values found: {total_non_finite_count}")
        print("\\nThis confirms that the feature engineering step is still producing unclean files.")
        print("The problem lies in `feature_engineering/advanced_feature_engineering.py`.")

        print("\\n--- Files with Issues ---")
        for i, (file, issues) in enumerate(files_with_issues.items()):
            if i >= 10:  # Print details for first 10 files
                print(f"\\n... and {len(files_with_issues) - 10} more files.")
                break
            print(f"\\n📄 File: {file}")
            for issue in issues:
                print(f"   - {issue}")

if __name__ == "__main__":
    investigate_files() 