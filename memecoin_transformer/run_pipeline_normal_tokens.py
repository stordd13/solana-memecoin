#!/usr/bin/env python3
"""
Pipeline script for processing normal behavior tokens
Runs feature engineering and data scaling on tokens from data/processed/normal_behavior_tokens
"""

import subprocess
import sys
from pathlib import Path

def run_step(step_name, script_path):
    print(f"\n{'='*60}")
    print(f"🚀 Running: {step_name}")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in {step_name}")
        sys.exit(1)
    
    print(f"✅ {step_name} complete!")

def main():
    print("🎯 MEMECOIN TRANSFORMER PIPELINE - NORMAL BEHAVIOR TOKENS")
    print("=========================================================")
    print("This pipeline will process all tokens from:")
    print("data/processed/normal_behavior_tokens/")
    print("")
    
    # Check if we should run with limited tokens for testing
    response = input("Run with all 3,427 tokens? (y/n) or enter number for subset: ")
    
    max_tokens = None
    if response.lower() == 'n':
        print("Exiting...")
        sys.exit(0)
    elif response.lower() != 'y':
        try:
            max_tokens = int(response)
            print(f"Will process {max_tokens} tokens")
            
            # We need to modify the feature_engineering script temporarily
            # For now, let's inform the user
            print("\n⚠️  Note: To run with a subset, please edit feature_engineering.py")
            print("   Change line 458 to:")
            print(f"   df = load_tokens_from_directory(data_dir, max_tokens={max_tokens})")
            print("")
            response = input("Have you made this change? (y/n): ")
            if response.lower() != 'y':
                print("Please make the change and run again.")
                sys.exit(0)
        except ValueError:
            print("Invalid input. Please enter 'y', 'n', or a number.")
            sys.exit(1)
    
    # 1. Feature engineering
    print("\n📊 Step 1: Feature Engineering")
    run_step("Feature Engineering", "data_preparation/feature_engineering.py")
    
    # 2. Data scaling
    print("\n⚖️ Step 2: Data Scaling")
    run_step("Data Scaling", "data_preparation/data_scaling.py")
    
    print("\n✨ Pipeline complete!")
    print("\n📁 Output files created:")
    print("  - data/processed/memecoin_features_from_normal_tokens.parquet")
    print("  - data/processed/sequences_from_normal_tokens.npz")
    print("  - data/processed/sequences_scaled_normal_tokens.npz")
    print("  - data/processed/sequences_metadata_normal_tokens.json")
    print("  - data/processed/scaler_params_normal_tokens.json")
    
    print("\n🚀 Ready for training! Use:")
    print("  python training/train.py --data-path data/processed/sequences_scaled_normal_tokens.npz")

if __name__ == "__main__":
    main()