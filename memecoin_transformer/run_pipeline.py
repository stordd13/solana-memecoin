#!/usr/bin/env python3
"""
Script simple pour lancer tout le pipeline
"""

import subprocess
import sys
import yaml
from pathlib import Path

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

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
    config = load_config()
    
    print("🎯 MEMECOIN TRANSFORMER PIPELINE")
    print("================================")
    
    # Check if we need to run data prep
    response = input("\nRun data preparation? (y/n): ")
    
    if response.lower() == 'y':
        # 1. Token scoring
        run_step("Token Scoring", "data_preparation/create_token_scores.py")
        
        # 2. Feature engineering
        run_step("Feature Engineering", "data_preparation/feature_engineering.py")
        
        # 3. Data scaling
        run_step("Data Scaling", "data_preparation/data_scaling.py")
    
    # 4. Training
    response = input("\nRun training? (y/n): ")
    if response.lower() == 'y':
        run_step("Model Training", "training/train.py")
    
    print("\n✨ Pipeline complete!")

if __name__ == "__main__":
    main()