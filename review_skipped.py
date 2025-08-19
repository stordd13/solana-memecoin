#!/usr/bin/env python3
"""
Utility script to review and re-process skipped files.
"""

import shutil
from pathlib import Path

def review_skipped_files():
    """Move skipped files back to unprocessed pool for review."""
    skipped_dir = Path("data/handpicked/skipped")
    raw_dir = Path("data/raw/dataset")
    
    if not skipped_dir.exists():
        print("No skipped directory found.")
        return
    
    skipped_files = list(skipped_dir.glob("*.parquet"))
    
    if not skipped_files:
        print("No skipped files to review.")
        return
    
    print(f"Found {len(skipped_files)} skipped files.")
    response = input("Move them back for review? (y/n): ")
    
    if response.lower() == 'y':
        for file in skipped_files:
            # Just delete from skipped directory
            # The main app will automatically pick them up as unprocessed
            file.unlink()
            print(f"Removed from skipped: {file.name}")
        
        print(f"\n✅ {len(skipped_files)} files ready for review.")
        print("Run the labeling app again to process them.")
    else:
        print("Skipped files remain in place.")

if __name__ == "__main__":
    review_skipped_files()