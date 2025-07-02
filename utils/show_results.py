#!/usr/bin/env python3
"""
Display a summary of all generated analysis results
"""

import os
from pathlib import Path
from datetime import datetime

def show_results_summary():
    """Show a summary of all files in the utils/results directory"""
    
    results_dir = Path(__file__).parent / 'results'
    
    print("📊 UTILS ANALYSIS RESULTS SUMMARY")
    print("=" * 60)
    
    if not results_dir.exists():
        print("❌ Results directory not found. Run some analysis scripts first!")
        return
    
    # Get all files in results directory
    files = list(results_dir.glob('*'))
    
    if not files:
        print("📁 Results directory is empty. Run some analysis scripts to generate results!")
        return
    
    print(f"📁 Results directory: {results_dir}")
    print(f"🔍 Found {len(files)} files:")
    print()
    
    # Categorize files
    plots = []
    docs = []
    other = []
    
    for file in files:
        if file.suffix == '.png':
            plots.append(file)
        elif file.suffix == '.md':
            docs.append(file)
        else:
            other.append(file)
    
    # Show plots
    if plots:
        print("📈 ANALYSIS PLOTS:")
        for plot in sorted(plots):
            # Get file size and modification time
            size_mb = plot.stat().st_size / 1024 / 1024
            mod_time = datetime.fromtimestamp(plot.stat().st_mtime)
            
            print(f"  📊 {plot.name}")
            print(f"      Size: {size_mb:.2f} MB")
            print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Determine plot type
            if 'variability_analysis' in plot.name:
                print(f"      Type: 🔬 Comprehensive variability analysis")
            elif 'token_analysis' in plot.name:
                print(f"      Type: 🔍 Individual token examination")
            elif 'pattern_analysis' in plot.name:
                print(f"      Type: 🪜 Corruption pattern analysis")
            else:
                print(f"      Type: 📈 General analysis")
            print()
    
    # Show documentation
    if docs:
        print("📚 DOCUMENTATION:")
        for doc in sorted(docs):
            print(f"  📄 {doc.name}")
        print()
    
    # Show other files
    if other:
        print("📋 OTHER FILES:")
        for file in sorted(other):
            print(f"  📄 {file.name}")
        print()
    
    print("🎯 ANALYSIS TYPES AVAILABLE:")
    print("  🔬 Variability Analysis - Distinguish real market movements from straight lines")
    print("  🪜 Corruption Detection - Identify staircase artifacts vs legitimate pumps") 
    print("  🔍 Individual Token Analysis - Detailed examination with metrics")
    print()
    
    print("💡 TO GENERATE MORE RESULTS:")
    print("  python utils/run_all_tests.py                              # Run all tests")
    print("  python utils/variability_analysis/analyze_token_variability.py  # Comprehensive analysis")
    print("  python utils/corruption_detection/examine_specific_token.py     # Pattern analysis")

if __name__ == "__main__":
    show_results_summary()