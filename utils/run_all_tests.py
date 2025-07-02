#!/usr/bin/env python3
"""
Run all utility tests to validate the improved data processing pipeline
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_script(script_path, description):
    """Run a script and capture its output"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"Running: {script_path}")
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=30, cwd=project_root)
        
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\n📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ STDERR:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Script timed out (30s limit)")
        return False
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def main():
    """Run all utility tests"""
    print("🚀 MEMECOIN UTILS TEST SUITE")
    print("="*80)
    print("Testing all utility scripts to validate the improved pipeline")
    
    tests = [
        {
            'script': 'utils/corruption_detection/test_specific_extreme_token.py',
            'description': 'Test Extreme Token Corruption Detection'
        },
        {
            'script': 'utils/corruption_detection/test_improved_corruption_detection.py', 
            'description': 'Test Improved Corruption Detection on Multiple Tokens'
        }
    ]
    
    results = []
    
    for test in tests:
        script_path = project_root / test['script']
        if script_path.exists():
            success = run_script(script_path, test['description'])
            results.append({
                'name': test['description'],
                'success': success
            })
        else:
            print(f"❌ Script not found: {script_path}")
            results.append({
                'name': test['description'], 
                'success': False
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"  {status} {result['name']}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The improved pipeline is working correctly.")
        print("\n📋 Key Improvements Validated:")
        print("  ✅ Legitimate massive pumps are preserved")
        print("  ✅ Staircase artifacts are correctly detected and removed")
        print("  ✅ Temporal pattern analysis distinguishes real vs fake movements")
        print("  ✅ Multi-granularity approach considers post-move volatility")
        print("\n📊 Analysis Results:")
        print("  📁 All plots and charts saved to: utils/results/")
        print("  📈 View detailed visualizations for pattern validation")
        print("\n🚀 Ready to run the full pipeline with confidence!")
        print("   Command: python run_pipeline.py --fast")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the output above.")

if __name__ == "__main__":
    main()