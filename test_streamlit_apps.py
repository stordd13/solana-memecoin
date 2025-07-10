#!/usr/bin/env python3
"""
Test script for Streamlit applications
Tests components and formatting without running full Streamlit server
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock streamlit to avoid full server startup
sys.modules['streamlit'] = Mock()
import streamlit as st

# Setup basic mocks
st.sidebar = Mock()
st.session_state = {}
st.title = Mock()
st.header = Mock()
st.subheader = Mock()
st.info = Mock()
st.warning = Mock()
st.error = Mock()
st.success = Mock()
st.metric = Mock()
st.selectbox = Mock(return_value=0)
st.multiselect = Mock(return_value=[])
st.radio = Mock(return_value="Single Token")
st.button = Mock(return_value=False)
st.rerun = Mock()

def test_components():
    """Test the shared components"""
    print("\n=== Testing Shared Components ===")
    
    try:
        from streamlit_utils.components import DataSourceManager, TokenSelector, NavigationManager
        
        # Test DataSourceManager
        print("✅ DataSourceManager imports successfully")
        
        # Test TokenSelector with mock data loader
        mock_data_loader = Mock()
        mock_data_loader.get_available_tokens.return_value = [
            {'symbol': 'TEST1', 'file': 'test1.parquet'},
            {'symbol': 'TEST2', 'file': 'test2.parquet'}
        ]
        
        token_selector = TokenSelector(mock_data_loader, key_prefix="test_")
        print("✅ TokenSelector initializes successfully")
        
        # Test NavigationManager
        nav_manager = NavigationManager(session_key_prefix="test_")
        print("✅ NavigationManager initializes successfully")
        
        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_app_imports():
    """Test that all apps import without errors"""
    print("\n=== Testing App Imports ===")
    
    apps = [
        ("data_analysis.app", "data_analysis/app.py"),
        ("time_series.autocorrelation_app", "time_series/autocorrelation_app.py"),
        ("quant_analysis.quant_app", "quant_analysis/quant_app.py"),
        ("feature_engineering.app", "feature_engineering/app.py")
    ]
    
    results = []
    
    for module_name, file_path in apps:
        try:
            # Import the module
            if module_name == "data_analysis.app":
                from data_analysis.app import main
            elif module_name == "time_series.autocorrelation_app":
                from time_series.autocorrelation_app import main
            elif module_name == "quant_analysis.quant_app":
                from quant_analysis.quant_app import main
            elif module_name == "feature_engineering.app":
                # This app doesn't have a main function, just import it
                import feature_engineering.app
            
            print(f"✅ {file_path} imports successfully")
            results.append(True)
        except Exception as e:
            print(f"❌ {file_path} import failed: {e}")
            results.append(False)
    
    return all(results)

def test_formatting_integration():
    """Test that formatting functions work with app contexts"""
    print("\n=== Testing Formatting Integration ===")
    
    try:
        from streamlit_utils.formatting import (
            format_large_number, format_percentage, 
            format_data_points, format_file_count
        )
        
        # Test typical values from memecoin analysis
        test_cases = [
            (1500000, "1.50e+06", "Large token count"),
            (25000000, "2.50e+07", "Large data points"),
            (15000.5, "1.50e+04%", "Large percentage"),
            (500, "500 files", "File count"),
        ]
        
        functions = [format_large_number, format_large_number, format_percentage, format_file_count]
        
        all_passed = True
        for i, (value, expected, description) in enumerate(test_cases):
            result = functions[i](value)
            if expected in result or result == expected:
                print(f"✅ {description}: {value} → {result}")
            else:
                print(f"❌ {description}: {value} → {result} (expected: {expected})")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Formatting integration test failed: {e}")
        return False

def test_data_analysis_components():
    """Test specific components in data_analysis app"""
    print("\n=== Testing Data Analysis App Components ===")
    
    try:
        # Mock session state for data_analysis app
        st.session_state.update({
            'data_loaded': False,
            'data_loader': None,
            'quality_analyzer': None,
            'price_analyzer': None
        })
        
        # Import and test components
        from streamlit_utils.components import DataSourceManager
        
        # Test data source manager with project structure
        dsm = DataSourceManager()
        available_folders = dsm.get_available_subfolders()
        print(f"✅ Found {len(available_folders)} data subfolders")
        
        return True
    except Exception as e:
        print(f"❌ Data analysis component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("STREAMLIT APPS TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_components()
    all_passed &= test_app_imports()  
    all_passed &= test_formatting_integration()
    all_passed &= test_data_analysis_components()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL STREAMLIT TESTS PASSED!")
        print("Note: This tests imports and component initialization.")
        print("Full UI testing requires manual verification with `streamlit run`.")
    else:
        print("❌ SOME STREAMLIT TESTS FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())