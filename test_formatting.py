#!/usr/bin/env python3
"""
Test script for scientific notation formatting utilities
Tests all formatting functions with edge cases
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from streamlit_utils.formatting import (
    format_large_number, 
    format_percentage, 
    format_currency, 
    format_file_count, 
    format_data_points,
    format_time_duration
)


def test_format_large_number():
    """Test format_large_number with various edge cases"""
    print("\n=== Testing format_large_number ===")
    
    test_cases = [
        # (input, expected_output, description)
        (0, "0", "Zero"),
        (1, "1", "One"),
        (100, "100", "Hundred"),
        (1000, "1,000", "Thousand"),
        (10000, "10,000", "Ten thousand"),
        (100000, "100,000", "Hundred thousand"),
        (999999, "999,999", "Just under threshold"),
        (1000000, "1.00e+06", "Exactly at threshold"),
        (1500000, "1.50e+06", "One and half million"),
        (10000000, "1.00e+07", "Ten million"),
        (1.23456789e10, "1.23e+10", "Large scientific"),
        (-1500000, "-1.50e+06", "Negative large"),
        (0.001, "0.001", "Small decimal"),
        (0.0001, "1.00e-04", "Very small decimal"),
        (np.nan, "nan", "NaN value"),
        (np.inf, "inf", "Infinity"),
        (-np.inf, "-inf", "Negative infinity"),
    ]
    
    passed = 0
    failed = 0
    
    for value, expected, description in test_cases:
        result = format_large_number(value)
        if result == expected:
            print(f"✅ {description}: {value} → {result}")
            passed += 1
        else:
            print(f"❌ {description}: {value} → {result} (expected: {expected})")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


def test_format_percentage():
    """Test format_percentage with various edge cases"""
    print("\n=== Testing format_percentage ===")
    
    test_cases = [
        # (input, expected_output, description)
        (0, "0.00%", "Zero"),
        (0.15, "15.00%", "15% as decimal"),
        (15, "15.00%", "15% as percentage"),
        (100, "100.00%", "100%"),
        (150.5, "150.50%", "150.5%"),
        (1000, "1000.00%", "1000%"),
        (9999, "9999.00%", "Just under threshold"),
        (10000, "1.00e+04%", "At threshold"),
        (1500000, "1.50e+06%", "Large percentage"),
        (-50, "-50.00%", "Negative percentage"),
        (np.nan, "nan%", "NaN value"),
        (np.inf, "inf%", "Infinity"),
    ]
    
    passed = 0
    failed = 0
    
    for value, expected, description in test_cases:
        result = format_percentage(value)
        if result == expected:
            print(f"✅ {description}: {value} → {result}")
            passed += 1
        else:
            print(f"❌ {description}: {value} → {result} (expected: {expected})")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


def test_format_currency():
    """Test format_currency with various edge cases"""
    print("\n=== Testing format_currency ===")
    
    test_cases = [
        # (input, expected_output, description)
        (0, "$0.00", "Zero"),
        (1, "$1.00", "One dollar"),
        (1500, "$1,500.00", "Thousand"),
        (999999.99, "$999,999.99", "Just under threshold"),
        (1000000, "$1.00e+06", "At threshold"),
        (1500000, "$1.50e+06", "Large amount"),
        (0.0001, "$0.0001", "Small amount"),
        (0.00001, "$1.00e-05", "Very small amount"),
        (np.nan, "$nan", "NaN value"),
        (np.inf, "$inf", "Infinity"),
    ]
    
    passed = 0
    failed = 0
    
    for value, expected, description in test_cases:
        result = format_currency(value)
        if result == expected:
            print(f"✅ {description}: {value} → {result}")
            passed += 1
        else:
            print(f"❌ {description}: {value} → {result} (expected: {expected})")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


def test_format_file_count():
    """Test format_file_count with various edge cases"""
    print("\n=== Testing format_file_count ===")
    
    test_cases = [
        # (input, expected_output, description)
        (0, "0 files", "Zero files"),
        (1, "1 file", "Single file"),
        (100, "100 files", "Hundred files"),
        (1500, "1,500 files", "Thousand files"),
        (9999, "9,999 files", "Just under threshold"),
        (10000, "1.00e+04 files", "At threshold"),
        (1000000, "1.00e+06 files", "Million files"),
    ]
    
    passed = 0
    failed = 0
    
    for value, expected, description in test_cases:
        result = format_file_count(value)
        if result == expected:
            print(f"✅ {description}: {value} → {result}")
            passed += 1
        else:
            print(f"❌ {description}: {value} → {result} (expected: {expected})")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


def test_format_data_points():
    """Test format_data_points with various edge cases"""
    print("\n=== Testing format_data_points ===")
    
    test_cases = [
        # (input, expected_output, description)
        (0, "0 points", "Zero points"),
        (1, "1 point", "Single point"),
        (1000, "1,000 points", "Thousand points"),
        (999999, "999,999 points", "Just under threshold"),
        (1000000, "1.00e+06 points", "At threshold"),
        (50000000, "5.00e+07 points", "50 million points"),
    ]
    
    passed = 0
    failed = 0
    
    for value, expected, description in test_cases:
        result = format_data_points(value)
        if result == expected:
            print(f"✅ {description}: {value} → {result}")
            passed += 1
        else:
            print(f"❌ {description}: {value} → {result} (expected: {expected})")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


def test_format_time_duration():
    """Test format_time_duration with various edge cases"""
    print("\n=== Testing format_time_duration ===")
    
    test_cases = [
        # (input, expected_output, description)
        (0, "0.0 minutes", "Zero minutes"),
        (30, "30.0 minutes", "Half hour"),
        (59, "59.0 minutes", "Just under an hour"),
        (60, "1.0 hours", "One hour"),
        (90, "1.5 hours", "One and half hours"),
        (1440, "1.0 days", "One day"),
        (10080, "168.0 hours", "One week"),
        (100000, "1.67e+03 hours", "Large duration"),
    ]
    
    passed = 0
    failed = 0
    
    for value, expected, description in test_cases:
        result = format_time_duration(value)
        if result == expected:
            print(f"✅ {description}: {value} → {result}")
            passed += 1
        else:
            print(f"❌ {description}: {value} → {result} (expected: {expected})")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all formatting tests"""
    print("=" * 60)
    print("SCIENTIFIC NOTATION FORMATTING TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_format_large_number()
    all_passed &= test_format_percentage()
    all_passed &= test_format_currency()
    all_passed &= test_format_file_count()
    all_passed &= test_format_data_points()
    all_passed &= test_format_time_duration()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())