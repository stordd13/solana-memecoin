"""
Formatting utilities for Streamlit applications.
Provides consistent number, percentage, and currency formatting.
"""

import numpy as np
from typing import Union


def format_large_number(value: Union[int, float], threshold: float = 1_000_000, precision: int = 2) -> str:
    """
    Format large numbers with scientific notation or appropriate units.
    
    Args:
        value: Number to format
        threshold: Threshold above which to use scientific notation (default: 1M)
        precision: Decimal places for scientific notation (default: 2)
        
    Returns:
        Formatted string with appropriate notation
        
    Examples:
        >>> format_large_number(1500)
        '1,500'
        >>> format_large_number(1500000)
        '1.50e+06'
        >>> format_large_number(2.5e9)
        '2.50e+09'
    """
    if np.isnan(value) or np.isinf(value):
        return str(value)
    
    abs_value = abs(value)
    
    if abs_value >= threshold:
        # Use scientific notation for very large numbers
        return f"{value:.{precision}e}"
    elif abs_value >= 1000:
        # Use comma formatting for thousands
        if isinstance(value, int) or value.is_integer():
            return f"{int(value):,}"
        else:
            return f"{value:,.{precision}f}"
    elif abs_value >= 1:
        # Regular formatting for normal numbers
        if isinstance(value, int) or value.is_integer():
            return str(int(value))
        else:
            return f"{value:.{precision}f}"
    else:
        # Small numbers with appropriate precision
        if abs_value == 0:
            return "0"
        elif abs_value < 0.001:
            return f"{value:.{precision}e}"
        else:
            # Remove trailing zeros for small decimals
            formatted = f"{value:.{min(precision + 2, 6)}f}"
            return formatted.rstrip('0').rstrip('.')


def format_percentage(value: Union[int, float], precision: int = 2, threshold: float = 10000) -> str:
    """
    Format percentage values with scientific notation for extreme values.
    
    Args:
        value: Percentage value (e.g., 0.15 for 15% or 15 for 15%)
        precision: Decimal places (default: 2)
        threshold: Threshold above which to use scientific notation (default: 10000%)
        
    Returns:
        Formatted percentage string
        
    Examples:
        >>> format_percentage(0.15)
        '15.00%'
        >>> format_percentage(150.5)
        '150.50%'
        >>> format_percentage(1500000)
        '1.50e+06%'
    """
    if np.isnan(value) or np.isinf(value):
        return f"{value}%"
    
    # Handle both decimal (0.15) and percentage (15) inputs
    if abs(value) <= 1 and abs(value) > 0.001:
        # Assume decimal format, convert to percentage
        percentage_value = value * 100
    else:
        # Assume already in percentage format
        percentage_value = value
    
    if abs(percentage_value) >= threshold:
        return f"{percentage_value:.{precision}e}%"
    else:
        return f"{percentage_value:.{precision}f}%"


def format_currency(value: Union[int, float], currency: str = "$", threshold: float = 1_000_000, precision: int = 2) -> str:
    """
    Format currency values with appropriate notation.
    
    Args:
        value: Currency value
        currency: Currency symbol (default: "$")
        threshold: Threshold for scientific notation (default: 1M)
        precision: Decimal places (default: 2)
        
    Returns:
        Formatted currency string
        
    Examples:
        >>> format_currency(1500)
        '$1,500.00'
        >>> format_currency(1500000)
        '$1.50e+06'
        >>> format_currency(0.0001)
        '$0.0001'
    """
    if np.isnan(value) or np.isinf(value):
        return f"{currency}{value}"
    
    abs_value = abs(value)
    
    if abs_value >= threshold:
        # Scientific notation for very large amounts
        return f"{currency}{value:.{precision}e}"
    elif abs_value >= 1:
        # Standard currency formatting
        return f"{currency}{value:,.{precision}f}"
    elif abs_value == 0:
        return f"{currency}0.00"
    else:
        # Small amounts with appropriate precision
        if abs_value < 0.0001:
            return f"{currency}{value:.{precision}e}"
        else:
            # Determine appropriate precision for small amounts
            small_precision = max(precision, 4)
            return f"{currency}{value:.{small_precision}f}"


def format_file_count(count: int) -> str:
    """
    Format file counts with appropriate units.
    
    Args:
        count: Number of files
        
    Returns:
        Formatted file count string
        
    Examples:
        >>> format_file_count(1500)
        '1,500 files'
        >>> format_file_count(1500000)
        '1.50e+06 files'
    """
    if count == 1:
        return "1 file"
    
    formatted_count = format_large_number(count, threshold=10000, precision=2)
    return f"{formatted_count} files"


def format_data_points(count: int) -> str:
    """
    Format data point counts with appropriate units.
    
    Args:
        count: Number of data points
        
    Returns:
        Formatted data points string
        
    Examples:
        >>> format_data_points(1500000)
        '1.50e+06 points'
    """
    if count == 1:
        return "1 point"
    
    formatted_count = format_large_number(count, threshold=1000000, precision=2)
    return f"{formatted_count} points"


def format_time_duration(minutes: float) -> str:
    """
    Format time duration in appropriate units.
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted duration string
        
    Examples:
        >>> format_time_duration(90)
        '1.5 hours'
        >>> format_time_duration(1500)
        '25.0 hours'
        >>> format_time_duration(100000)
        '1.67e+03 hours'
    """
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes / 60
        return f"{hours:.1f} hours"
    elif minutes < 10080:  # Less than 7 days
        days = minutes / 1440
        return f"{days:.1f} days"
    else:
        hours = minutes / 60
        if hours >= 1000:  # Use scientific notation for >1000 hours
            return f"{hours:.2e} hours"
        else:
            return f"{hours:.1f} hours"