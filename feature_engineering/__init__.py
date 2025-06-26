"""
Feature Engineering Package for Memecoin Analysis

This package contains advanced feature engineering modules for extracting
meaningful patterns and relationships from cleaned memecoin price data.

Modules:
- advanced_feature_engineering: Comprehensive feature extraction
- correlation_analysis: Multi-token relationship analysis  
- roadmap_dashboard: Interactive feature engineering dashboard

Usage:
    from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineer
    from feature_engineering.correlation_analysis import TokenCorrelationAnalyzer
"""

from .advanced_feature_engineering import AdvancedFeatureEngineer, batch_feature_engineering
from .correlation_analysis import TokenCorrelationAnalyzer, load_tokens_for_correlation

__version__ = "1.0.0"
__author__ = "Memecoin Analysis Team"

__all__ = [
    "AdvancedFeatureEngineer",
    "batch_feature_engineering", 
    "TokenCorrelationAnalyzer",
    "load_tokens_for_correlation"
] 