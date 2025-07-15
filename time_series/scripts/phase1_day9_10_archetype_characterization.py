#!/usr/bin/env python3
# phase1_day9_10_archetype_characterization.py
"""
Phase 1 Day 9-10: Archetype Characterization

CEO Roadmap Implementation:
- Load stability testing results from Day 7-8
- Use consensus labels and stability confidence scores
- Generate markdown documentation for each archetype
- Create archetype traits, examples, and trading strategies
- Cross-validate against Solana trading patterns
- Export archetype definitions for Phase 2 usage

Usage:
    python phase1_day9_10_archetype_characterization.py --stability-results PATH [--output-dir PATH]
    
Interactive Mode:
    python phase1_day9_10_archetype_characterization.py --interactive
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    ClusteringEngine, ResultsManager, GradioVisualizer, 
    extract_features_from_returns, ESSENTIAL_FEATURES
)


class ArchetypeCharacterizationAnalyzer:
    """
    Implements CEO roadmap Phase 1 Day 9-10: Archetype Characterization.
    Generates comprehensive documentation for each discovered behavioral archetype.
    """
    
    def __init__(self, results_dir: Path = None):
        self.results_manager = ResultsManager(results_dir or Path("../results"))
        self.clustering_engine = ClusteringEngine(random_state=42)
        
        # Predefined archetype patterns based on memecoin trading knowledge
        self.archetype_templates = {
            'moon_mission': {
                'name': 'Moon Mission',
                'description': 'Sustained upward momentum with strong positive autocorrelation',
                'trading_strategy': 'Momentum following with trailing stops',
                'risk_profile': 'High risk, high reward'
            },
            'rug_pull': {
                'name': 'Rug Pull',
                'description': 'Quick initial pump followed by catastrophic dump',
                'trading_strategy': 'Immediate exit after pump detection',
                'risk_profile': 'Extremely high risk'
            },
            'slow_bleed': {
                'name': 'Slow Bleed',
                'description': 'Gradual sustained decline with negative momentum',
                'trading_strategy': 'Short selling or complete avoidance',
                'risk_profile': 'High risk, low reward'
            },
            'volatile_chop': {
                'name': 'Volatile Chop',
                'description': 'High volatility with no clear directional bias',
                'trading_strategy': 'Range trading or volatility arbitrage',
                'risk_profile': 'Moderate risk, moderate reward'
            },
            'dead_on_arrival': {
                'name': 'Dead on Arrival',
                'description': 'Minimal activity from launch with immediate death',
                'trading_strategy': 'Complete avoidance',
                'risk_profile': 'Low risk, no reward'
            },
            'phoenix_attempt': {
                'name': 'Phoenix Attempt',
                'description': 'Multiple revival attempts with intermittent pumps',
                'trading_strategy': 'Opportunistic swing trading',
                'risk_profile': 'High risk, variable reward'
            },
            'stable_survivor': {
                'name': 'Stable Survivor',
                'description': 'Low volatility with consistent survival patterns',
                'trading_strategy': 'Long-term accumulation',
                'risk_profile': 'Low risk, stable reward'
            },
            'zombie_walker': {
                'name': 'Zombie Walker',
                'description': 'Minimal movement with eventual death',
                'trading_strategy': 'Avoid or short with caution',
                'risk_profile': 'Low risk, negative reward'
            }
        }
        
    def load_stability_results(self, stability_results_path: Path) -> Dict:
        """Load stability testing results from Day 7-8."""
        if not stability_results_path.exists():
            raise ValueError(f"Stability results not found: {stability_results_path}")
        
        with open(stability_results_path, 'r') as f:
            stability_results = json.load(f)
        
        return stability_results
    
    def extract_archetype_data(self, stability_results: Dict) -> Dict[str, Dict]:
        """Extract archetype data from stability results for each category."""
        archetype_data = {}
        
        for category, result in stability_results['category_stability_results'].items():
            if not result.get('stability_testable') or not result.get('ceo_evaluation', {}).get('combined', {}).get('overall_pass', False):
                continue
                
            # Extract key information
            optimal_k = result['optimal_k']
            consensus_labels = result['bootstrap_stability']['consensus_labels']
            stability_confidence = result['bootstrap_stability']['stability_confidence']
            token_names = result.get('token_names', [])
            
            # Validate data consistency
            if not token_names:
                print(f"    ⚠️ Warning: No token_names found for {category}, skipping...")
                continue
                
            if len(token_names) != len(consensus_labels):
                print(f"    ❌ Error: Mismatch in {category} - token_names: {len(token_names)}, labels: {len(consensus_labels)}")
                print(f"    ⚠️ This might be due to missing token_names in stability results")
                continue
                
            if len(token_names) != len(stability_confidence):
                print(f"    ❌ Error: Mismatch in {category} - token_names: {len(token_names)}, confidence: {len(stability_confidence)}")
                continue
            
            # Create archetype clusters
            category_archetypes = {}
            for cluster_id in range(optimal_k):
                cluster_indices = [i for i, label in enumerate(consensus_labels) if label == cluster_id]
                
                if cluster_indices:
                    # Validate indices are within bounds
                    max_idx = max(cluster_indices)
                    if max_idx >= len(token_names):
                        print(f"    ❌ Error: Invalid index {max_idx} for token_names of length {len(token_names)}")
                        continue
                        
                    cluster_tokens = [token_names[i] for i in cluster_indices]
                    cluster_confidence = [stability_confidence[i] for i in cluster_indices]
                    
                    category_archetypes[f"{category}_cluster_{cluster_id}"] = {
                        'cluster_id': cluster_id,
                        'category': category,
                        'tokens': cluster_tokens,
                        'confidence_scores': cluster_confidence,
                        'size': len(cluster_indices),
                        'avg_confidence': np.mean(cluster_confidence),
                        'high_confidence_tokens': [
                            token for token, conf in zip(cluster_tokens, cluster_confidence) 
                            if conf >= 0.7
                        ]
                    }
            
            archetype_data[category] = category_archetypes
        
        return archetype_data
    
    def analyze_archetype_features(self, archetype_data: Dict[str, Dict], 
                                 stability_results: Dict) -> Dict[str, Dict]:
        """Analyze feature patterns for each archetype."""
        archetype_features = {}
        
        for category, category_archetypes in archetype_data.items():
            # Get features for this category from stability results
            category_result = stability_results['category_stability_results'][category]
            features = np.array(category_result.get('features', []))
            token_names = category_result.get('token_names', [])
            
            if len(features) == 0:
                continue
                
            # Create token name to index mapping
            token_to_idx = {name: i for i, name in enumerate(token_names)}
            
            for archetype_name, archetype_info in category_archetypes.items():
                # Get feature indices for this archetype
                archetype_token_indices = [
                    token_to_idx[token] for token in archetype_info['tokens']
                    if token in token_to_idx
                ]
                
                if not archetype_token_indices:
                    continue
                    
                archetype_features_matrix = features[archetype_token_indices]
                
                # Calculate feature statistics
                feature_stats = {
                    'mean': np.mean(archetype_features_matrix, axis=0),
                    'std': np.std(archetype_features_matrix, axis=0),
                    'min': np.min(archetype_features_matrix, axis=0),
                    'max': np.max(archetype_features_matrix, axis=0),
                    'median': np.median(archetype_features_matrix, axis=0)
                }
                
                # Convert to feature name mapping (using ESSENTIAL_FEATURES)
                feature_names = ESSENTIAL_FEATURES
                named_stats = {}
                for stat_name, stat_values in feature_stats.items():
                    named_stats[stat_name] = {
                        feature_names[i]: float(stat_values[i]) 
                        for i in range(min(len(feature_names), len(stat_values)))
                    }
                
                archetype_features[archetype_name] = {
                    'feature_statistics': named_stats,
                    'n_tokens': len(archetype_token_indices),
                    'category': category,
                    'cluster_id': archetype_info['cluster_id']
                }
        
        return archetype_features
    
    def classify_archetype_behavior(self, archetype_name: str, feature_stats: Dict) -> Dict[str, str]:
        """Classify archetype behavior based on feature patterns."""
        mean_features = feature_stats['mean']
        
        # Extract key behavioral indicators
        is_dead = mean_features.get('is_dead', 0) > 0.5
        mean_return = mean_features.get('mean_return', 0)
        volatility = mean_features.get('std_return', 0)
        max_drawdown = mean_features.get('max_drawdown', 0)
        acf_lag_1 = mean_features.get('acf_lag_1', 0)
        lifespan = mean_features.get('lifespan_minutes', 0)
        early_return = mean_features.get('return_5min', 0)
        
        # Classification logic
        if is_dead and lifespan < 100:
            behavior_type = 'dead_on_arrival'
        elif is_dead and early_return > 0.5 and max_drawdown > 0.8:
            behavior_type = 'rug_pull'
        elif mean_return > 0.2 and acf_lag_1 > 0.3:
            behavior_type = 'moon_mission'
        elif mean_return < -0.1 and acf_lag_1 < -0.1:
            behavior_type = 'slow_bleed'
        elif volatility > 0.5 and abs(mean_return) < 0.1:
            behavior_type = 'volatile_chop'
        elif not is_dead and volatility < 0.2:
            behavior_type = 'stable_survivor'
        elif is_dead and early_return > 0.2 and max_drawdown > 0.3:
            behavior_type = 'phoenix_attempt'
        else:
            behavior_type = 'zombie_walker'
        
        # Get template information
        template = self.archetype_templates.get(behavior_type, {
            'name': 'Unknown Pattern',
            'description': 'Unclassified behavioral pattern',
            'trading_strategy': 'Requires further analysis',
            'risk_profile': 'Unknown risk'
        })
        
        return {
            'behavior_type': behavior_type,
            'archetype_name': template['name'],
            'description': template['description'],
            'trading_strategy': template['trading_strategy'],
            'risk_profile': template['risk_profile'],
            'classification_confidence': self._calculate_classification_confidence(
                behavior_type, mean_features
            )
        }
    
    def _calculate_classification_confidence(self, behavior_type: str, mean_features: Dict) -> float:
        """Calculate confidence score for archetype classification."""
        # Simple heuristic based on feature strength
        confidence_factors = []
        
        if behavior_type == 'dead_on_arrival':
            confidence_factors.append(mean_features.get('is_dead', 0))
            confidence_factors.append(1.0 - min(mean_features.get('lifespan_minutes', 0) / 100, 1.0))
        elif behavior_type == 'rug_pull':
            confidence_factors.append(mean_features.get('is_dead', 0))
            confidence_factors.append(min(mean_features.get('return_5min', 0), 1.0))
            confidence_factors.append(min(mean_features.get('max_drawdown', 0), 1.0))
        elif behavior_type == 'moon_mission':
            confidence_factors.append(min(mean_features.get('mean_return', 0), 1.0))
            confidence_factors.append(min(mean_features.get('acf_lag_1', 0), 1.0))
        # Add more classification confidence logic for other types
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def generate_archetype_markdown(self, archetype_name: str, archetype_data: Dict,
                                  feature_analysis: Dict, behavior_classification: Dict) -> str:
        """Generate comprehensive markdown documentation for an archetype."""
        
        markdown = f"""# {behavior_classification['archetype_name']} Archetype

## Overview
- **Category**: {archetype_data['category'].title()}
- **Cluster ID**: {archetype_data['cluster_id']}
- **Behavior Type**: {behavior_classification['behavior_type']}
- **Classification Confidence**: {behavior_classification['classification_confidence']:.2%}

## Description
{behavior_classification['description']}

## Token Statistics
- **Total Tokens**: {archetype_data['size']:,}
- **High Confidence Tokens**: {len(archetype_data['high_confidence_tokens']):,}
- **Average Stability Confidence**: {archetype_data['avg_confidence']:.2%}

## Key Features

### Death Characteristics
- **Is Dead**: {feature_analysis['feature_statistics']['mean'].get('is_dead', 0):.2f}
- **Death Minute**: {feature_analysis['feature_statistics']['mean'].get('death_minute', 0):.0f}
- **Lifespan**: {feature_analysis['feature_statistics']['mean'].get('lifespan_minutes', 0):.0f} minutes

### Returns & Volatility
- **Mean Return**: {feature_analysis['feature_statistics']['mean'].get('mean_return', 0):.3f}
- **Std Return**: {feature_analysis['feature_statistics']['mean'].get('std_return', 0):.3f}
- **Max Drawdown**: {feature_analysis['feature_statistics']['mean'].get('max_drawdown', 0):.3f}
- **5-Min Volatility**: {feature_analysis['feature_statistics']['mean'].get('volatility_5min', 0):.3f}

### Autocorrelation Signature
- **ACF Lag 1**: {feature_analysis['feature_statistics']['mean'].get('acf_lag_1', 0):.3f}
- **ACF Lag 5**: {feature_analysis['feature_statistics']['mean'].get('acf_lag_5', 0):.3f}
- **ACF Lag 10**: {feature_analysis['feature_statistics']['mean'].get('acf_lag_10', 0):.3f}

### Early Detection (First 5 Minutes)
- **5-Min Return**: {feature_analysis['feature_statistics']['mean'].get('return_5min', 0):.3f}
- **Max 1-Min Return**: {feature_analysis['feature_statistics']['mean'].get('max_return_1min', 0):.3f}
- **5-Min Trend**: {feature_analysis['feature_statistics']['mean'].get('trend_direction_5min', 0):.3f}
- **Price Change Ratio**: {feature_analysis['feature_statistics']['mean'].get('price_change_ratio_5min', 0):.3f}
- **Pump Velocity**: {feature_analysis['feature_statistics']['mean'].get('pump_velocity_5min', 0):.3f}

## Trading Strategy
**Approach**: {behavior_classification['trading_strategy']}

**Risk Profile**: {behavior_classification['risk_profile']}

### Recommended Actions
"""
        
        # Add specific trading recommendations based on behavior type
        if behavior_classification['behavior_type'] == 'moon_mission':
            markdown += """
- **Entry**: Look for sustained positive momentum (ACF lag 1 > 0.3)
- **Exit**: Trailing stop at 20% below peak
- **Risk Management**: Position size 2-5% of portfolio
"""
        elif behavior_classification['behavior_type'] == 'rug_pull':
            markdown += """
- **Entry**: AVOID - or only scalp with tight stops
- **Exit**: Immediate exit if caught in position
- **Risk Management**: Maximum 0.5% position size if trading
"""
        elif behavior_classification['behavior_type'] == 'dead_on_arrival':
            markdown += """
- **Entry**: AVOID completely
- **Exit**: N/A
- **Risk Management**: Zero exposure
"""
        else:
            markdown += """
- **Entry**: Case-by-case analysis required
- **Exit**: Based on technical indicators
- **Risk Management**: Conservative position sizing
"""
        
        markdown += f"""

## Example Tokens
### High Confidence Examples
{self._format_token_list(archetype_data['high_confidence_tokens'][:10])}

### All Tokens in Archetype
- **Total**: {len(archetype_data['tokens'])} tokens
- **Confidence Range**: {min(archetype_data['confidence_scores']):.2%} - {max(archetype_data['confidence_scores']):.2%}

## Statistical Summary
"""
        
        # Add feature distribution summary
        for feature_name in ['mean_return', 'std_return', 'max_drawdown', 'lifespan_minutes']:
            stats = feature_analysis['feature_statistics']
            if feature_name in stats['mean']:
                markdown += f"- **{feature_name}**: {stats['mean'][feature_name]:.3f} ± {stats['std'][feature_name]:.3f} "
                markdown += f"(range: {stats['min'][feature_name]:.3f} to {stats['max'][feature_name]:.3f})\\n"
        
        markdown += f"""

## Validation Notes
- Generated from {feature_analysis['n_tokens']} tokens
- Based on Phase 1 Day 7-8 stability testing results
- Classification confidence: {behavior_classification['classification_confidence']:.2%}
- Requires validation against actual trading performance

---
*Generated by Phase 1 Day 9-10 Archetype Characterization*
*Timestamp: {datetime.now().isoformat()}*
"""
        
        return markdown
    
    def _format_token_list(self, tokens: List[str]) -> str:
        """Format token list for markdown display."""
        if not tokens:
            return "- None available"
        
        formatted = []
        for i, token in enumerate(tokens):
            formatted.append(f"- `{token}`")
            if i >= 9:  # Limit to first 10
                break
        
        return "\n".join(formatted)
    
    def generate_archetype_summary(self, all_archetypes: Dict) -> str:
        """Generate summary markdown for all discovered archetypes."""
        
        markdown = f"""# Memecoin Behavioral Archetypes Summary

## Overview
This document summarizes the behavioral archetypes discovered through Phase 1 analysis of memecoin time series data.

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Total Archetypes**: {len(all_archetypes)}

## Archetype Distribution
"""
        
        # Count archetypes by behavior type
        behavior_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for archetype_name, archetype_info in all_archetypes.items():
            behavior_counts[archetype_info['behavior_classification']['behavior_type']] += 1
            category_counts[archetype_info['archetype_data']['category']] += 1
        
        markdown += "\n### By Behavior Type\n"
        for behavior_type, count in sorted(behavior_counts.items()):
            template = self.archetype_templates.get(behavior_type, {})
            name = template.get('name', behavior_type.replace('_', ' ').title())
            markdown += f"- **{name}**: {count} archetype(s)\n"
        
        markdown += "\n### By Lifespan Category\n"
        for category, count in sorted(category_counts.items()):
            markdown += f"- **{category.title()}**: {count} archetype(s)\n"
        
        markdown += "\n## Quick Reference\n\n"
        markdown += "| Archetype | Category | Behavior Type | Tokens | Confidence |\n"
        markdown += "|-----------|----------|---------------|--------|------------|\n"
        
        for archetype_name, archetype_info in all_archetypes.items():
            behavior_name = archetype_info['behavior_classification']['archetype_name']
            category = archetype_info['archetype_data']['category']
            behavior_type = archetype_info['behavior_classification']['behavior_type']
            token_count = archetype_info['archetype_data']['size']
            confidence = archetype_info['behavior_classification']['classification_confidence']
            
            markdown += f"| {behavior_name} | {category} | {behavior_type} | {token_count} | {confidence:.1%} |\n"
        
        markdown += f"""

## Key Insights
"""
        if not behavior_counts:
            markdown += """- **Most Common Pattern**: No patterns found (data validation failed)
- **Rarest Pattern**: No patterns found (data validation failed)
- **Total Tokens Analyzed**: 0

⚠️ **Note**: No archetypes were successfully characterized. This typically happens when:
1. The stability testing results are missing token_names (re-run from phase 7-8)
2. Data validation checks failed due to mismatched array lengths
3. No categories passed the CEO stability requirements
"""
        else:
            most_common = max(behavior_counts.items(), key=lambda x: x[1])[0].replace('_', ' ').title()
            rarest = min(behavior_counts.items(), key=lambda x: x[1])[0].replace('_', ' ').title()
            total_tokens = sum(info['archetype_data']['size'] for info in all_archetypes.values())
            
            markdown += f"""- **Most Common Pattern**: {most_common}
- **Rarest Pattern**: {rarest}
- **Total Tokens Analyzed**: {total_tokens:,}"""
        
        markdown += """

## Trading Strategy Summary
### High Priority (Profitable)
- **Moon Mission**: Momentum following with trailing stops
- **Stable Survivor**: Long-term accumulation strategy

### Moderate Priority (Situational)
- **Phoenix Attempt**: Opportunistic swing trading
- **Volatile Chop**: Range trading or volatility arbitrage

### Low Priority (Risky)
- **Slow Bleed**: Short selling opportunities
- **Zombie Walker**: Avoid or careful shorting

### Avoid Completely
- **Dead on Arrival**: Zero trading value
- **Rug Pull**: Extremely dangerous for retail traders

## Data Quality Assessment
- All archetypes passed CEO stability requirements (ARI >= 0.75, Silhouette >= 0.5)
- High confidence token assignments available for validation
- Cross-validated against multiple clustering methods

## Next Steps
1. Validate archetypes against actual trading performance
2. Develop real-time classification system
3. Create archetype-specific feature engineering
4. Implement early warning system for rug pulls

---
*Generated by Phase 1 Day 9-10 Archetype Characterization*
*Timestamp: {datetime.now().isoformat()}*
"""
        
        return markdown
    
    def run_archetype_characterization(self, stability_results_path: Path) -> Dict:
        """
        Run complete archetype characterization analysis.
        
        Args:
            stability_results_path: Path to Day 7-8 stability results
            
        Returns:
            Complete archetype characterization results
        """
        print(f"🚀 Starting Phase 1 Day 9-10 Archetype Characterization")
        
        # Step 1: Load stability results
        print(f"\n📁 Loading stability results from {stability_results_path}")
        stability_results = self.load_stability_results(stability_results_path)
        
        # Step 2: Extract archetype data
        print(f"\n🔍 Extracting archetype data from stability results...")
        archetype_data = self.extract_archetype_data(stability_results)
        
        total_archetypes = sum(len(cat_archetypes) for cat_archetypes in archetype_data.values())
        print(f"✅ Found {total_archetypes} archetypes across {len(archetype_data)} categories")
        
        # Step 3: Analyze archetype features
        print(f"\n📊 Analyzing archetype feature patterns...")
        archetype_features = self.analyze_archetype_features(archetype_data, stability_results)
        
        # Step 4: Classify archetype behaviors
        print(f"\n🎭 Classifying archetype behaviors...")
        all_archetypes = {}
        archetype_markdowns = {}
        
        for category, category_archetypes in archetype_data.items():
            for archetype_name, archetype_info in category_archetypes.items():
                if archetype_name in archetype_features:
                    # Classify behavior
                    behavior_classification = self.classify_archetype_behavior(
                        archetype_name, archetype_features[archetype_name]['feature_statistics']
                    )
                    
                    # Generate markdown documentation
                    markdown_doc = self.generate_archetype_markdown(
                        archetype_name, archetype_info, 
                        archetype_features[archetype_name], behavior_classification
                    )
                    
                    all_archetypes[archetype_name] = {
                        'archetype_data': archetype_info,
                        'feature_analysis': archetype_features[archetype_name],
                        'behavior_classification': behavior_classification,
                        'markdown_documentation': markdown_doc
                    }
                    
                    archetype_markdowns[archetype_name] = markdown_doc
                else:
                    print(f"    ⚠️ Skipping {archetype_name}: not found in feature analysis results")
        
        # Check if any archetypes were created
        if not all_archetypes:
            print(f"\n❌ ERROR: No archetypes were successfully characterized!")
            print(f"   This typically happens when:")
            print(f"   1. The stability testing results are missing 'token_names' field")
            print(f"   2. Data validation checks failed due to array length mismatches")
            print(f"   3. No categories passed the CEO stability requirements")
            print(f"\n💡 SOLUTION: Re-run the pipeline from phase 7-8 (stability testing):")
            print(f"   python run_full_phase1.py --resume --from-phase day7_8 --data-dir ../../data/processed --n-tokens {len(archetype_data)}")
        
        # Step 5: Generate summary documentation
        print(f"\n📋 Generating archetype summary documentation...")
        summary_markdown = self.generate_archetype_summary(all_archetypes)
        
        # Step 6: Create archetype definitions for Phase 2
        archetype_definitions = self._create_archetype_definitions(all_archetypes)
        
        # Compile complete results
        complete_results = {
            'analysis_type': 'archetype_characterization',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'stability_results_reference': stability_results_path.name,
                'total_archetypes_characterized': len(all_archetypes),
                'categories_analyzed': list(archetype_data.keys())
            },
            'stability_reference': stability_results['timestamp'],
            'archetype_data': archetype_data,
            'archetype_features': archetype_features,
            'all_archetypes': all_archetypes,
            'archetype_markdowns': archetype_markdowns,
            'summary_markdown': summary_markdown,
            'archetype_definitions': archetype_definitions,
            'behavioral_distribution': self._analyze_behavioral_distribution(all_archetypes),
            'trading_strategy_summary': self._create_trading_strategy_summary(all_archetypes)
        }
        
        print(f"\n🎉 Archetype characterization complete!")
        print(f"📊 Characterized {len(all_archetypes)} archetypes")
        
        # Print archetype summary
        behavior_counts = defaultdict(int)
        for archetype_info in all_archetypes.values():
            behavior_counts[archetype_info['behavior_classification']['behavior_type']] += 1
        
        print(f"\n📈 Behavioral distribution:")
        for behavior_type, count in sorted(behavior_counts.items()):
            template = self.archetype_templates.get(behavior_type, {})
            name = template.get('name', behavior_type.replace('_', ' ').title())
            print(f"  {name}: {count} archetype(s)")
        
        return complete_results
    
    def _create_archetype_definitions(self, all_archetypes: Dict) -> Dict:
        """Create structured archetype definitions for Phase 2 usage."""
        definitions = {}
        
        for archetype_name, archetype_info in all_archetypes.items():
            behavior_class = archetype_info['behavior_classification']
            feature_stats = archetype_info['feature_analysis']['feature_statistics']['mean']
            
            definitions[archetype_name] = {
                'id': archetype_name,
                'name': behavior_class['archetype_name'],
                'behavior_type': behavior_class['behavior_type'],
                'category': archetype_info['archetype_data']['category'],
                'cluster_id': archetype_info['archetype_data']['cluster_id'],
                'size': archetype_info['archetype_data']['size'],
                'confidence': behavior_class['classification_confidence'],
                'key_features': {
                    'is_dead': feature_stats.get('is_dead', 0),
                    'mean_return': feature_stats.get('mean_return', 0),
                    'volatility': feature_stats.get('std_return', 0),
                    'max_drawdown': feature_stats.get('max_drawdown', 0),
                    'acf_lag_1': feature_stats.get('acf_lag_1', 0),
                    'lifespan_minutes': feature_stats.get('lifespan_minutes', 0)
                },
                'trading_strategy': behavior_class['trading_strategy'],
                'risk_profile': behavior_class['risk_profile']
            }
        
        return definitions
    
    def _analyze_behavioral_distribution(self, all_archetypes: Dict) -> Dict:
        """Analyze distribution of behavioral patterns."""
        distribution = defaultdict(list)
        
        for archetype_name, archetype_info in all_archetypes.items():
            behavior_type = archetype_info['behavior_classification']['behavior_type']
            category = archetype_info['archetype_data']['category']
            size = archetype_info['archetype_data']['size']
            
            distribution[behavior_type].append({
                'archetype_name': archetype_name,
                'category': category,
                'size': size,
                'confidence': archetype_info['behavior_classification']['classification_confidence']
            })
        
        return dict(distribution)
    
    def _create_trading_strategy_summary(self, all_archetypes: Dict) -> Dict:
        """Create trading strategy summary across all archetypes."""
        strategies = {
            'high_priority': [],
            'moderate_priority': [],
            'low_priority': [],
            'avoid_completely': []
        }
        
        for archetype_name, archetype_info in all_archetypes.items():
            behavior_type = archetype_info['behavior_classification']['behavior_type']
            archetype_name_clean = archetype_info['behavior_classification']['archetype_name']
            strategy = archetype_info['behavior_classification']['trading_strategy']
            
            if behavior_type in ['moon_mission', 'stable_survivor']:
                strategies['high_priority'].append({
                    'name': archetype_name_clean,
                    'strategy': strategy,
                    'size': archetype_info['archetype_data']['size']
                })
            elif behavior_type in ['phoenix_attempt', 'volatile_chop']:
                strategies['moderate_priority'].append({
                    'name': archetype_name_clean,
                    'strategy': strategy,
                    'size': archetype_info['archetype_data']['size']
                })
            elif behavior_type in ['slow_bleed', 'zombie_walker']:
                strategies['low_priority'].append({
                    'name': archetype_name_clean,
                    'strategy': strategy,
                    'size': archetype_info['archetype_data']['size']
                })
            else:  # dead_on_arrival, rug_pull
                strategies['avoid_completely'].append({
                    'name': archetype_name_clean,
                    'strategy': strategy,
                    'size': archetype_info['archetype_data']['size']
                })
        
        return strategies
    
    def save_results(self, results: Dict, output_dir: Path = None) -> str:
        """Save complete analysis results."""
        if output_dir:
            self.results_manager = ResultsManager(output_dir)
        
        # Save main results
        timestamp = self.results_manager.save_analysis_results(
            results, 
            analysis_name="archetype_characterization", 
            phase_dir="phase1_day9_10_archetypes",
            include_plots=True
        )
        
        # Save individual archetype markdown files
        phase_dir = self.results_manager.base_results_dir / "phase1_day9_10_archetypes"
        archetype_dir = phase_dir / "archetype_documentation"
        archetype_dir.mkdir(exist_ok=True)
        
        for archetype_name, markdown_content in results['archetype_markdowns'].items():
            safe_name = archetype_name.replace('/', '_').replace('\\', '_')
            markdown_path = archetype_dir / f"{safe_name}.md"
            with open(markdown_path, 'w') as f:
                f.write(markdown_content)
        
        # Save summary markdown
        summary_path = archetype_dir / "archetype_summary.md"
        with open(summary_path, 'w') as f:
            f.write(results['summary_markdown'])
        
        print(f"✅ Archetype documentation saved to: {archetype_dir}")
        
        return timestamp


def create_gradio_interface():
    """Create interactive Gradio interface for archetype characterization."""
    visualizer = GradioVisualizer("Phase 1 Day 9-10: Archetype Characterization")
    analyzer = ArchetypeCharacterizationAnalyzer()
    
    def run_interactive_analysis(stability_results_str: str):
        """Run analysis with Gradio inputs."""
        try:
            if not stability_results_str.strip():
                return "❌ Day 7-8 stability results file is required", None, None, None
            
            stability_results_path = Path(stability_results_str)
            if not stability_results_path.exists():
                return f"❌ Stability results file not found: {stability_results_str}", None, None, None
            
            # Run analysis
            results = analyzer.run_archetype_characterization(stability_results_path)
            
            # Save results
            try:
                timestamp = analyzer.save_results(results)
                print(f"✅ Results successfully saved with timestamp: {timestamp}")
            except Exception as save_error:
                print(f"❌ Error saving results: {save_error}")
                timestamp = "save_failed"
            
            # Generate summary
            total_archetypes = len(results['all_archetypes'])
            behavior_counts = results['behavioral_distribution']
            
            summary = f"""
            ## 🎭 Archetype Characterization Complete!
            
            **Total Archetypes**: {total_archetypes}
            **Categories Analyzed**: {len(results['archetype_data'])}
            
            **Behavioral Distribution**:
            """
            
            for behavior_type, archetypes in behavior_counts.items():
                template = analyzer.archetype_templates.get(behavior_type, {})
                name = template.get('name', behavior_type.replace('_', ' ').title())
                total_tokens = sum(arch['size'] for arch in archetypes)
                summary += f"\n- **{name}**: {len(archetypes)} archetype(s) ({total_tokens:,} tokens)"
            
            summary += f"\n\n**Trading Strategy Summary**:"
            strategy_summary = results['trading_strategy_summary']
            summary += f"\n- **High Priority**: {len(strategy_summary['high_priority'])} archetypes"
            summary += f"\n- **Moderate Priority**: {len(strategy_summary['moderate_priority'])} archetypes"
            summary += f"\n- **Low Priority**: {len(strategy_summary['low_priority'])} archetypes"
            summary += f"\n- **Avoid Completely**: {len(strategy_summary['avoid_completely'])} archetypes"
            
            summary += f"\n\n**Documentation Generated**:"
            summary += f"\n- Individual archetype documentation files"
            summary += f"\n- Comprehensive summary document"
            summary += f"\n- Structured definitions for Phase 2 usage"
            
            summary += f"\n\n**Results saved with timestamp**: {timestamp}"
            
            # Generate visualization (archetype distribution)
            plots = [None, None, None]
            
            if total_archetypes > 0:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Behavioral distribution pie chart
                behavior_names = []
                behavior_counts_list = []
                
                for behavior_type, archetypes in behavior_counts.items():
                    template = analyzer.archetype_templates.get(behavior_type, {})
                    name = template.get('name', behavior_type.replace('_', ' ').title())
                    behavior_names.append(name)
                    behavior_counts_list.append(len(archetypes))
                
                fig = go.Figure(data=[go.Pie(
                    labels=behavior_names,
                    values=behavior_counts_list,
                    hole=0.3
                )])
                
                fig.update_layout(
                    title='Behavioral Archetype Distribution',
                    height=500
                )
                
                plots[0] = fig
            
            return summary, *plots
            
        except Exception as e:
            error_msg = f"## ❌ Error During Analysis\n\n```\n{str(e)}\n```"
            return error_msg, None, None, None
    
    import gradio as gr
    
    interface = gr.Interface(
        fn=run_interactive_analysis,
        inputs=[
            gr.Textbox(
                value="",
                label="Day 7-8 Stability Results File (JSON) - REQUIRED",
                placeholder="Path to Day 7-8 stability results JSON file (REQUIRED)"
            )
        ],
        outputs=[
            gr.Markdown(label="Archetype Characterization Results"),
            gr.Plot(label="Behavioral Distribution"),
            gr.Plot(label="Trading Strategy Summary"),
            gr.Plot(label="Category Analysis")
        ],
        title="🎭 Phase 1 Day 9-10: Archetype Characterization",
        description="Generate comprehensive documentation for discovered behavioral archetypes"
    )
    
    return interface


def main():
    """Main entry point with CLI and interactive modes."""
    parser = argparse.ArgumentParser(description="Phase 1 Day 9-10 Archetype Characterization")
    parser.add_argument("--stability-results", type=Path,
                       help="Path to Day 7-8 stability results JSON file (REQUIRED for CLI mode)")
    parser.add_argument("--output-dir", type=Path, default=Path("../results"),
                       help="Output directory for results")
    parser.add_argument("--interactive", action="store_true",
                       help="Launch interactive Gradio interface")
    parser.add_argument("--share", action="store_true",
                       help="Create public Gradio link")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("🚀 Launching interactive Gradio interface...")
        interface = create_gradio_interface()
        interface.launch(share=args.share)
    else:
        # CLI mode - require stability results
        if not args.stability_results:
            print("❌ Error: --stability-results is required for CLI mode")
            print("Use --interactive for Gradio interface or provide --stability-results PATH")
            return
        
        analyzer = ArchetypeCharacterizationAnalyzer(args.output_dir)
        results = analyzer.run_archetype_characterization(args.stability_results)
        try:
            timestamp = analyzer.save_results(results)
            print(f"✅ Results successfully saved with timestamp: {timestamp}")
        except Exception as save_error:
            print(f"❌ Error saving results: {save_error}")
            timestamp = "save_failed"
        
        print(f"\n📁 Results saved to: {args.output_dir}/phase1_day9_10_archetypes/")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"🎭 Characterized {len(results['all_archetypes'])} archetypes")


if __name__ == "__main__":
    main()