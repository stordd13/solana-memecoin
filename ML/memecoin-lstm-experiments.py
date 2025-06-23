import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import polars as pl
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

class TokenAnalyzer:
    """Analyze tokens to detect overlaps and characteristics"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.token_info = {}
        
    def analyze_all_tokens(self):
        """Analyze all tokens across categories"""
        categories = {
            'normal_behavior': self.base_dir / 'normal_behavior_tokens',
            'with_gaps': self.base_dir / 'tokens_with_gaps',
            'with_extremes': self.base_dir / 'tokens_with_extremes',
            'dead': self.base_dir / 'dead_tokens'
        }
        
        for cat_name, cat_path in categories.items():
            if cat_path.exists():
                for file in cat_path.glob("*.parquet"):
                    token_name = file.stem
                    if token_name not in self.token_info:
                        self.token_info[token_name] = {
                            'categories': [],
                            'characteristics': {}
                        }
                    
                    self.token_info[token_name]['categories'].append(cat_name)
                    
                    # Analyze token characteristics
                    try:
                        df = pl.read_parquet(file)
                        prices = df['price'].to_numpy()
                        
                        # Check if dead
                        is_dead, dead_after = self._check_if_dead(prices)
                        
                        # Check for extremes
                        has_extremes, extreme_info = self._check_extremes(prices)
                        
                        self.token_info[token_name]['characteristics'][cat_name] = {
                            'is_dead': is_dead,
                            'dead_after_minutes': dead_after,
                            'has_extremes': has_extremes,
                            'max_pump': extreme_info.get('max_pump', 0),
                            'max_dump': extreme_info.get('max_dump', 0),
                            'total_minutes': len(prices)
                        }
                    except Exception as e:
                        print(f"Error analyzing {token_name}: {e}")
        
        return self._generate_report()
    
    def _check_if_dead(self, prices: np.ndarray, threshold: float = 0.0001) -> Tuple[bool, int]:
        """Check if token becomes dead (price stops changing)"""
        price_changes = np.abs(np.diff(prices))
        
        # Find consecutive periods of no change
        no_change = price_changes < threshold
        
        # Find the longest streak
        max_streak = 0
        current_streak = 0
        dead_start_idx = -1
        
        for i, nc in enumerate(no_change):
            if nc:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    dead_start_idx = i - current_streak + 1
            else:
                current_streak = 0
        
        # Consider dead if no movement for > 60 minutes
        is_dead = max_streak > 60
        dead_after = dead_start_idx if is_dead else -1
        
        return is_dead, dead_after
    
    def _check_extremes(self, prices: np.ndarray, threshold: float = 5.0) -> Tuple[bool, Dict]:
        """Check for extreme price movements"""
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = returns[~np.isnan(returns)]
        
        # Calculate rolling std
        window = min(60, len(returns) // 4)
        rolling_std = pd.Series(returns).rolling(window, center=True).std()
        
        # Detect outliers
        z_scores = np.abs((returns - np.mean(returns)) / np.std(returns))
        has_extremes = np.any(z_scores > threshold)
        
        extreme_info = {
            'max_pump': np.max(returns) if len(returns) > 0 else 0,
            'max_dump': np.min(returns) if len(returns) > 0 else 0,
            'extreme_count': np.sum(z_scores > threshold)
        }
        
        return has_extremes, extreme_info
    
    def _generate_report(self) -> Dict:
        """Generate analysis report"""
        report = {
            'total_tokens': len(self.token_info),
            'overlap_analysis': {},
            'category_stats': {},
            'problematic_tokens': []
        }
        
        # Find overlaps
        for token, info in self.token_info.items():
            if len(info['categories']) > 1:
                # Check if dead token is also in extremes
                cats = info['categories']
                if 'dead' in cats and 'with_extremes' in cats:
                    report['problematic_tokens'].append({
                        'token': token,
                        'issue': 'dead_and_extreme',
                        'categories': cats,
                        'dead_after': info['characteristics'].get('dead', {}).get('dead_after_minutes', -1)
                    })
        
        # Category statistics
        for cat in ['normal_behavior', 'with_gaps', 'with_extremes', 'dead']:
            tokens_in_cat = [t for t, i in self.token_info.items() if cat in i['categories']]
            report['category_stats'][cat] = {
                'count': len(tokens_in_cat),
                'exclusive_count': len([t for t in tokens_in_cat 
                                      if len(self.token_info[t]['categories']) == 1])
            }
        
        return report


class ExperimentRunner:
    """Run experiments with different LSTM configurations"""
    
    def __init__(self, base_model_class, base_dir: Path):
        self.base_model_class = base_model_class
        self.base_dir = base_dir
        self.results = {}
        
    def run_forecast_horizon_experiment(self, 
                                      horizons: List[int] = [5, 10, 15, 30, 60],
                                      lookback: int = 60):
        """Test different forecast horizons"""
        
        for horizon in horizons:
            print(f"\nTesting forecast horizon: {horizon} minutes")
            
            # Create model
            model = self.base_model_class(
                input_size=1,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                forecast_horizon=horizon
            )
            
            # Train and evaluate (simplified for demonstration)
            # In practice, use the full training pipeline
            metrics = self._train_and_evaluate(model, lookback, horizon)
            
            self.results[f'horizon_{horizon}'] = metrics
        
        return self._plot_horizon_results()
    
    def run_architecture_experiment(self):
        """Test different model architectures"""
        architectures = [
            {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2},
            {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3},
            {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.4},
            {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3},
        ]
        
        for i, config in enumerate(architectures):
            print(f"\nTesting architecture {i+1}: {config}")
            
            model = self.base_model_class(
                input_size=1,
                forecast_horizon=15,
                **config
            )
            
            metrics = self._train_and_evaluate(model, lookback=60, horizon=15)
            self.results[f'arch_{i+1}'] = {**metrics, **config}
        
        return self._compare_architectures()
    
    def _train_and_evaluate(self, model, lookback, horizon):
        """Simplified training and evaluation"""
        # This is a placeholder - use the full training pipeline from the main script
        return {
            'mse': np.random.random() * 0.1,
            'mae': np.random.random() * 0.05,
            'direction_accuracy': 0.5 + np.random.random() * 0.3,
            'best_val_loss': np.random.random() * 0.1
        }
    
    def _plot_horizon_results(self):
        """Plot results for different horizons"""
        horizons = []
        accuracies = []
        mses = []
        
        for key, metrics in self.results.items():
            if key.startswith('horizon_'):
                h = int(key.split('_')[1])
                horizons.append(h)
                accuracies.append(metrics['direction_accuracy'])
                mses.append(metrics['mse'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(horizons, accuracies, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Forecast Horizon (minutes)')
        ax1.set_ylabel('Direction Accuracy')
        ax1.set_title('Prediction Accuracy vs Forecast Horizon')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(horizons, mses, 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Forecast Horizon (minutes)')
        ax2.set_ylabel('MSE')
        ax2.set_title('Mean Squared Error vs Forecast Horizon')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _compare_architectures(self):
        """Compare different architectures"""
        arch_data = []
        
        for key, metrics in self.results.items():
            if key.startswith('arch_'):
                arch_data.append({
                    'Architecture': key,
                    'Hidden Size': metrics['hidden_size'],
                    'Layers': metrics['num_layers'],
                    'Direction Accuracy': metrics['direction_accuracy'],
                    'MSE': metrics['mse']
                })
        
        df = pd.DataFrame(arch_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped bar chart
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Direction Accuracy'], width, label='Accuracy')
        ax.bar(x + width/2, df['MSE'] * 10, width, label='MSE (x10)')
        
        ax.set_xlabel('Architecture')
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{row['Hidden Size']}_L{row['Layers']}" 
                           for _, row in df.iterrows()])
        ax.legend()
        ax.set_title('Model Architecture Comparison')
        
        return fig, df


class TradingSignalGenerator:
    """Generate trading signals from LSTM predictions"""
    
    def __init__(self, model, scaler, lookback: int = 60):
        self.model = model
        self.scaler = scaler
        self.lookback = lookback
        
    def generate_signals(self, prices: np.ndarray, 
                        confidence_threshold: float = 0.02) -> Dict:
        """Generate trading signals from price data"""
        
        # Normalize prices
        prices_norm = self.scaler.transform(prices.reshape(-1, 1)).flatten()
        
        # Create sequences
        sequences = []
        for i in range(len(prices_norm) - self.lookback):
            sequences.append(prices_norm[i:i + self.lookback])
        
        if not sequences:
            return {'signals': [], 'confidence': []}
        
        # Get predictions
        X = torch.FloatTensor(sequences)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X).numpy()
        
        # Inverse transform predictions
        pred_prices = self.scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        
        # Generate signals
        signals = []
        confidences = []
        
        for i, pred in enumerate(pred_prices):
            current_price = prices[i + self.lookback - 1]
            pred_return = (pred[-1] - current_price) / current_price
            
            # Signal generation
            if pred_return > confidence_threshold:
                signal = 'BUY'
                confidence = min(pred_return / confidence_threshold, 2.0)
            elif pred_return < -confidence_threshold:
                signal = 'SELL'
                confidence = min(abs(pred_return) / confidence_threshold, 2.0)
            else:
                signal = 'HOLD'
                confidence = 1.0 - abs(pred_return) / confidence_threshold
            
            signals.append({
                'time': i + self.lookback,
                'signal': signal,
                'confidence': confidence,
                'predicted_return': pred_return,
                'predicted_price': pred[-1],
                'current_price': current_price
            })
            confidences.append(confidence)
        
        return {
            'signals': signals,
            'avg_confidence': np.mean(confidences),
            'signal_distribution': pd.Series([s['signal'] for s in signals]).value_counts().to_dict()
        }


def analyze_token_quality(base_dir: Path) -> pd.DataFrame:
    """Analyze token quality across categories"""
    
    analyzer = TokenAnalyzer(base_dir)
    report = analyzer.analyze_all_tokens()
    
    print("\n=== Token Analysis Report ===")
    print(f"Total tokens analyzed: {report['total_tokens']}")
    print("\nCategory Statistics:")
    for cat, stats in report['category_stats'].items():
        print(f"  {cat}: {stats['count']} tokens ({stats['exclusive_count']} exclusive)")
    
    print(f"\nProblematic tokens (dead + extreme): {len(report['problematic_tokens'])}")
    
    # Create DataFrame of problematic tokens
    if report['problematic_tokens']:
        df_problematic = pd.DataFrame(report['problematic_tokens'])
        print("\nFirst 10 problematic tokens:")
        print(df_problematic.head(10))
        
        # Save for filtering
        df_problematic.to_csv('problematic_tokens.csv', index=False)
        print("\nSaved problematic tokens to 'problematic_tokens.csv'")
    
    return report


def create_filtered_dataset(base_dir: Path, 
                          exclude_dead_extremes: bool = True,
                          min_active_minutes: int = 60) -> List[Path]:
    """Create a filtered dataset excluding problematic tokens"""
    
    # First, analyze all tokens
    analyzer = TokenAnalyzer(base_dir)
    analyzer.analyze_all_tokens()
    
    filtered_paths = []
    excluded_count = 0
    
    # Categories to include (in priority order)
    categories = [
        'cleaned_normal_behavior_tokens',
        'cleaned_tokens_with_gaps',
        'cleaned_tokens_with_extremes'
    ]
    
    for category in categories:
        cat_path = base_dir / category
        if not cat_path.exists():
            continue
            
        for file in cat_path.glob("*.parquet"):
            token_name = file.stem
            token_info = analyzer.token_info.get(token_name, {})
            
            # Exclusion criteria
            exclude = False
            
            # Check if token appears in dead category
            if 'dead' in token_info.get('categories', []):
                if exclude_dead_extremes:
                    exclude = True
                    
            # Check if token dies too early
            for cat_chars in token_info.get('characteristics', {}).values():
                if cat_chars.get('is_dead', False):
                    dead_after = cat_chars.get('dead_after_minutes', -1)
                    if dead_after > 0 and dead_after < min_active_minutes:
                        exclude = True
                        break
            
            if not exclude:
                filtered_paths.append(file)
            else:
                excluded_count += 1
    
    print(f"\nFiltered dataset: {len(filtered_paths)} tokens retained, {excluded_count} excluded")
    
    return filtered_paths


def run_comprehensive_experiment():
    """Run a comprehensive experiment with proper data filtering"""
    
    # Configuration
    BASE_DIR = Path("data/processed")
    LOOKBACK = 60
    FORECAST_HORIZONS = [5, 10, 15, 30, 60]
    
    # Step 1: Analyze token quality
    print("Step 1: Analyzing token quality...")
    token_report = analyze_token_quality(BASE_DIR)
    
    # Step 2: Create filtered dataset
    print("\nStep 2: Creating filtered dataset...")
    filtered_paths = create_filtered_dataset(
        BASE_DIR, 
        exclude_dead_extremes=True,
        min_active_minutes=60
    )
    
    # Step 3: Run horizon experiments
    print("\nStep 3: Running forecast horizon experiments...")
    
    # Import the main LSTM class and dataset
    from memecoin_lstm_model import PricePredictionLSTM, MemecoinDataset
    
    experiment_runner = ExperimentRunner(PricePredictionLSTM, BASE_DIR)
    
    # For each horizon, train a model
    results = {}
    for horizon in FORECAST_HORIZONS:
        print(f"\n--- Training for {horizon}-minute forecast ---")
        
        # Split data
        train_size = int(0.7 * len(filtered_paths))
        val_size = int(0.15 * len(filtered_paths))
        
        train_paths = filtered_paths[:train_size]
        val_paths = filtered_paths[train_size:train_size + val_size]
        test_paths = filtered_paths[train_size + val_size:]
        
        # Create datasets
        train_dataset = MemecoinDataset(train_paths, LOOKBACK, horizon)
        val_dataset = MemecoinDataset(val_paths, LOOKBACK, horizon, 
                                     scaler=train_dataset.scaler)
        
        # Create model
        model = PricePredictionLSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            forecast_horizon=horizon
        )
        
        # Store results
        results[f'horizon_{horizon}'] = {
            'model': model,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'scaler': train_dataset.scaler
        }
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
    
    # Step 4: Generate trading signals for a sample token
    print("\nStep 4: Testing trading signal generation...")
    
    # Load a sample token
    sample_token_path = filtered_paths[0]
    df = pl.read_parquet(sample_token_path)
    prices = df['price'].to_numpy()
    
    # Use the 15-minute model for signals
    model_15 = results['horizon_15']['model']
    scaler_15 = results['horizon_15']['scaler']
    
    signal_generator = TradingSignalGenerator(model_15, scaler_15, LOOKBACK)
    signals = signal_generator.generate_signals(prices)
    
    print(f"\nSignal Analysis for {sample_token_path.stem}:")
    print(f"  Average confidence: {signals['avg_confidence']:.2f}")
    print(f"  Signal distribution: {signals['signal_distribution']}")
    
    # Plot example predictions
    if signals['signals']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price chart with signals
        ax1.plot(prices, label='Actual Price', alpha=0.7)
        
        buy_signals = [(s['time'], s['current_price']) for s in signals['signals'] 
                      if s['signal'] == 'BUY']
        sell_signals = [(s['time'], s['current_price']) for s in signals['signals'] 
                       if s['signal'] == 'SELL']
        
        if buy_signals:
            ax1.scatter(*zip(*buy_signals), color='green', marker='^', s=100, label='Buy')
        if sell_signals:
            ax1.scatter(*zip(*sell_signals), color='red', marker='v', s=100, label='Sell')
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Price and Trading Signals - {sample_token_path.stem}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence over time
        confidences = [s['confidence'] for s in signals['signals']]
        times = [s['time'] for s in signals['signals']]
        
        ax2.plot(times, confidences, label='Signal Confidence')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Trading Signal Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('trading_signals_example.png', dpi=150)
        print("\nSaved trading signals visualization to 'trading_signals_example.png'")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive experiment
    results = run_comprehensive_experiment()
    
    print("\n=== Experiment Complete ===")
    print("Next steps:")
    print("1. Review 'problematic_tokens.csv' for tokens to exclude")
    print("2. Check 'trading_signals_example.png' for signal quality")
    print("3. Run the main training script with filtered data")
    print("4. Test different confidence thresholds for signal generation")