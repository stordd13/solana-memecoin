import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class TokenAnalyzer:
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.results = {}
        
    def load_token_data(self, file_path: Path) -> pd.DataFrame:
        """Load token data from parquet file"""
        try:
            df = pd.read_parquet(file_path)
            # Ensure datetime column is properly formatted
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistics for a token"""
        if df is None or len(df) == 0:
            return None
            
        stats = {
            'price_mean': df['price'].mean(),
            'price_std': df['price'].std(),
            'price_min': df['price'].min(),
            'price_max': df['price'].max(),
            'price_range': df['price'].max() - df['price'].min(),
            'price_volatility': df['price'].std() / df['price'].mean() if df['price'].mean() != 0 else 0,
            'total_volume': df['volume'].sum() if 'volume' in df.columns else None,
            'time_span': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600,  # in hours
            'data_points': len(df)
        }
        return stats

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and rolling statistics"""
        if df is None or len(df) == 0:
            return None
            
        df = df.copy()
        df['returns'] = df['price'].pct_change()
        df['rolling_std'] = df['returns'].rolling(window=60).std()  # 1-hour rolling volatility
        df['rolling_mean'] = df['returns'].rolling(window=60).mean()
        return df

    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 3) -> Dict:
        """Detect price anomalies using z-score"""
        if df is None or len(df) == 0:
            return None
            
        returns = df['price'].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        anomalies = z_scores[z_scores > threshold]
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_times': anomalies.index.tolist(),
            'max_price_change': returns.max(),
            'min_price_change': returns.min()
        }

    def analyze_token(self, file_path: Path) -> Dict:
        """Perform comprehensive analysis on a single token"""
        df = self.load_token_data(file_path)
        if df is None:
            return None
            
        # Calculate basic statistics
        basic_stats = self.calculate_basic_stats(df)
        
        # Calculate returns and rolling statistics
        df_with_returns = self.calculate_returns(df)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(df)
        
        # Combine all results
        analysis = {
            'token_name': file_path.stem.split('_')[0],
            'basic_stats': basic_stats,
            'anomalies': anomalies,
            'data': df_with_returns
        }
        
        return analysis

    def analyze_multiple_tokens(self, limit: int = None) -> Dict:
        """Analyze multiple tokens from the dataset"""
        all_results = {}
        
        # Get all dataset folders
        dataset_folders = [f for f in self.data_path.iterdir() 
                         if f.is_dir() and f.name.startswith('dataset')]
        
        for dataset_folder in dataset_folders:
            # Get all parquet files in the dataset
            parquet_files = list(dataset_folder.rglob("*.parquet"))
            
            if limit:
                parquet_files = parquet_files[:limit]
            
            for file_path in parquet_files:
                print(f"Analyzing {file_path.name}...")
                analysis = self.analyze_token(file_path)
                if analysis:
                    all_results[file_path.name] = analysis
        
        return all_results

    def generate_summary_report(self, results: Dict) -> pd.DataFrame:
        """Generate a summary report of all analyzed tokens"""
        summary_data = []
        
        for token_name, analysis in results.items():
            if analysis and analysis['basic_stats']:
                stats = analysis['basic_stats']
                anomalies = analysis['anomalies']
                
                summary_data.append({
                    'token': token_name.split('_')[0],
                    'mean_price': stats['price_mean'],
                    'price_volatility': stats['price_volatility'],
                    'total_volume': stats['total_volume'],
                    'time_span_hours': stats['time_span'],
                    'data_points': stats['data_points'],
                    'anomaly_count': anomalies['anomaly_count'] if anomalies else 0,
                    'max_price_change': anomalies['max_price_change'] if anomalies else None,
                    'min_price_change': anomalies['min_price_change'] if anomalies else None
                })
        
        return pd.DataFrame(summary_data)

    def plot_token_analysis(self, results: Dict, output_dir: str = "analysis_results"):
        """Generate plots for token analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for token_name, analysis in results.items():
            if analysis and analysis['data'] is not None:
                df = analysis['data']
                
                # Create figure with subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # Plot price
                ax1.plot(df['datetime'], df['price'])
                ax1.set_title(f'Price Analysis - {token_name.split("_")[0]}')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Price')
                
                # Plot returns and volatility
                ax2.plot(df['datetime'], df['returns'], label='Returns')
                ax2.plot(df['datetime'], df['rolling_std'], label='1-hour Rolling Volatility')
                ax2.set_title('Returns and Volatility')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Value')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(output_path / f"{token_name.split('_')[0]}_analysis.png")
                plt.close()

def main():
    # Initialize analyzer
    analyzer = TokenAnalyzer()
    
    # Analyze tokens (limit to 10 for initial analysis)
    results = analyzer.analyze_multiple_tokens(limit=10)
    
    # Generate summary report
    summary_df = analyzer.generate_summary_report(results)
    print("\nSummary Report:")
    print(summary_df)
    
    # Save summary to CSV
    summary_df.to_csv("analysis_results/token_summary.csv", index=False)
    
    # Generate plots
    analyzer.plot_token_analysis(results)
    
    print("\nAnalysis complete! Results saved in 'analysis_results' directory.")

if __name__ == "__main__":
    main() 