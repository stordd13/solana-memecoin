import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class TokenAnalyzer:
    def __init__(self, data_subfolder: str = "raw"):
        # Auto-detect project root
        current_dir = Path(__file__).parent  # data_analysis/
        project_root = current_dir.parent    # memecoin2/
        self.data_root = project_root / "data"
        self.data_path = self.data_root / data_subfolder
        self.results = {}
        
    def load_token_data(self, file_path: Path) -> pl.DataFrame:
        """Load token data from parquet file using Polars"""
        try:
            df = pl.read_parquet(file_path)
            # Ensure datetime column is properly formatted
            if 'datetime' in df.columns:
                # Only parse if it's not already a datetime type
                if df['datetime'].dtype != pl.Datetime:
                    df = df.with_columns(pl.col('datetime').str.strptime(pl.Datetime))
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_basic_stats(self, df: pl.DataFrame) -> Dict:
        """Calculate basic statistics for a token using Polars"""
        if df is None or len(df) == 0:
            return None
            
        # Calculate price statistics using Polars
        price_stats = df.select([
            pl.col('price').mean().alias('price_mean'),
            pl.col('price').std().alias('price_std'),
            pl.col('price').min().alias('price_min'),
            pl.col('price').max().alias('price_max'),
        ]).row(0)
        
        price_mean, price_std, price_min, price_max = price_stats
        
        # Calculate time span using Polars
        if 'datetime' in df.columns:
            time_span_seconds = (df['datetime'].max() - df['datetime'].min()).total_seconds()
            time_span_hours = time_span_seconds / 3600
        else:
            time_span_hours = 0
        
        # Volume sum if available
        total_volume = df.select(pl.col('volume').sum()).item() if 'volume' in df.columns else None
        
        stats = {
            'price_mean': price_mean,
            'price_std': price_std,
            'price_min': price_min,
            'price_max': price_max,
            'price_range': price_max - price_min,
            'price_volatility': price_std / price_mean if price_mean and price_mean != 0 else 0,
            'total_volume': total_volume,
            'time_span': time_span_hours,
            'data_points': len(df)
        }
        return stats

    def calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate returns and rolling statistics using Polars"""
        if df is None or len(df) == 0:
            return None
            
        # Calculate returns using Polars
        df_with_returns = df.with_columns([
            pl.col('price').pct_change().alias('returns'),
        ]).with_columns([
            # 1-hour rolling volatility and mean
            pl.col('returns').rolling_std(window_size=60).alias('rolling_std'),
            pl.col('returns').rolling_mean(window_size=60).alias('rolling_mean'),
        ])
        
        return df_with_returns

    def detect_anomalies(self, df: pl.DataFrame, threshold: float = 3) -> Dict:
        """Detect price anomalies using z-score with Polars"""
        if df is None or len(df) == 0:
            return None
            
        # Calculate returns and z-scores using Polars
        returns_analysis = df.with_columns([
            pl.col('price').pct_change().alias('returns')
        ]).with_columns([
            # Calculate z-scores
            ((pl.col('returns') - pl.col('returns').mean()) / pl.col('returns').std()).abs().alias('z_scores')
        ])
        
        # Find anomalies
        anomalies = returns_analysis.filter(pl.col('z_scores') > threshold)
        
        # Get return statistics
        returns_stats = returns_analysis.select([
            pl.col('returns').max().alias('max_return'),
            pl.col('returns').min().alias('min_return'),
        ]).row(0)
        
        max_return, min_return = returns_stats
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_times': anomalies['datetime'].to_list() if 'datetime' in anomalies.columns else [],
            'max_price_change': max_return,
            'min_price_change': min_return
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

    def generate_summary_report(self, results: Dict) -> pl.DataFrame:
        """Generate a summary report of all analyzed tokens using Polars"""
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
        
        return pl.DataFrame(summary_data)

    def plot_token_analysis(self, results: Dict, output_dir: str = "analysis_results"):
        """Generate plots for token analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for token_name, analysis in results.items():
            if analysis and analysis['data'] is not None:
                df = analysis['data']
                
                # Convert Polars DataFrame to pandas for plotting (matplotlib needs pandas/numpy)
                df_plot = df.to_pandas()
                
                # Create figure with subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # Plot price
                ax1.plot(df_plot['datetime'], df_plot['price'])
                ax1.set_title(f'Price Analysis - {token_name.split("_")[0]}')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Price')
                
                # Plot returns and volatility
                ax2.plot(df_plot['datetime'], df_plot['returns'], label='Returns')
                ax2.plot(df_plot['datetime'], df_plot['rolling_std'], label='1-hour Rolling Volatility')
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
    
    # Save summary to CSV using Polars
    summary_df.write_csv("analysis_results/token_summary.csv")
    
    # Generate plots
    analyzer.plot_token_analysis(results)
    
    print("\nAnalysis complete! Results saved in 'analysis_results' directory.")

if __name__ == "__main__":
    main() 