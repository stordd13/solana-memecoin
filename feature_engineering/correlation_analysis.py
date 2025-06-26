"""
Correlation Analysis for Memecoin Token Relationships
Implements roadmap requirement for correlation matrix & heatmap analysis
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TokenCorrelationAnalyzer:
    """Analyze correlations and relationships between memecoin tokens"""
    
    def __init__(self):
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        self.time_windows = [60, 240, 720, 1440]  # 1h, 4h, 12h, 24h
    
    def analyze_token_correlations(self, 
                                 token_data: Dict[str, pl.DataFrame],
                                 method: str = 'pearson',
                                 min_overlap: int = 100,
                                 use_log_returns: bool = True) -> Dict:
        """
        Analyze correlations between multiple tokens - HIGH PRIORITY ROADMAP
        """
        
        if len(token_data) < 2:
            return {'error': 'Need at least 2 tokens for correlation analysis'}
        
        try:
            # 1. Prepare synchronized data
            sync_data = self._synchronize_token_data(token_data, use_log_returns)
            
            if sync_data is None or sync_data.shape[1] < 2:
                return {'error': 'Insufficient synchronized data'}
            
            # 2. Calculate correlation matrices - ROADMAP REQUIREMENT
            correlation_matrices = self._calculate_correlation_matrices(sync_data, method)
            
            # 3. Find highly correlated pairs
            corr_pairs = self._find_significant_correlations(correlation_matrices['main'], 
                                                           threshold=0.5)
            
            # 4. PCA analysis for redundancy detection - ROADMAP REQUIREMENT
            pca_analysis = self._pca_redundancy_analysis(sync_data)
            
            return {
                'status': 'success',
                'method': method,
                'tokens_analyzed': list(token_data.keys()),
                'data_points': sync_data.shape[0],
                'correlation_matrices': correlation_matrices,
                'significant_pairs': corr_pairs,
                'pca_analysis': pca_analysis,
                'summary_stats': self._generate_correlation_summary(correlation_matrices['main'])
            }
            
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}
    
    def _synchronize_token_data(self, 
                               token_data: Dict[str, pl.DataFrame],
                               use_log_returns: bool = True) -> Optional[pd.DataFrame]:
        """Synchronize token data to common timeframe"""
        
        synchronized_dfs = []
        
        for token_name, df in token_data.items():
            try:
                # Ensure proper datetime sorting
                df = df.sort('datetime')
                
                # Calculate the metric to correlate using Polars
                if use_log_returns:
                    # Calculate log returns - ROADMAP REQUIREMENT
                    df_with_returns = df.with_columns([
                        (pl.col('price').log() - pl.col('price').shift(1).log()).alias('returns')
                    ])
                    returns_data = df_with_returns[['datetime', 'returns']].to_pandas()
                    returns_data.set_index('datetime', inplace=True)
                    df_sync = pd.DataFrame({token_name: returns_data['returns']})
                else:
                    # Use normalized prices
                    first_price = df['price'][0]
                    df_with_norm = df.with_columns([
                        (pl.col('price') / first_price).alias('normalized_price')
                    ])
                    norm_data = df_with_norm[['datetime', 'normalized_price']].to_pandas()
                    norm_data.set_index('datetime', inplace=True)
                    df_sync = pd.DataFrame({token_name: norm_data['normalized_price']})
                
                synchronized_dfs.append(df_sync)
                
            except Exception as e:
                print(f"Error processing {token_name}: {e}")
                continue
        
        if not synchronized_dfs:
            return None
        
        # Merge all dataframes on datetime index
        merged_df = synchronized_dfs[0]
        for df_sync in synchronized_dfs[1:]:
            merged_df = merged_df.join(df_sync, how='outer')
        
        # Drop rows with any NaN values
        merged_df = merged_df.dropna()
        
        return merged_df
    
    def _calculate_correlation_matrices(self, 
                                      sync_data: pd.DataFrame,
                                      method: str) -> Dict:
        """Calculate correlation matrices - ROADMAP REQUIREMENT"""
        
        matrices = {}
        
        # Main correlation matrix
        matrices['main'] = sync_data.corr(method=method)
        
        # Rolling correlation (if enough data)
        if len(sync_data) > 240:  # 4 hours of data
            matrices['rolling_240'] = sync_data.rolling(240).corr()  # pandas rolling doesn't support method param
        
        return matrices
    
    def _find_significant_correlations(self, 
                                     corr_matrix: pd.DataFrame,
                                     threshold: float = 0.5) -> List[Dict]:
        """Find significantly correlated token pairs"""
        
        significant_pairs = []
        
        # Get upper triangle to avoid duplicates
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        upper_triangle = corr_matrix.where(mask)
        
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if mask[i, j]:
                    corr_val = upper_triangle.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        significant_pairs.append({
                            'token1': corr_matrix.index[i],
                            'token2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'strength': self._interpret_correlation_strength(abs(corr_val)),
                            'direction': 'positive' if corr_val > 0 else 'negative'
                        })
        
        # Sort by absolute correlation value
        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_pairs
    
    def _pca_redundancy_analysis(self, sync_data: pd.DataFrame) -> Dict:
        """PCA analysis - ROADMAP REQUIREMENT: explained_variance_ratio"""
        
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(sync_data.fillna(0))
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate explained variance ratio - ROADMAP REQUIREMENT
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Find number of components for 95% variance
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            return {
                'explained_variance_ratio': explained_variance[:10].tolist(),  # ROADMAP REQUIREMENT
                'cumulative_variance': cumulative_variance[:10].tolist(),
                'n_components_95_variance': int(n_components_95),
                'redundancy_level': 'high' if n_components_95 < len(sync_data.columns) * 0.5 else 'low',
                'first_pc_explains': float(explained_variance[0])
            }
            
        except Exception as e:
            return {'error': f'PCA analysis failed: {str(e)}'}
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength"""
        if abs_corr >= 0.8:
            return 'very_strong'
        elif abs_corr >= 0.6:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_correlation_summary(self, corr_matrix: pd.DataFrame) -> Dict:
        """Generate summary statistics for correlation matrix"""
        
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        upper_triangle_values = corr_matrix.values[mask]
        
        return {
            'avg_correlation': float(np.nanmean(upper_triangle_values)),
            'max_correlation': float(np.nanmax(upper_triangle_values)),
            'min_correlation': float(np.nanmin(upper_triangle_values)),
            'std_correlation': float(np.nanstd(upper_triangle_values)),
            'high_correlations_count': int(np.sum(np.abs(upper_triangle_values) >= 0.7))
        }
    
    def create_correlation_heatmap(self, 
                                 correlation_matrix: pd.DataFrame,
                                 title: str = "Token Correlation Heatmap") -> go.Figure:
        """Create interactive correlation heatmap - ROADMAP REQUIREMENT"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tokens",
            yaxis_title="Tokens",
            width=800,
            height=600
        )
        
        return fig


def load_tokens_for_correlation(data_paths: List[Path], 
                              limit: Optional[int] = None) -> Dict[str, pl.DataFrame]:
    """Load multiple token datasets for correlation analysis"""
    
    token_data = {}
    files_to_process = data_paths[:limit] if limit else data_paths
    
    print(f"Loading {len(files_to_process)} tokens for correlation analysis...")
    
    for path in files_to_process:
        try:
            token_name = path.stem
            df = pl.read_parquet(path)
            
            # Ensure required columns
            if 'price' in df.columns and 'datetime' in df.columns:
                # Basic filtering
                if len(df) >= 60:  # At least 1 hour of data
                    token_data[token_name] = df
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    print(f"Successfully loaded {len(token_data)} tokens")
    return token_data
