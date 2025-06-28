"""
Enhanced Token Correlation Analysis for Memecoin Analysis
Implements multi-method correlation analysis with PCA redundancy detection - ROADMAP SECTION 3

Key Features:
- Multi-method correlations (Pearson, Spearman, Kendall) - ROADMAP REQUIREMENT
- PCA explained variance ratio analysis - ROADMAP REQUIREMENT
- Log-returns correlation analysis - ROADMAP REQUIREMENT
- Interactive correlation heatmaps
- Significant correlation pair detection
"""

import polars as pl
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
                                 use_log_returns: bool = True,
                                 use_robust_scaling: bool = False,
                                 use_rolling: bool = False,
                                 rolling_window: Optional[int] = None) -> Dict:
        """
        Analyze correlations between multiple tokens - HIGH PRIORITY ROADMAP
        
        Args:
            token_data: Dictionary of token dataframes
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_overlap: Minimum number of overlapping data points required
            use_log_returns: Whether to use log returns instead of prices
            use_robust_scaling: Whether to use RobustScaler for price normalization
            use_rolling: Whether to calculate rolling correlations
            rolling_window: Size of rolling window in minutes
        """
        
        if len(token_data) < 2:
            return {'error': 'Need at least 2 tokens for correlation analysis'}
        
        try:
            # 1. Prepare synchronized data
            sync_data = self._synchronize_token_data(
                token_data, 
                use_log_returns, 
                use_robust_scaling
            )
            
            if sync_data is None or sync_data.shape[1] < 2:
                return {'error': 'Insufficient synchronized data'}
            
            # 2. Check minimum overlap requirement
            if sync_data.height < min_overlap:
                return {
                    'error': f'Insufficient data overlap: {sync_data.height} points < {min_overlap} required',
                    'data_points_found': sync_data.height,
                    'min_overlap_required': min_overlap
                }
            
            # 3. Calculate correlation matrices - ROADMAP REQUIREMENT
            correlation_matrices = self._calculate_correlation_matrices(
                sync_data, 
                method,
                use_rolling,
                rolling_window
            )
            
            # 4. Find highly correlated pairs
            corr_pairs = self._find_significant_correlations(correlation_matrices['main'], 
                                                           threshold=0.5)
            
            # 5. PCA analysis for redundancy detection - ROADMAP REQUIREMENT
            pca_analysis = self._pca_redundancy_analysis(sync_data)
            
            return {
                'status': 'success',
                'method': method,
                'tokens_analyzed': list(token_data.keys()),
                'data_points': sync_data.shape[0],
                'min_overlap_met': True,
                'correlation_matrices': correlation_matrices,
                'significant_pairs': corr_pairs,
                'pca_analysis': pca_analysis,
                'summary_stats': self._generate_correlation_summary(correlation_matrices['main']),
                'rolling_analysis': use_rolling,
                'rolling_window': rolling_window,
                'scaling_method': 'robust_scaler' if use_robust_scaling else 'simple_normalization' if not use_log_returns else 'log_returns'
            }
            
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}
    
    def _synchronize_token_data(self, 
                               token_data: Dict[str, pl.DataFrame],
                               use_log_returns: bool = True,
                               use_robust_scaling: bool = False) -> Optional[pl.DataFrame]:
        """Synchronize token data to common timeframe using Polars"""
        
        prepared_dfs = []
        
        # If using robust scaling, fit scaler on all price data first
        if use_robust_scaling and not use_log_returns:
            # Collect all prices for fitting the scaler
            all_prices = []
            for df in token_data.values():
                prices = df['price'].to_numpy()
                all_prices.extend(prices[~np.isnan(prices)])
            
            if len(all_prices) == 0:
                return None
                
            # Fit RobustScaler on all price data
            scaler = RobustScaler()
            scaler.fit(np.array(all_prices).reshape(-1, 1))
        else:
            scaler = None
        
        for token_name, df in token_data.items():
            try:
                # Ensure proper datetime sorting with Polars
                df = df.sort('datetime')
                
                # Calculate the metric to correlate using Polars
                if use_log_returns:
                    # Calculate log returns - ROADMAP REQUIREMENT
                    df_with_returns = df.with_columns([
                        (pl.col('price').log() - pl.col('price').shift(1).log()).alias('returns')
                    ]).drop_nulls('returns')
                    
                    # Rename the column to include token name
                    df_sync = df_with_returns.select([
                        pl.col('datetime'),
                        pl.col('returns').alias(token_name)
                    ])
                
                elif use_robust_scaling:
                    # Use RobustScaler for price normalization
                    prices = df['price'].to_numpy().reshape(-1, 1)
                    scaled_prices = scaler.transform(prices).flatten()
                    
                    # Create DataFrame with scaled prices
                    df_with_scaled = df.with_columns([
                        pl.Series('scaled_price', scaled_prices)
                    ])
                    
                    df_sync = df_with_scaled.select([
                        pl.col('datetime'),
                        pl.col('scaled_price').alias(token_name)
                    ])
                
                else:
                    # Use simple normalization (divide by first price)
                    first_price = df['price'][0]
                    df_with_norm = df.with_columns([
                        (pl.col('price') / first_price).alias('normalized_price')
                    ])
                    
                    df_sync = df_with_norm.select([
                        pl.col('datetime'),
                        pl.col('normalized_price').alias(token_name)
                    ])
                
                prepared_dfs.append(df_sync)
                
            except Exception as e:
                print(f"Error processing {token_name}: {e}")
                continue
        
        if not prepared_dfs:
            return None
        
        # Merge all dataframes using Polars join operations
        merged_df = prepared_dfs[0]
        for df_sync in prepared_dfs[1:]:
            merged_df = merged_df.join(df_sync, on='datetime', how='inner')
        
        # Drop rows with any null values
        merged_df = merged_df.drop_nulls()
        
        # Return Polars DataFrame instead of converting to pandas
        if merged_df.height > 0:
            return merged_df
        
        return None
    
    def _calculate_correlation_matrices(self, 
                                      sync_data: pl.DataFrame,
                                      method: str,
                                      use_rolling: bool = False,
                                      rolling_window: Optional[int] = None) -> Dict:
        """Calculate correlation matrices - ROADMAP REQUIREMENT"""
        
        matrices = {}
        
        # Get numeric columns (exclude datetime)
        numeric_cols = [col for col in sync_data.columns if col != 'datetime']
        
        # Convert to numpy for correlation calculation
        data_matrix = sync_data.select(numeric_cols).to_numpy()
        
        # Calculate main correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(data_matrix.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(data_matrix, axis=0)
        else:  # kendall
            from scipy.stats import kendalltau
            n_cols = len(numeric_cols)
            corr_matrix = np.zeros((n_cols, n_cols))
            for i in range(n_cols):
                for j in range(n_cols):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        tau, _ = kendalltau(data_matrix[:, i], data_matrix[:, j])
                        corr_matrix[i, j] = tau
        
        # Convert to Polars DataFrame with token names as both index and columns
        matrices['main'] = pl.DataFrame(
            corr_matrix,
            schema={col: pl.Float64 for col in numeric_cols}
        ).with_columns(pl.Series("token", numeric_cols))
        
        # Enhanced rolling correlation analysis
        if use_rolling and rolling_window and len(sync_data) > rolling_window:
            matrices['rolling_correlations'] = self._calculate_rolling_correlations(
                sync_data, numeric_cols, rolling_window, method
            )
        
        return matrices
    
    def _calculate_rolling_correlations(self, 
                                      sync_data: pl.DataFrame,
                                      numeric_cols: List[str],
                                      window: int,
                                      method: str) -> pl.DataFrame:
        """Calculate rolling correlations between all token pairs"""
        
        rolling_results = []
        
        # Calculate rolling correlations for all unique pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                token1, token2 = numeric_cols[i], numeric_cols[j]
                
                # Calculate rolling correlation using Polars
                rolling_corr = sync_data.select([
                    pl.col('datetime'),
                    pl.corr(
                        pl.col(token1).rolling_mean(window),
                        pl.col(token2).rolling_mean(window)
                    ).alias(f'{token1}_{token2}_correlation')
                ]).drop_nulls()
                
                rolling_results.append({
                    'token_pair': f"{token1}_{token2}",
                    'rolling_data': rolling_corr
                })
        
        # Combine into single result
        if rolling_results:
            return rolling_results[0]['rolling_data']  # Return first pair for now
        
        return pl.DataFrame()  # Empty DataFrame if no results
    
    def _find_significant_correlations(self, 
                                     corr_matrix: pl.DataFrame,
                                     threshold: float = 0.5) -> List[Dict]:
        """Find significantly correlated token pairs"""
        
        significant_pairs = []
        
        # Get numeric columns and token names
        numeric_cols = [col for col in corr_matrix.columns if col != 'token']
        token_names = corr_matrix['token'].to_list()
        
        # Convert to numpy for analysis
        corr_values = corr_matrix.select(numeric_cols).to_numpy()
        
        # Get upper triangle to avoid duplicates
        mask = np.triu(np.ones_like(corr_values), k=1).astype(bool)
        
        for i in range(len(token_names)):
            for j in range(len(token_names)):
                if mask[i, j]:
                    corr_val = corr_values[i, j]
                    if abs(corr_val) >= threshold:
                        significant_pairs.append({
                            'token1': token_names[i],
                            'token2': token_names[j],
                            'correlation': float(corr_val),
                            'strength': self._interpret_correlation_strength(abs(corr_val)),
                            'direction': 'positive' if corr_val > 0 else 'negative'
                        })
        
        # Sort by absolute correlation value
        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_pairs
    
    def _pca_redundancy_analysis(self, sync_data: pl.DataFrame) -> Dict:
        """PCA analysis - ROADMAP REQUIREMENT: explained_variance_ratio"""
        
        try:
            # Get numeric columns (exclude datetime)
            numeric_cols = [col for col in sync_data.columns if col != 'datetime']
            
            # Convert to numpy and handle nulls
            data_matrix = sync_data.select(numeric_cols).fill_null(0).to_numpy()
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_matrix)
            
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
                'redundancy_level': 'high' if n_components_95 < len(numeric_cols) * 0.5 else 'low',
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
    
    def _generate_correlation_summary(self, corr_matrix: pl.DataFrame) -> Dict:
        """Generate summary statistics for correlation matrix"""
        
        # Get numeric columns (exclude token)
        numeric_cols = [col for col in corr_matrix.columns if col != 'token']
        corr_values = corr_matrix.select(numeric_cols).to_numpy()
        
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(corr_values), k=1).astype(bool)
        upper_triangle_values = corr_values[mask]
        
        return {
            'avg_correlation': float(np.nanmean(upper_triangle_values)),
            'max_correlation': float(np.nanmax(upper_triangle_values)),
            'min_correlation': float(np.nanmin(upper_triangle_values)),
            'std_correlation': float(np.nanstd(upper_triangle_values)),
            'high_correlations_count': int(np.sum(np.abs(upper_triangle_values) >= 0.7))
        }
    
    def create_correlation_heatmap(self, 
                                 correlation_matrix: pl.DataFrame,
                                 title: str = "Token Correlation Heatmap") -> go.Figure:
        """Create interactive correlation heatmap - ROADMAP REQUIREMENT"""
        
        # Get numeric columns and token names
        numeric_cols = [col for col in correlation_matrix.columns if col != 'token']
        token_names = correlation_matrix['token'].to_list()
        corr_values = correlation_matrix.select(numeric_cols).to_numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_values,
            x=token_names,
            y=token_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_values, 2),
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
