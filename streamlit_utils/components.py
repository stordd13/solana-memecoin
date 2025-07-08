"""
Shared Streamlit components to eliminate redundancy across apps.
Provides common UI patterns for token selection, data source management, and navigation.
"""

import streamlit as st
import polars as pl
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import random

from .formatting import format_large_number, format_file_count


class DataSourceManager:
    """Manages data source selection with consistent UI patterns across apps."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the data source manager.
        
        Args:
            project_root: Path to project root. If None, attempts to auto-detect.
        """
        if project_root is None:
            # Try to detect project root from caller
            import inspect
            frame = inspect.currentframe().f_back
            caller_file = Path(frame.f_globals['__file__'])
            self.project_root = caller_file.parent.parent
        else:
            self.project_root = project_root
            
        self.data_root = self.project_root / "data"
    
    def get_available_subfolders(self) -> List[Tuple[str, int]]:
        """Get all available data subfolders with parquet files."""
        common_subfolders = [
            "raw/dataset",
            "processed", 
            "cleaned",
            "features"
        ]
        
        available_subfolders = []
        
        # Check common subfolders
        for subfolder in common_subfolders:
            subfolder_path = self.data_root / subfolder
            if subfolder_path.exists():
                parquet_files = list(subfolder_path.rglob('*.parquet'))
                if parquet_files:
                    available_subfolders.append((subfolder, len(parquet_files)))
        
        # Find other subfolders with parquet files
        for root, dirs, files in os.walk(self.data_root):
            if any(f.endswith('.parquet') for f in files):
                rel_path = os.path.relpath(root, self.data_root)
                parquet_count = len([f for f in files if f.endswith('.parquet')])
                if rel_path not in common_subfolders and (rel_path, parquet_count) not in available_subfolders:
                    available_subfolders.append((rel_path, parquet_count))
        
        return available_subfolders
    
    def render_data_source_selection(self, 
                                   session_key_prefix: str = "", 
                                   show_info: bool = True) -> Optional[str]:
        """
        Render data source selection UI in sidebar.
        
        Args:
            session_key_prefix: Prefix for session state keys to avoid conflicts
            show_info: Whether to show selected folder info
            
        Returns:
            Selected subfolder name if data loaded, None otherwise
        """
        if f'{session_key_prefix}data_loaded' in st.session_state and st.session_state[f'{session_key_prefix}data_loaded']:
            return st.session_state.get(f'{session_key_prefix}selected_subfolder')
        
        st.sidebar.subheader("Select Data Source")
        
        available_subfolders = self.get_available_subfolders()
        
        if not available_subfolders:
            st.sidebar.error("No parquet files found in data directory!")
            return None
        
        # Create selectbox with subfolder info
        subfolder_options = [f"{sf} ({format_file_count(count)})" for sf, count in available_subfolders]
        
        if f'{session_key_prefix}selected_subfolder_idx' not in st.session_state:
            st.session_state[f'{session_key_prefix}selected_subfolder_idx'] = 0
        
        selected_idx = st.sidebar.selectbox(
            "Choose data subfolder:",
            range(len(subfolder_options)),
            format_func=lambda x: subfolder_options[x],
            index=st.session_state[f'{session_key_prefix}selected_subfolder_idx'],
            key=f"{session_key_prefix}subfolder_select"
        )
        
        selected_subfolder = available_subfolders[selected_idx][0]
        file_count = available_subfolders[selected_idx][1]
        
        if show_info:
            st.sidebar.info(f"Selected: `data/{selected_subfolder}`\\n{format_file_count(file_count)}")
        
        return selected_subfolder, file_count
    
    def render_load_button(self, 
                          selected_subfolder: str, 
                          session_key_prefix: str = "",
                          data_loader_class=None) -> bool:
        """
        Render load data button and handle loading.
        
        Args:
            selected_subfolder: Subfolder to load data from
            session_key_prefix: Prefix for session state keys
            data_loader_class: Class to instantiate for data loading
            
        Returns:
            True if data was successfully loaded, False otherwise
        """
        if st.sidebar.button("Load Data", type="primary", key=f"{session_key_prefix}load_data"):
            try:
                if data_loader_class:
                    data_loader = data_loader_class(subfolder=selected_subfolder)
                    # Pre-cache the tokens
                    data_loader.get_available_tokens()
                    st.session_state[f'{session_key_prefix}data_loader'] = data_loader
                
                st.session_state[f'{session_key_prefix}data_loaded'] = True
                st.session_state[f'{session_key_prefix}selected_subfolder'] = selected_subfolder
                st.success(f"Data loaded from data/{selected_subfolder} successfully!")
                st.rerun()
                return True
            except Exception as e:
                st.error(f"Error loading data from data/{selected_subfolder}: {e}")
                return False
        return False


class TokenSelector:
    """Manages token selection with consistent UI patterns across apps."""
    
    def __init__(self, data_loader, key_prefix: str = ""):
        """
        Initialize the token selector.
        
        Args:
            data_loader: Data loader instance with get_available_tokens() method
            key_prefix: Prefix for session state keys to avoid conflicts
        """
        self.data_loader = data_loader
        self.key_prefix = key_prefix
    
    def get_available_tokens(self) -> List[Dict]:
        """Get available tokens from data loader."""
        return self.data_loader.get_available_tokens()
    
    def render_selection_mode(self) -> str:
        """Render selection mode radio buttons in sidebar."""
        return st.sidebar.radio(
            "Token Selection Mode",
            ["Single Token", "Multiple Tokens", "Random Tokens", "All Tokens"],
            help="Choose how to select tokens for analysis",
            key=f"{self.key_prefix}selection_mode"
        )
    
    def render_single_token_selector(self, available_tokens: List[Dict]) -> List[Dict]:
        """Render single token selection."""
        token_symbols = sorted([t['symbol'] for t in available_tokens])
        
        if f'{self.key_prefix}single_token' not in st.session_state or st.session_state[f'{self.key_prefix}single_token'] not in token_symbols:
            st.session_state[f'{self.key_prefix}single_token'] = token_symbols[0] if token_symbols else None
        
        selected_symbol = st.sidebar.selectbox(
            "Select Token:",
            token_symbols,
            index=token_symbols.index(st.session_state[f'{self.key_prefix}single_token']) if st.session_state[f'{self.key_prefix}single_token'] in token_symbols else 0,
            key=f"{self.key_prefix}single_token_select"
        )
        
        st.session_state[f'{self.key_prefix}single_token'] = selected_symbol
        return [t for t in available_tokens if t['symbol'] == selected_symbol]
    
    def render_multiple_token_selector(self, available_tokens: List[Dict]) -> List[Dict]:
        """Render multiple token selection."""
        token_symbols = sorted([t['symbol'] for t in available_tokens])
        
        if f'{self.key_prefix}selected_tokens' not in st.session_state:
            st.session_state[f'{self.key_prefix}selected_tokens'] = token_symbols[:5] if len(token_symbols) >= 5 else token_symbols
        
        selected_symbols = st.sidebar.multiselect(
            "Select Tokens:",
            token_symbols,
            default=st.session_state[f'{self.key_prefix}selected_tokens'],
            key=f"{self.key_prefix}multi_token_select"
        )
        
        st.session_state[f'{self.key_prefix}selected_tokens'] = selected_symbols
        return [t for t in available_tokens if t['symbol'] in selected_symbols]
    
    def render_random_token_selector(self, available_tokens: List[Dict]) -> List[Dict]:
        """Render random token selection."""
        max_tokens = len(available_tokens)
        num_tokens = st.sidebar.number_input(
            "Number of Random Tokens:",
            min_value=1,
            max_value=max_tokens,
            value=min(10, max_tokens),
            key=f"{self.key_prefix}num_random_tokens"
        )
        
        if st.sidebar.button("Select Random Tokens", key=f"{self.key_prefix}select_random"):
            random.seed(42)  # For reproducibility
            st.session_state[f'{self.key_prefix}random_tokens'] = random.sample(available_tokens, num_tokens)
            st.sidebar.success(f"Selected {num_tokens} random tokens")
        
        return st.session_state.get(f'{self.key_prefix}random_tokens', [])
    
    def render_token_selection(self) -> List[Dict]:
        """
        Render complete token selection UI based on mode.
        
        Returns:
            List of selected token dictionaries
        """
        available_tokens = self.get_available_tokens()
        
        if not available_tokens:
            st.sidebar.error("No tokens available!")
            return []
        
        selection_mode = self.render_selection_mode()
        
        if selection_mode == "Single Token":
            return self.render_single_token_selector(available_tokens)
        elif selection_mode == "Multiple Tokens":
            return self.render_multiple_token_selector(available_tokens)
        elif selection_mode == "Random Tokens":
            return self.render_random_token_selector(available_tokens)
        elif selection_mode == "All Tokens":
            return available_tokens
        
        return []
    
    def display_selection_summary(self, selected_tokens: List[Dict]) -> None:
        """Display summary of selected tokens."""
        if selected_tokens:
            st.sidebar.success(f"Selected {format_large_number(len(selected_tokens))} tokens")
            if len(selected_tokens) <= 10:
                token_names = [t['symbol'] for t in selected_tokens]
                st.sidebar.write(f"**Tokens**: {', '.join(token_names)}")
            else:
                st.sidebar.write(f"**Tokens**: {', '.join([t['symbol'] for t in selected_tokens[:5]])} and {len(selected_tokens)-5} more...")


class NavigationManager:
    """Manages common navigation elements and controls."""
    
    def __init__(self, session_key_prefix: str = ""):
        """
        Initialize navigation manager.
        
        Args:
            session_key_prefix: Prefix for session state keys
        """
        self.key_prefix = session_key_prefix
    
    def render_change_data_source_button(self) -> bool:
        """
        Render change data source button.
        
        Returns:
            True if button was clicked
        """
        if st.sidebar.button("Change Data Source", key=f"{self.key_prefix}change_data_source"):
            # Clear data-related session state
            keys_to_clear = [
                f'{self.key_prefix}data_loaded',
                f'{self.key_prefix}data_loader', 
                f'{self.key_prefix}selected_subfolder',
                f'{self.key_prefix}selected_subfolder_idx',
                f'{self.key_prefix}selected_datasets'
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()
            return True
        return False
    
    def render_refresh_analyzers_button(self, analyzer_keys: List[str]) -> bool:
        """
        Render refresh analyzers button.
        
        Args:
            analyzer_keys: List of session state keys for analyzers to refresh
            
        Returns:
            True if button was clicked
        """
        if st.sidebar.button("Refresh Analyzers", key=f"{self.key_prefix}refresh_analyzers"):
            for key in analyzer_keys:
                if key in st.session_state:
                    st.session_state[key] = None
            
            st.success("Analyzers refreshed!")
            st.rerun()
            return True
        return False
    
    def render_common_sidebar_controls(self, analyzer_keys: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Render common sidebar controls.
        
        Args:
            analyzer_keys: Keys for analyzers to refresh
            
        Returns:
            Dictionary with button states
        """
        if analyzer_keys is None:
            analyzer_keys = []
            
        results = {}
        results['change_data_source'] = self.render_change_data_source_button()
        results['refresh_analyzers'] = self.render_refresh_analyzers_button(analyzer_keys)
        
        return results