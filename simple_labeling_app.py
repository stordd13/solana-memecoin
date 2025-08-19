#!/usr/bin/env python3
"""
Simplified Panel app for manually labeling memecoin parquet files.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import HoverTool
import warnings

warnings.filterwarnings('ignore')

# Configure Panel
pn.extension('bokeh', template='material')

class SimpleMemecoinLabeler:
    def __init__(self):
        # Paths
        self.data_dir = Path("data/raw/dataset")
        self.selected_dir = Path("data/handpicked/selected")
        self.not_selected_dir = Path("data/handpicked/not_selected")
        
        # Create directories
        self.selected_dir.mkdir(parents=True, exist_ok=True)
        self.not_selected_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files
        self.all_files = sorted(list(self.data_dir.glob("*.parquet")))
        
        # Filter processed files
        processed = set()
        processed.update([f.name for f in self.selected_dir.glob("*.parquet")])
        processed.update([f.name for f in self.not_selected_dir.glob("*.parquet")])
        
        self.unprocessed_files = [f for f in self.all_files if f.name not in processed]
        self.current_index = 0
        
        # UI Components
        self.info = pn.pane.Markdown("**Loading...**")
        self.plot = pn.pane.Bokeh()
        
        # Buttons
        self.yes_btn = pn.widgets.Button(name='✅ YES', button_type='success', width=120)
        self.no_btn = pn.widgets.Button(name='❌ NO', button_type='danger', width=120)
        self.skip_btn = pn.widgets.Button(name='⏭️ Skip', button_type='warning', width=120)
        
        # Connect callbacks
        self.yes_btn.on_click(self.select_yes)
        self.no_btn.on_click(self.select_no)
        self.skip_btn.on_click(self.skip_file)
        
        # Load first file
        self.load_current_file()
    
    def select_yes(self, event):
        self.label_file('selected')
    
    def select_no(self, event):
        self.label_file('not_selected')
    
    def skip_file(self, event):
        self.next_file()
    
    def label_file(self, label):
        if not self.unprocessed_files or self.current_index >= len(self.unprocessed_files):
            return
        
        file_path = self.unprocessed_files[self.current_index]
        dest_dir = self.selected_dir if label == 'selected' else self.not_selected_dir
        
        # Copy file
        shutil.copy2(file_path, dest_dir / file_path.name)
        
        # Remove from list and move to next
        self.unprocessed_files.pop(self.current_index)
        if self.current_index >= len(self.unprocessed_files) and self.current_index > 0:
            self.current_index -= 1
        
        self.load_current_file()
    
    def next_file(self):
        if self.current_index < len(self.unprocessed_files) - 1:
            self.current_index += 1
            self.load_current_file()
    
    def load_current_file(self):
        if not self.unprocessed_files or self.current_index >= len(self.unprocessed_files):
            self.info.object = "**All files processed!**"
            self.plot.object = None
            return
        
        file_path = self.unprocessed_files[self.current_index]
        
        # Update info
        selected_count = len(list(self.selected_dir.glob("*.parquet")))
        not_selected_count = len(list(self.not_selected_dir.glob("*.parquet")))
        
        self.info.object = f"""
        **File:** {file_path.name}
        **Progress:** {self.current_index + 1} / {len(self.unprocessed_files)}
        **Selected:** {selected_count} | **Not Selected:** {not_selected_count}
        """
        
        try:
            # Load and plot data
            df = pd.read_parquet(file_path)
            self.plot.object = self.create_plot(df, file_path.name)
        except Exception as e:
            self.info.object = f"**Error loading {file_path.name}:** {str(e)}"
            self.plot.object = None
    
    def create_plot(self, df, filename):
        # Prepare data
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['log_price'] = np.log10(df['price'].replace(0, np.nan))
        
        # Create plot
        p = figure(
            title=f"{filename}",
            x_axis_type='datetime',
            width=700,
            height=400,
            tools='pan,wheel_zoom,box_zoom,reset'
        )
        
        p.line(df['datetime'], df['log_price'], line_width=2, color='blue')
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Log10(Price)'
        
        return p
    
    def get_panel(self):
        return pn.Column(
            "# 📊 Memecoin Data Labeling Tool",
            self.info,
            self.plot,
            pn.Row(self.yes_btn, self.no_btn, self.skip_btn),
            pn.pane.Markdown("""
            **Instructions:**
            - Review the price chart (log scale)
            - Click YES if good for training, NO if bad
            - Skip to defer decision
            """),
            sizing_mode='stretch_width'
        )

# Create app instance
labeler = SimpleMemecoinLabeler()

def app():
    print(f"🚀 Starting app with {len(labeler.unprocessed_files)} files to process")
    return labeler.get_panel()

# For Panel serve
pn.serve(app, port=5007, show=True)