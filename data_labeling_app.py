#!/usr/bin/env python3
"""
Interactive Panel app for manually labeling memecoin parquet files.
Displays price charts and allows user to classify as 'selected' or 'not_selected'.
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

# Enable Panel extensions
pn.extension(sizing_mode='stretch_width')

class MemecoinLabeler:
    def __init__(self):
        # Paths
        self.data_dir = Path("data/raw/dataset")
        self.selected_dir = Path("data/handpicked/selected")
        self.not_selected_dir = Path("data/handpicked/not_selected")
        self.skipped_dir = Path("data/handpicked/skipped")
        self.progress_file = Path("data/handpicked/progress.json")
        self.log_file = Path("data/handpicked/processing_log.txt")
        
        # Create directories if they don't exist
        self.selected_dir.mkdir(parents=True, exist_ok=True)
        self.not_selected_dir.mkdir(parents=True, exist_ok=True)
        self.skipped_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_processed = 0
        
        # Track recent actions for undo
        self.recent_actions = []  # List of (filename, action, source_dir)
        
        # Navigation history - keep track of files we've seen
        self.navigation_history = []  # List of file paths we've viewed
        self.history_index = -1  # Current position in history
        self.viewing_history = False  # Flag to track if we're browsing history
        
        # Get list of all parquet files
        self.all_files = sorted(list(self.data_dir.glob("*.parquet")))
        
        # Filter out already processed files
        self.processed_files = set()
        self.processed_files.update([f.name for f in self.selected_dir.glob("*.parquet")])
        self.processed_files.update([f.name for f in self.not_selected_dir.glob("*.parquet")])
        
        # Track skipped files separately
        self.skipped_files = set([f.name for f in self.skipped_dir.glob("*.parquet")])
        
        self.unprocessed_files = [f for f in self.all_files 
                                   if f.name not in self.processed_files 
                                   and f.name not in self.skipped_files]
        
        # Load or initialize progress
        self.progress = self.load_progress()
        
        # Current file index - resume from saved position or start fresh
        self.current_index = self.progress.get('last_index', 0)
        if self.current_index >= len(self.unprocessed_files):
            self.current_index = 0
        
        # Load navigation history from progress
        saved_history = self.progress.get('navigation_history', [])
        self.navigation_history = [Path(p) for p in saved_history if Path(p).exists()]
        self.history_index = self.progress.get('history_index', -1)
        
        # Validate history_index
        if self.history_index >= len(self.navigation_history):
            self.history_index = len(self.navigation_history) - 1
        
        # Initialize UI components
        self.file_label = pn.pane.Markdown("", styles={'font-size': '16px', 'font-weight': 'bold'})
        self.progress_label = pn.pane.Markdown("", styles={'font-size': '14px'})
        self.stats_label = pn.pane.Markdown("", styles={'font-size': '12px', 'color': '#666'})
        self.session_stats = pn.pane.Markdown("", styles={'font-size': '12px', 'color': '#444'})
        
        # Buttons
        self.yes_button = pn.widgets.Button(
            name='✅ YES - Good Pattern', 
            button_type='success',
            width=200,
            height=50,
            margin=(10, 5)
        )
        self.no_button = pn.widgets.Button(
            name='❌ NO - Bad Pattern', 
            button_type='danger',
            width=200,
            height=50,
            margin=(10, 5)
        )
        self.skip_button = pn.widgets.Button(
            name='⏭️ Skip for Later', 
            button_type='warning',
            width=150,
            height=50,
            margin=(10, 5)
        )
        self.prev_button = pn.widgets.Button(
            name='⬅️ Previous', 
            button_type='primary',
            width=100,
            height=50,
            margin=(10, 5)
        )
        self.next_button = pn.widgets.Button(
            name='➡️ Next', 
            button_type='primary',
            width=100,
            height=50,
            margin=(10, 5)
        )
        self.undo_button = pn.widgets.Button(
            name='↩️ UNDO Last', 
            button_type='light',
            width=120,
            height=50,
            margin=(10, 5)
        )
        
        # Connect buttons to actions
        self.yes_button.on_click(lambda event: self.label_file('selected'))
        self.no_button.on_click(lambda event: self.label_file('not_selected'))
        self.skip_button.on_click(lambda event: self.skip_file())
        self.prev_button.on_click(lambda event: self.prev_file())
        self.next_button.on_click(lambda event: self.next_in_history())
        self.undo_button.on_click(lambda event: self.undo_last_action())
        
        # Plot pane
        self.plot_pane = pn.pane.Bokeh(sizing_mode='stretch_both')
        
        # Keyboard shortcuts (JavaScript)
        self.keyboard_js = pn.pane.HTML("""
        <script>
        document.addEventListener('keydown', function(event) {
            if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
                switch(event.key.toLowerCase()) {
                    case 'y':
                        document.querySelector('button[name*="YES"]')?.click();
                        break;
                    case 'n':
                        document.querySelector('button[name*="NO"]')?.click();
                        break;
                    case 'arrowright':
                        document.querySelector('button[name*="Skip"]')?.click();
                        break;
                    case 'arrowleft':
                        document.querySelector('button[name*="Previous"]')?.click();
                        break;
                }
            }
        });
        </script>
        """, height=0, width=0)
        
        # Load first file
        if self.unprocessed_files:
            self.load_current_file()
        else:
            self.file_label.object = "**All files have been processed!**"
    
    def load_progress(self):
        """Load progress from JSON file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'last_index': 0,
            'total_processed': 0,
            'sessions': []
        }
    
    def save_progress(self):
        """Save current progress to JSON file."""
        self.progress['last_index'] = self.current_index
        self.progress['last_file'] = self.unprocessed_files[self.current_index].name if self.current_index < len(self.unprocessed_files) else None
        self.progress['last_updated'] = datetime.now().isoformat()
        
        # Save navigation history
        self.progress['navigation_history'] = [str(p) for p in self.navigation_history]
        self.progress['history_index'] = self.history_index
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def log_action(self, filename, action):
        """Log each labeling action with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp} | {filename} | {action}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def add_to_history(self, file_path):
        """Add a file to navigation history."""
        # Remove from history if it already exists (to avoid duplicates)
        if file_path in self.navigation_history:
            self.navigation_history.remove(file_path)
        
        # Add to end of history
        self.navigation_history.append(file_path)
        self.history_index = len(self.navigation_history) - 1
        
        # Keep history manageable
        if len(self.navigation_history) > 50:
            self.navigation_history.pop(0)
            self.history_index -= 1
    
    def load_current_file(self):
        """Load and display the current file."""
        if not self.unprocessed_files or self.current_index >= len(self.unprocessed_files):
            self.file_label.object = "**No more files to process!**"
            self.plot_pane.object = None
            return
        
        file_path = self.unprocessed_files[self.current_index]
        
        # Always add to navigation history when viewing a new file (not navigating through history)
        if not self.viewing_history:
            self.add_to_history(file_path)
        
        # Reset viewing history mode
        self.viewing_history = False
        
        # Save progress
        self.save_progress()
        
        # Use the display helper
        self._display_file(file_path, f"**File:** {file_path.name}")
        
        # Update progress specifically for current file
        self.progress_label.object = f"**Progress:** {self.current_index + 1} / {len(self.unprocessed_files)} unprocessed files"
    
    def create_plot(self, df, filename):
        """Create an interactive Bokeh plot with log price."""
        # Prepare data
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calculate log price (handle zeros)
        df['log_price'] = np.log10(df['price'].replace(0, np.nan))
        
        # Calculate price change metrics
        price_change = (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100 if df['price'].iloc[0] > 0 else 0
        max_price = df['price'].max()
        min_price = df['price'].min()
        
        # Create figure
        p = figure(
            title=f"{filename} | Change: {price_change:.1f}% | Max: {max_price:.2e} | Min: {min_price:.2e}",
            x_axis_type='datetime',
            x_axis_label='Time',
            y_axis_label='Log10(Price)',
            width=800,
            height=500,
            tools='pan,wheel_zoom,box_zoom,reset,save',
            active_scroll='wheel_zoom'
        )
        
        # Main price line
        p.line(df['datetime'], df['log_price'], 
               line_width=2, color='blue', alpha=0.8)
        
        # Add points for hover
        points = p.circle(df['datetime'], df['log_price'], 
                         size=3, color='blue', alpha=0.5)
        
        # Add hover tool
        hover = HoverTool(renderers=[points])
        hover.tooltips = [
            ('Time', '@x{%F %H:%M}'),
            ('Price', '@y{0.000000}'),
            ('Log Price', '@y{0.00}')
        ]
        hover.formatters = {'@x': 'datetime'}
        p.add_tools(hover)
        
        # Style
        p.title.text_font_size = '14pt'
        p.xaxis.axis_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_size = '12pt'
        
        return p
    
    def label_file(self, label):
        """Label the current file and move to the appropriate directory."""
        if not self.unprocessed_files or self.current_index >= len(self.unprocessed_files):
            return
        
        file_path = self.unprocessed_files[self.current_index]
        
        # Determine destination
        if label == 'selected':
            dest_dir = self.selected_dir
        else:
            dest_dir = self.not_selected_dir
        
        # Copy file
        dest_path = dest_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        
        # Log the action
        self.log_action(file_path.name, label.upper())
        
        # Track for undo
        self.recent_actions.append((file_path.name, label, dest_dir))
        if len(self.recent_actions) > 10:  # Keep last 10 actions
            self.recent_actions.pop(0)
        
        # Update session stats
        self.session_processed += 1
        self.progress['total_processed'] = self.progress.get('total_processed', 0) + 1
        
        # Remove from unprocessed list
        self.unprocessed_files.pop(self.current_index)
        
        # Adjust index if necessary
        if self.current_index >= len(self.unprocessed_files) and self.current_index > 0:
            self.current_index -= 1
        
        # Load next file
        self.load_current_file()
    
    def skip_file(self):
        """Skip current file for later review."""
        if not self.unprocessed_files or self.current_index >= len(self.unprocessed_files):
            return
        
        file_path = self.unprocessed_files[self.current_index]
        
        # Copy to skipped directory
        dest_path = self.skipped_dir / file_path.name
        shutil.copy2(file_path, dest_path)
        
        # Log the action
        self.log_action(file_path.name, "SKIPPED")
        
        # Track for undo
        self.recent_actions.append((file_path.name, "skipped", dest_path))
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        # Remove from unprocessed list
        self.unprocessed_files.pop(self.current_index)
        
        # Adjust index if necessary
        if self.current_index >= len(self.unprocessed_files) and self.current_index > 0:
            self.current_index -= 1
        
        # Load next file
        self.load_current_file()
    
    def next_file(self):
        """Skip to the next file without labeling."""
        if self.current_index < len(self.unprocessed_files) - 1:
            self.current_index += 1
            self.load_current_file()
    
    def prev_file(self):
        """Go back to the previous file in navigation history."""
        if len(self.navigation_history) == 0:
            print("No navigation history available")
            return
        
        if self.history_index > 0:
            self.history_index -= 1
            self.viewing_history = True
            previous_file = self.navigation_history[self.history_index]
            self.load_file_from_history(previous_file)
            self.save_progress()
        else:
            print("Already at the beginning of history")
    
    def next_in_history(self):
        """Go forward in navigation history."""
        if len(self.navigation_history) == 0:
            # No history, just go to next unprocessed file
            self.next_file()
            return
        
        if self.history_index < len(self.navigation_history) - 1:
            self.history_index += 1
            self.viewing_history = True
            next_file = self.navigation_history[self.history_index]
            self.load_file_from_history(next_file)
            self.save_progress()
        else:
            # At the end of history, go to next unprocessed file
            self.viewing_history = False
            self.next_file()
    
    def load_current_file_from_history(self):
        """Load current file without adding to history."""
        file_path = self.unprocessed_files[self.current_index]
        self._display_file(file_path, f"**File:** {file_path.name} (in history)")
    
    def load_file_from_history(self, file_path):
        """Load a file that may have been processed."""
        # Check where this file ended up
        status = "unknown"
        if (self.selected_dir / file_path.name).exists():
            status = "✅ Previously SELECTED"
        elif (self.not_selected_dir / file_path.name).exists():
            status = "❌ Previously NOT SELECTED"
        elif (self.skipped_dir / file_path.name).exists():
            status = "⏭️ Previously SKIPPED"
        
        self._display_file(file_path, f"**File:** {file_path.name} ({status})")
    
    def _display_file(self, file_path, title):
        """Helper method to display a file."""
        # Update labels
        self.file_label.object = title
        
        # Show different progress info based on whether we're viewing history
        if self.viewing_history and len(self.navigation_history) > 0:
            self.progress_label.object = f"**Viewing History:** {self.history_index + 1} / {len(self.navigation_history)} | Current unprocessed: {self.current_index + 1} / {len(self.unprocessed_files)}"
        else:
            self.progress_label.object = f"**Progress:** {self.current_index + 1} / {len(self.unprocessed_files)} unprocessed files"
        
        # Update statistics
        total_files = len(self.all_files)
        selected_count = len(list(self.selected_dir.glob("*.parquet")))
        not_selected_count = len(list(self.not_selected_dir.glob("*.parquet")))
        skipped_count = len(list(self.skipped_dir.glob("*.parquet")))
        self.stats_label.object = f"**Total:** {total_files} | **Selected:** {selected_count} | **Not Selected:** {not_selected_count} | **Skipped:** {skipped_count} | **Remaining:** {len(self.unprocessed_files)}"
        
        # Session statistics
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        avg_time = session_duration / max(self.session_processed, 1)
        est_remaining = (len(self.unprocessed_files) - self.current_index) * avg_time / 60
        
        self.session_stats.object = f"**Session:** {self.session_processed} processed | {session_duration:.1f} min elapsed | ~{est_remaining:.1f} hours remaining"
        
        try:
            # Load data
            df = pd.read_parquet(file_path)
            
            # Create plot
            self.plot_pane.object = self.create_plot(df, file_path.name)
            
        except Exception as e:
            self.plot_pane.object = None
            self.file_label.object = f"**Error loading file:** {str(e)}"
    
    def undo_last_action(self):
        """Undo the last labeling action."""
        if not self.recent_actions:
            print("No recent actions to undo")
            return
        
        # Get last action
        filename, action, dest_path = self.recent_actions.pop()
        
        try:
            # Remove file from destination directory
            if isinstance(dest_path, Path):
                dest_path.unlink()
            else:
                (dest_path / filename).unlink()
            
            # Add file back to unprocessed list
            original_file = self.data_dir / filename
            if original_file.exists() and original_file not in self.unprocessed_files:
                self.unprocessed_files.insert(self.current_index, original_file)
            
            # Log the undo
            self.log_action(filename, f"UNDO_{action.upper()}")
            
            # Reload current file
            self.load_current_file()
            
            print(f"✅ Undone: {filename} ({action})")
            
        except Exception as e:
            print(f"❌ Error undoing action: {e}")
            # Put the action back if it failed
            self.recent_actions.append((filename, action, dest_path))
    
    def get_layout(self):
        """Return the Panel layout."""
        # Instructions
        instructions = pn.pane.Markdown("""
        ## 📊 Memecoin Data Labeling Tool
        
        **Instructions:**
        - Review the price chart (log scale)
        - Click **YES** if the pattern looks good for training
        - Click **NO** if the pattern looks bad/unusable
        - Click **Skip for Later** to review later
        - Use **Previous** to go back
        
        **Keyboard Shortcuts:**
        - `Y` = Yes
        - `N` = No
        - `→` = Skip
        - `←` = Previous
        
        **UNDO Feature:**
        - Click "UNDO Last" to reverse your last action
        - Restores file back to unprocessed state
        - Can undo last 10 actions
        
        **Resume Feature:**
        - Progress auto-saves after each action
        - Close anytime and resume later
        - Already processed files are skipped
        - Check `processing_log.txt` for history
        """, width=300)
        
        # Button row
        button_row = pn.Row(
            self.yes_button,
            self.no_button,
            self.skip_button,
            self.prev_button,
            self.next_button,
            self.undo_button
        )
        
        # Info panel
        info_panel = pn.Column(
            self.file_label,
            self.progress_label,
            self.stats_label,
            self.session_stats,
            margin=(10, 0)
        )
        
        # Main layout - Use simple Column instead of template
        layout = pn.Column(
            "# 📊 Memecoin Data Labeling Tool",
            pn.Row(
                pn.Column(instructions, width=300),
                pn.Column(
                    info_panel,
                    self.plot_pane,
                    button_row,
                    self.keyboard_js
                )
            ),
            sizing_mode='stretch_width'
        )
        
        return layout

# Create function to build the app
def create_app():
    labeler = MemecoinLabeler()
    
    print("\n🚀 Starting Panel app...")
    print("📂 Files to process:", len(labeler.unprocessed_files))
    print("✅ Selected files:", len(list(labeler.selected_dir.glob("*.parquet"))))
    print("❌ Not selected files:", len(list(labeler.not_selected_dir.glob("*.parquet"))))
    print("⏭️  Skipped files:", len(list(labeler.skipped_dir.glob("*.parquet"))))
    
    if labeler.progress.get('last_file'):
        print(f"\n🔄 Resuming from: {labeler.progress['last_file']}")
        print(f"📊 Total processed so far: {labeler.progress.get('total_processed', 0)}")
        print(f"📚 Navigation history: {len(labeler.navigation_history)} files")
    
    return labeler.get_layout()

# Serve the app
app = create_app().servable()