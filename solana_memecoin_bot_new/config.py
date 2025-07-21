import os
import logging

# config.py - Global constants with dynamic paths for multi-file loading
BOT_NEW_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_DIR = os.path.join(BOT_NEW_ROOT, '..', 'data/raw/dataset')  # Parent data/raw/dataset/
DEBUG_MODE = True  # Toggle for verbose debug logs
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO
LOG_FILE = os.path.join(BOT_NEW_ROOT, 'logs/pipeline.log')
TOKENS_LIST_FILE = '_tokens_list.parquet'  # File to exclude during scan
RESAMPLE_INTERVALS = ["1m", "5m"]
N_ARCHETYPES_RANGE = (2, 10)
ZERO_THRESHOLD = 120  # Minutes for death (2 hours - adjusted for memecoin context)
DUMP_RETURN_THRESHOLD = -0.1
EARLY_MINUTES = 10  # Increased to enable rolling features and natural filtering of ultra-short tokens
IMBALANCE_BUY_THRESHOLD = 0.2
HOLD_MINUTES = 10
TRAILING_STOP = 0.2
STAKE_RANGE = (5.0, 50.0)
SLIPPAGE_RATE = 0.03
FEE_RATE = 0.005
MAX_POSITION_FRACTION = 0.005
MAX_DRAWDOWN = 0.05
UPDATE_INTERVAL = 300