import polars as pl
import sqlite3
import os
import glob
import sys

# Dynamic path fix
scripts_dir = os.path.dirname(__file__)
bot_new_root = os.path.abspath(os.path.join(scripts_dir, '..'))
sys.path.append(bot_new_root)
import config
from utils import setup_logger  # New import for logging

logger = setup_logger(__name__)

def setup_database():
    """
    Migrates 30k+ Parquets to local SQLite DB (memecoin.db in root): List files in data/raw/dataset/, skip mismatched schemas (e.g., '_tokens_list.parquet'), load token files, extract 'token_id' from filenames, add as column, insert to 'token_prices' table (token_id, datetime, price).
    Run locally; no leakage (raw data only). Adds indexes for perf.
    """
    data_dir = config.DATA_RAW_DIR
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    
    db_path = os.path.join(bot_new_root, 'memecoin.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS token_prices (token_id TEXT, datetime TEXT, price FLOAT)')
    
    loaded = 0
    skipped = 0
    for file_path in all_files:
        try:
            lazy_df = pl.scan_parquet(file_path)
            schema = lazy_df.schema
            if 'datetime' in schema and 'price' in schema:
                token_id = os.path.splitext(os.path.basename(file_path))[0]
                df = lazy_df.with_columns(pl.lit(token_id).alias("token_id")).collect()
                for row in df.iter_rows():
                    cursor.execute('INSERT INTO token_prices VALUES (?, ?, ?)', (row[2], row[0], row[1]))  # token_id, datetime, price
                loaded += 1
                logger.info(f"Loaded {os.path.basename(file_path)}")
            else:
                skipped += 1
                logger.warning(f"Skipped {os.path.basename(file_path)} (schema: {schema})")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            skipped += 1
    
    # Add indexes for performance (on frequently queried columns)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_id ON token_prices (token_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_datetime ON token_prices (datetime)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_datetime ON token_prices (token_id, datetime)')
    
    conn.commit()
    conn.close()
    logger.info(f"Migration complete: Loaded {loaded} files, skipped {skipped}. Indexes added. DB at {db_path}")

if __name__ == "__main__":
    setup_database()