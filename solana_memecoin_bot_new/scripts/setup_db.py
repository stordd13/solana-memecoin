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

def setup_database():
    """
    Migrates 30k+ Parquets to local SQLite DB (memecoin.db in root): List files in data/raw/dataset/, skip mismatched schemas (e.g., '_tokens_list.parquet'), load token files, extract 'token_id' from filenames, add as column, insert to 'token_prices' table (token_id, datetime, price).
    Run locally; no leakage (raw data only).
    """
    data_dir = config.DATA_RAW_DIR
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    
    conn = sqlite3.connect(os.path.join(bot_new_root, 'memecoin.db'))
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
            else:
                skipped += 1
                print(f"Skipped {os.path.basename(file_path)} (schema: {schema})")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped += 1
    
    conn.commit()
    conn.close()
    print(f"Migration complete: Loaded {loaded} files, skipped {skipped}. DB at memecoin.db")

if __name__ == "__main__":
    setup_database()