import shutil
from pathlib import Path

def get_project_root():
    # This assumes export_utils.py is always in data_analysis/
   return Path(__file__).resolve().parent.parent

def export_parquet_files(token_list, group_name):
    """
    Export parquet files for a list of tokens to the processed/ subfolder based on group_name.
    Paths are always resolved relative to the project root (parent of data_analysis).
    """
    project_root = get_project_root()
    raw_dirs = [
        project_root / "data" / "raw" / "dataset",
        project_root / "data" / "raw" / "dataset-fresh"
    ]
    group_folder_map = {
        "High Quality Tokens": "high_quality_tokens",  # Legacy support
        "Normal Behavior Tokens": "normal_behavior_tokens",
        "Dead Tokens": "dead_tokens",
        "Tokens with Gaps": "tokens_with_gaps",
        "Tokens with Issues": "tokens_with_issues",  # Legacy support
        "Tokens with Extremes": "tokens_with_extremes"
    }
    subfolder = group_folder_map.get(group_name, group_name.replace(" ", "_").lower())
    processed_dir = project_root / "data" / "processed" / subfolder
    if not token_list:
        raise ValueError(f"No tokens to export for {group_name}.")
    processed_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for token in token_list:
        src = None
        for rdir in raw_dirs:
            candidate = rdir / f"{token}.parquet"
            if candidate.exists():
                src = candidate
                break
        if src:
            dst = processed_dir / f"{token}.parquet"
            shutil.copy2(src, dst)
            exported.append(token)
        # If not found in any folder, we silently skip (could log)
    if not exported:
        raise FileNotFoundError(f"No files were exported for {group_name}. Check if the source files exist.")
    return exported 
# %%
