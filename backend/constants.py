from pathlib import Path

# Point DATA_PATH to cleaned unique data directory by default.
# Run `python scripts/clean_data.py` to populate `data_cleaned/unique`.
DATA_PATH = Path(__file__).parents[1] / "data_cleaned" / "unique"
VECTOR_DATABASE_PATH = Path(__file__).parents[1] / "knowledge_base"