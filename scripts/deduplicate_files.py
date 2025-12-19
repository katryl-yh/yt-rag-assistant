"""Deduplicate markdown files in `data/` using exact hashing.

Usage:
The script creates folders: /unique/` and `/duplicates/` in data_cleaned/01_deduplicated
and a CSV report.
"""
from pathlib import Path
import hashlib
import shutil
import csv


DATA_DIR = Path(__file__).parents[1] / "data"
OUT_DIR = Path(__file__).parents[1] / "data_cleaned" / "01_deduplicated"


def get_file_hash(file_path: Path) -> str:
    """Get SHA256 hash of normalized file content."""
    text = file_path.read_text(encoding="utf-8")
    # normalize: strip and collapse whitespace
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main():
    # Make idempotent: remove existing data_cleaned and recreate
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    
    unique_dir = OUT_DIR / "unique"
    dup_dir = OUT_DIR / "duplicates"
    unique_dir.mkdir(parents=True)
    dup_dir.mkdir(parents=True)

    files = sorted(DATA_DIR.glob("*.md"))
    if not files:
        print("No .md files found in data/")
        return

    # Find duplicates by hash
    seen_hashes = {}
    report_rows = []
    
    for file_path in files:
        file_hash = get_file_hash(file_path)
        
        if file_hash in seen_hashes:
            # Duplicate - copy to duplicates folder
            dst = dup_dir / file_path.name
            shutil.copy2(file_path, dst)
            canonical_name = seen_hashes[file_hash].name
            report_rows.append((str(file_path), str(dst), file_hash, f"duplicate_of:{canonical_name}"))
        else:
            # Unique - copy to unique folder
            dst = unique_dir / file_path.name
            shutil.copy2(file_path, dst)
            seen_hashes[file_hash] = file_path
            report_rows.append((str(file_path), str(dst), file_hash, "unique"))

    # Write CSV report
    report_file = OUT_DIR / "data_cleaning_report.csv"
    with report_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["original_path", "new_path", "hash", "status"])
        writer.writerows(report_rows)

    unique_count = len(list(unique_dir.glob("*.md")))
    dup_count = len(list(dup_dir.glob("*.md")))
    
    print(f"Done! Processed {len(files)} files:")
    print(f" - {unique_count} unique files → {unique_dir}")
    print(f" - {dup_count} duplicates → {dup_dir}")
    print(f" - Report: {report_file}")


if __name__ == "__main__":
    main()
