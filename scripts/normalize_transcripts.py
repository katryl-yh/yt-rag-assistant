import re
from pathlib import Path
import shutil

# Anchor paths to the project root
BASE_DIR = Path(__file__).parents[1]
IN_DIR = BASE_DIR / "data_cleaned" / "01_deduplicated" / "unique"
OUT_DIR = BASE_DIR / "data_cleaned" / "02_normalized"

# Pre-compile regex for efficiency
TIMESTAMP_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]")
TILDE_RE = re.compile(r"~~.*?~~")
SPEAKER_LABEL_RE = re.compile(r"(?i)\*\*Kokchun Giang-\d+:\*\*\s*")  # **Kokchun Giang-N:** format
HORIZONTAL_SPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    # 1. Remove artifacts (replace with space to prevent clumping)
    text = TIMESTAMP_RE.sub(" ", text)
    text = TILDE_RE.sub(" ", text)
    text = SPEAKER_LABEL_RE.sub("", text)  

    # 2 Collapse multiple horizontal spaces into one (but preserve newlines)
    text = HORIZONTAL_SPACE_RE.sub(" ", text)

    # 3. SPLIT BY PARAGRAPHS (The gaps you want to keep)
    raw_paragraphs = text.split("\n\n")

    # 4. The Final Assembly
    final_blocks = []
    for para in raw_paragraphs:
        # Just strip whitespace from each paragraph, don't collapse internal lines
        clean_para = para.strip()
        if clean_para:  # Only add non-empty paragraphs
            final_blocks.append(clean_para)
            
    # 5. JOIN PARAGRAPHS WITH DOUBLE NEWLINES
    # This restores the structural gaps you want to see.
    return "\n\n".join(final_blocks)

def main():
    if not IN_DIR.exists():
        print(f"Error: Source directory {IN_DIR} does not exist. Did you run deduplication?")
        return

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(IN_DIR.glob("*.md"))
    for file_path in files:
        # Added errors="replace" for safety
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        normalized_text = normalize_text(raw_text)
        
        out_path = OUT_DIR / file_path.name
        out_path.write_text(normalized_text, encoding="utf-8")
    
    print(f"Normalization complete. Processed {len(files)} files into {OUT_DIR}")

if __name__ == "__main__":
    main()