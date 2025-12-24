import re
from typing import List
from pathlib import Path
import sys
import tiktoken  # Add this import

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from data_models import TranscriptGeminiChunk

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Common standalone verbal fillers
VERBAL_FILLERS_RE = re.compile(r"(?i)\b(basically|actually|sort of|kind of|you know|et cetera)\b")
# 'So' at the start of paragraphs/lines FIRST (preserve the newlines)
SO_PARA_RE = re.compile(r"(?i)(\n+)\s*so\b(?!\s+(?:that|as))\s*")
#'So' at the start of a line or after a period (but not 'so that')
SO_START_RE = re.compile(r"(?i)([.!?])\s+so\b(?!\s+(?:that|as))")
# "And" at the start of paragraphs/lines (preserve the newlines)
AND_PARA_RE = re.compile(r"(?i)(\n+)\s*and\b\s*")
# "And" at the start of sentences (after punctuation)
AND_START_RE = re.compile(r"(?i)([.!?])\s+and\b\s*")

# Target common conversational starters followed by pronouns and fillers
# Example: "So you basically just", "And then we actually"
CONVERSATIONAL_RE = re.compile(
    r"(?i)\b(so|and|then|now)\b\s+(you|we|i)\s+\b(basically|actually|just|sort of|kind of)\b\s*", 
    re.IGNORECASE
    )

def heavy_clean(text: str) -> str:
    """
    Deep cleaning for LLM context (readability focused).
    
    Additional cleaning beyond light normalization:
    - Fix common speech artifacts
    - Remove filler words (um, uh, like)
    - Fix capitalization
    - Remove redundant punctuation
    """
    # Fillers
    # Remove conversational filler combinations
    text = CONVERSATIONAL_RE.sub("", text)

    # Remove standalone verbal fillers (general cleanup)
    text = VERBAL_FILLERS_RE.sub("", text)

    # Remove "so" at the start of paragraphs/lines FIRST (preserve the newlines)
    text = SO_PARA_RE.sub(r"\1", text)

    # Remove "and" at the start of paragraphs/lines (preserve the newlines)
    text = AND_PARA_RE.sub(r"\1", text)

    # Remove "so" at the start of sentences (after punctuation)
    text = SO_START_RE.sub(r"\1 ", text)

    # Remove "and" at the start of sentences (after punctuation)
    text = AND_START_RE.sub(r"\1 ", text)

    # Fix punctuation issues
    # Remove multiple consecutive punctuation: ".. " or ". ." → "."
    text = re.sub(r'([.!?,])\t*\1+', r'\1', text)
    
    # Clean up punctuation combinations: "., " or ".  ," → "."
    text = re.sub(r'\.\t*,\t*', '. ', text)
    text = re.sub(r',\t*\.\t*', '. ', text)
    
    # Remove stray punctuation at paragraph starts: "\n\n. " or "\n\n, "
    text = re.sub(r'(\n\n+)[ \t]*[.,?!]+[ \t]*', r'\1', text)

    # SPLIT BY PARAGRAPHS (The gaps you want to keep)
    raw_paragraphs = text.split("\n\n")

    # The Final Assembly
    final_blocks = []
    for para in raw_paragraphs:
        # Just strip whitespace from each paragraph, don't collapse internal lines
        clean_para = para.strip()
        if clean_para:  # Only add non-empty paragraphs
            final_blocks.append(clean_para)
            
    # JOIN PARAGRAPHS WITH DOUBLE NEWLINES
    # This restores the structural gaps you want to see.
    return "\n\n".join(final_blocks)
    


def chunk_transcript_two_stream(
    raw_text: str,
    md_id: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[TranscriptGeminiChunk]:
    """
    Two-stream chunking: raw for vectors, cleaned for LLM.
    
    Process:
    1. Chunk using RecursiveCharacterTextSplitter with tiktoken
    2. Heavy clean each chunk for LLM readability
    3. Return both versions aligned
    
    Args:
        raw_text: lightly normalized transcript text
        md_id: Document ID (file stem)
        chunk_size: Target tokens per chunk (default 300)
        chunk_overlap: Token overlap between chunks (default 50)
    
    Returns:
        List of Chunk objects with rawish and cleaned versions
    """
   
    # Step 1: Chunk using tiktoken-aware splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""] # According to Research
    )
    
    raw_chunks = text_splitter.split_text(raw_text)
    
    # Initialize tiktoken encoder for accurate token counts
    encoder = tiktoken.get_encoding("cl100k_base")
    
    # Step 2: Create two-stream chunks
    chunks: List[TranscriptGeminiChunk] = []
    
    for chunk_id, raw_chunk in enumerate(raw_chunks):
        # Heavy clean for LLM context
        cleaned_chunk = heavy_clean(raw_chunk)
        
        # Accurate token count using tiktoken
        token_count = len(encoder.encode(raw_chunk))
        
        chunks.append(TranscriptGeminiChunk(
            md_id=md_id,
            chunk_id=chunk_id,
            raw_content=raw_chunk,
            cleaned_content=cleaned_chunk,
            token_count=token_count
        ))
    
    return chunks


def write_chunks_as_markdown(chunks: List[TranscriptGeminiChunk], output_dir: Path) -> None:
    """
    Write chunks to markdown files for inspection (both versions).
    
    Args:
        chunks: List of Chunk objects
        output_dir: Directory to write .md files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk in chunks:
        out_name = f"{chunk.md_id}_chunk_{chunk.chunk_id:03d}.md"
        out_path = output_dir / out_name
        
        content = (
            f"<!-- \n"
            f"md_id: {chunk.md_id}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"token_count: {chunk.token_count}\n"
            f"-->\n\n"
            f"## RAW (for embeddings)\n\n"
            f"{chunk.raw_content}\n\n"
            f"## CLEANED (for LLM)\n\n"
            f"{chunk.cleaned_content}\n"
        )
        
        out_path.write_text(content, encoding="utf-8")


def main():
    """Main: two-stream chunking with light + heavy cleaning."""
    BASE_DIR = Path(__file__).parents[1]
    IN_DIR = BASE_DIR / "data_cleaned" / "02_normalized"  # Use lightly normalized transcripts
    OUT_DIR = BASE_DIR / "data_cleaned" / "03_chunked"
    
    all_chunks: List[TranscriptGeminiChunk] = []
    
    # Process each transcript
    for file_path in sorted(IN_DIR.glob("*.md")):
        raw_text = file_path.read_text(encoding="utf-8")
        
        # Get two-stream chunks
        chunks = chunk_transcript_two_stream(
            raw_text=raw_text,
            md_id=file_path.stem,
            chunk_size=300,
            chunk_overlap=50
        )
        
        all_chunks.extend(chunks)
        print(f"✓ {file_path.name}: {len(chunks)} chunks")
    
    # Write debug markdown output
    write_chunks_as_markdown(all_chunks, OUT_DIR)
    
    print(f"\n✓ Total chunks: {len(all_chunks)}")
    print(f"✓ Debug output written to: {OUT_DIR}")
    
    return all_chunks


if __name__ == "__main__":
    chunks = main()
    
    # Example: print first chunk as JSON
    if chunks:
        print("\nExample chunk (JSON):")
        print(chunks[0].model_dump_json(indent=2))