import re
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from pydantic import BaseModel
import sentencepiece as spm


class Chunk(BaseModel):
    """Structured chunk object with clean separation of metadata and content."""
    md_id: str
    chunk_id: int
    content: str
    token_count: int
    start_sentence_idx: int
    end_sentence_idx: int


class TokenCounter:
    """Token counter using SentencePiece (Google-style tokenization)."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to .model file. Required.
        
        Raises:
            ValueError: If model_path is None or file does not exist.
        """
        if not model_path:
            raise ValueError(
                "SentencePiece model_path is required. "
                "Please provide a valid path to a .model file."
            )
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(
                f"SentencePiece model not found at: {model_path.absolute()}"
            )
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.cache: Dict[str, int] = {}
    
    def count(self, text: str) -> int:
        """Count tokens in text using SentencePiece with caching."""
        if text in self.cache:
            return self.cache[text]
        
        token_count = len(self.sp.encode(text))
        self.cache[text] = token_count
        return token_count


def _calculate_overlap(
    current_chunk: List[str],
    target_tokens: int,
    overlap_ratio: float,
    token_counter: TokenCounter
) -> Tuple[List[str], int]:
    """
    Extract overlap sentences from current chunk.
    
    Returns:
        (overlap_sentences, overlap_token_count)
    """
    overlap_tokens = int(target_tokens * overlap_ratio)
    overlap_sentences = []
    overlap_token_count = 0
    
    for sent in reversed(current_chunk):
        sent_tokens = token_counter.count(sent)
        overlap_token_count += sent_tokens
        overlap_sentences.insert(0, sent)
        
        if overlap_token_count > overlap_tokens and len(overlap_sentences) > 1:
            overlap_sentences.pop(0)
            overlap_token_count -= sent_tokens
            break
    
    return overlap_sentences.copy(), overlap_token_count


def chunk_transcript(
    text: str,
    md_id: str,
    target_tokens: int = 350,
    hard_max_tokens: int = 600,
    hard_min_tokens: int = 100,
    overlap_ratio: float = 0.15,
    token_counter: Optional[TokenCounter] = None
) -> List[Chunk]:
    """
    Sentence-based chunking with SentencePiece token awareness.
    
    Returns structured Chunk objects with clean metadata separation.
    
    Properties:
    - Sentence-based: splits on sentence boundaries
    - Token-aware: uses SentencePiece for accurate token counting
    - Deterministic: same input → same output
    - Bounded: enforces hard min/max
    - Overlap: token-based with sentence truncation when necessary
    - Cached: token counts cached to reduce CPU overhead
    
    Args:
        text: Normalized transcript text
        md_id: Document ID (file stem)
        target_tokens: Desired chunk size (300-400)
        hard_max_tokens: Never exceed this (600)
        hard_min_tokens: Minimum viable chunk (100)
        overlap_ratio: Overlap percentage (0.15 = 15%)
        token_counter: TokenCounter instance (required)
    
    Returns:
        List of Chunk objects with structured metadata
    """
    
    if token_counter is None:
        raise ValueError("token_counter is required")
    
    # Validate configuration
    if target_tokens > hard_max_tokens:
        raise ValueError(f"target_tokens ({target_tokens}) cannot exceed hard_max_tokens ({hard_max_tokens})")
    
    # 1. Split into sentences (permissive for speech transcripts)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks: List[Chunk] = []
    chunk_id = 0
    current_chunk = []
    current_tokens = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = token_counter.count(sentence)
        
        # Check if adding this sentence exceeds hard max
        if current_tokens + sentence_tokens > hard_max_tokens and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                md_id=md_id,
                chunk_id=chunk_id,
                content=chunk_text,
                token_count=current_tokens,
                start_sentence_idx=i - len(current_chunk),
                end_sentence_idx=i - 1,
            ))
            chunk_id += 1
            
            # Calculate overlap using helper function
            current_chunk, current_tokens = _calculate_overlap(
                current_chunk, target_tokens, overlap_ratio, token_counter
            )
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        
        # If we've reached target, try to close chunk at sentence boundary
        if current_tokens >= target_tokens:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                md_id=md_id,
                chunk_id=chunk_id,
                content=chunk_text,
                token_count=current_tokens,
                start_sentence_idx=i - len(current_chunk) + 1,
                end_sentence_idx=i,
            ))
            chunk_id += 1
            
            # Calculate overlap using helper function
            current_chunk, current_tokens = _calculate_overlap(
                current_chunk, target_tokens, overlap_ratio, token_counter
            )
    
    # Handle remaining sentences
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_tokens = token_counter.count(chunk_text)
        
        # Only add if meets minimum token requirement
        if chunk_tokens >= hard_min_tokens:
            chunks.append(Chunk(
                md_id=md_id,
                chunk_id=chunk_id,
                content=chunk_text,
                token_count=chunk_tokens,
                start_sentence_idx=len(sentences) - len(current_chunk),
                end_sentence_idx=len(sentences) - 1,
            ))
        elif chunks:
            # If too small, merge with last chunk (if exists and won't exceed max)
            last_chunk = chunks[-1]
            merged_text = last_chunk.content + " " + chunk_text
            merged_tokens = token_counter.count(merged_text)
            
            if merged_tokens <= hard_max_tokens:
                chunks[-1] = Chunk(
                    md_id=last_chunk.md_id,
                    chunk_id=last_chunk.chunk_id,
                    content=merged_text,
                    token_count=merged_tokens,
                    start_sentence_idx=last_chunk.start_sentence_idx,
                    end_sentence_idx=len(sentences) - 1,
                )
    
    return chunks


def write_chunks_as_markdown(chunks: List[Chunk], output_dir: Path) -> None:
    """
    Write chunks to markdown files for inspection.
    
    Metadata is embedded as comment header for easy viewing.
    This is a DEBUG/INSPECTION output only.
    
    Args:
        chunks: List of Chunk objects
        output_dir: Directory to write .md files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk in chunks:
        out_name = f"{chunk.md_id}_chunk_{chunk.chunk_id:03d}.md"
        out_path = output_dir / out_name
        
        # Metadata header for inspection
        metadata = (
            f"<!-- \n"
            f"md_id: {chunk.md_id}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"token_count: {chunk.token_count}\n"
            f"start_sentence_idx: {chunk.start_sentence_idx}\n"
            f"end_sentence_idx: {chunk.end_sentence_idx}\n"
            f"-->\n\n"
        )
        
        out_path.write_text(metadata + chunk.content, encoding="utf-8")


def main():
    """Main: chunk normalized transcripts and write debug output."""
    BASE_DIR = Path(__file__).parents[1]
    IN_DIR = BASE_DIR / "data_cleaned" / "02_normalized"
    OUT_DIR = BASE_DIR / "data_cleaned" / "03_chunked"
    MODEL_PATH = BASE_DIR / "models" / "sentencepiece.model"
    
    # Initialize token counter (with cache)
    token_counter = TokenCounter(model_path=str(MODEL_PATH))
    
    all_chunks: List[Chunk] = []
    
    # Process each normalized transcript
    for file_path in sorted(IN_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        
        # Get structured chunks
        chunks = chunk_transcript(
            text=text,
            md_id=file_path.stem,
            target_tokens=350,
            hard_max_tokens=600,
            hard_min_tokens=100,
            overlap_ratio=0.15,
            token_counter=token_counter
        )
        
        all_chunks.extend(chunks)
        print(f"✓ {file_path.name}: {len(chunks)} chunks")
    
    # Write debug markdown output
    write_chunks_as_markdown(all_chunks, OUT_DIR)
    
    print(f"\n✓ Total chunks: {len(all_chunks)}")
    print(f"✓ Cache stats: {len(token_counter.cache)} unique sentences cached")
    print(f"✓ Debug output written to: {OUT_DIR}")
    
    return all_chunks


if __name__ == "__main__":
    chunks = main()
    
    # Example: print first chunk as JSON
    if chunks:
        print("\nExample chunk (JSON):")
        print(chunks[0].model_dump_json(indent=2))