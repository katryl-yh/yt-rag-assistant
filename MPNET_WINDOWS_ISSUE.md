# Local Embeddings Experiment: Windows Compatibility Issue

## Overview
Attempted to implement local sentence-transformers embeddings as an alternative to Gemini API embeddings for cost savings and offline capability.

## Goal
Replace Gemini API calls with local `sentence-transformers` models:
- **all-MiniLM-L6-v2** (384-dimensional, faster/lighter)
- **all-mpnet-base-v2** (768-dimensional, better quality)

## Problem: Segmentation Fault on Windows

### Error Details
```
Setting up vector DB at: C:\Users\Katrin\Documents\github\yt-rag-assistant\knowledge_base\transcripts_mpnet_whole
Ingesting files:   2%|▏          | 1/47 [00:05<04:33,  5.95s/it]Segmentation fault
Exit Code: 139
```

### What Was Tried

1. **Model Variations**
   - all-MiniLM-L6-v2 (384-dim) → Segmentation fault
   - all-mpnet-base-v2 (768-dim) → Segmentation fault

2. **Environment Workarounds**
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
   os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism
   os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
   ```
   **Result**: Still crashes

3. **Embedding Computation Approaches**
   - Let LanceDB compute embeddings automatically → Crash
   - Manually compute with SentenceTransformer → Crash
   - Crash occurs at the first file during embedding computation

### Root Cause
Known issue with sentence-transformers on Windows:
- The segmentation fault occurs during the actual embedding computation, not during model loading
- This is a system-level crash that cannot be caught/handled in Python

### Affected Files (on this branch)
- `backend/data_models.py` - Updated TranscriptLocalWhole schema for mpnet
- `ingestion_mpnet_whole.py` - Modified to use mpnet model
- `explorations.ipynb` - Added debugging cells

## Alternatives Considered

### 1. Use Linux/WSL 
The same code works fine on Linux systems. Consider
- Using Docker container for ingestion

### 2. Stick with Gemini Embeddings 
Continue using Gemini API which works reliably:
- Already implemented and tested
- Good quality embeddings (3072-dim)

## Recommendation
**Keep this experimental branch for future reference, but continue with Gemini embeddings on main branch.**

## Branch Info
- Branch: `experiment/mpnet-windows-segfault`
- Status: **NOT WORKING ON WINDOWS**
- Created: December 19, 2025
- DO NOT MERGE TO MAIN

## Useful Links
- [sentence-transformers Windows issues](https://github.com/UKPLab/sentence-transformers/issues)
- [ONNX Runtime Windows compatibility](https://github.com/microsoft/onnxruntime/issues)
