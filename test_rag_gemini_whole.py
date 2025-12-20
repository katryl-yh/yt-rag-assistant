import asyncio
import argparse
import os

async def test_rag():
    from backend.rag import rag_agent  # import after env is set
    query = "Explain what ETL is and give a short example."
    result = await rag_agent.run(query)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--whole", action="store_true", help="Use transcripts_gemini_whole")
    group.add_argument("--chunked", action="store_true", help="Use transcripts_gemini_chunked")
    args = parser.parse_args()

    db_type = "chunked" if args.chunked else "whole"
    os.environ["RAG_DB_TYPE"] = db_type

    asyncio.run(test_rag())