import lancedb
import asyncio
from backend.constants import VECTOR_DATABASE_PATH
from backend.rag import rag_agent  # uses your agent in backend/rag.py

DB = lancedb.connect(uri=VECTOR_DATABASE_PATH)
TBL = DB["transcripts"]

def get_top_docs(query: str, k: int = 5):
    return TBL.search(query=query).limit(k).to_list()

def assemble_context(docs, per_doc_chars=1000, total_chars=4000):
    parts = []
    for r in docs:
        content = r.get("content", "")
        if len(content) > per_doc_chars:
            content = content[:per_doc_chars] + "..."
        parts.append(
            f"Filename: {r.get('filename')}\nFilepath: {r.get('filepath')}\nContent:\n{content}"
        )
    ctx = "\n\n---\n\n".join(parts)
    return ctx[:total_chars]

def answer_with_context(query: str, k=5):
    docs = get_top_docs(query, k=k)
    if not docs:
        return "No documents found in knowledge base."
    context = assemble_context(docs)
    prompt = (
        "You are a helpful data-engineering expert answering only from the Sources below. "
        "If the answer is not present in the sources, say you don't know. "
        "Keep answers concise and reference the source filename.\n\n"
        f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    return asyncio.run(rag_agent.run(prompt))

if __name__ == "__main__":
    q = "Explain what ETL is and give a short example."
    print(answer_with_context(q, k=8))
    