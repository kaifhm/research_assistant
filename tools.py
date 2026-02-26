import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from langchain.tools import tool

from dotenv import load_dotenv
import os

load_dotenv()


CHROMA_PATH = os.environ['CHROMA_PATH']
COLLECTION = os.environ['COLLECTION']

### Define tools

class Retriever:
    def __init__(self, collection_name: str):
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name)
        print(f"[Retriever] Connected to '{collection_name}' "
              f"({self.collection.count()} chunks)")
        self.embedder = ONNXMiniLM_L6_V2()

    def search(self, query: str | list[str], top_k: int) -> str:

        query = query if isinstance(query, list) else [query]
        embedding = self.embedder(query)
        results = self.collection.query(
            query_embeddings=embedding,  # type: ignore
            n_results=min(top_k, self.collection.count()),  # type: ignore
            include=["documents", "metadatas", "distances"],  # type: ignore
        )

        hits = [
            {
                "text": doc,
                "source": meta.get("filename", "unknown"),
                "chunk": meta.get("chunk_index"),
                "similarity": round(1 - dist, 4),   # cosine similarity
            } for (d, m, di) in zip(results["documents"], # type: ignore
                                    results["metadatas"], # type: ignore
                                    results["distances"]) # type: ignore
                for (doc, meta, dist) in zip(d, m, di)
        ]
        
        if not hits:
            return "No relevant documents found."
        lines = [f"[Retrieved {len(hits)} chunk(s) for query: '{query}']"]
        for h in hits:
            lines.append(
                f"\n--- source: {h['source']} "
                f"| similarity: {h['similarity']} ---\n{h['text']}"
            )
        return "\n".join(lines)


def make_retriever():
    retriever = Retriever(COLLECTION)
    @tool
    def retrieve_documents(query: str | list[str], top_k: int = 5):
        "Search the scientific literature stored in ChromaDB. Call this when "
        "the user's question requires factual details, citations, methods, or "
        "findings from the stored papers. Change key verbs in query into their synonyms "
        "if you do not find results originally. Do NOT call it for greetings, "
        "meta-questions, or things you can answer with certainty from general knowledge."
        
        query = query if isinstance(query, list) else [query]
        messages = retriever.search(query, top_k=top_k)
        return messages

    return retrieve_documents

doc_search = make_retriever()

# ============= ALL TOOLS LIST =============

TOOLS = [
    doc_search,
    # search_web,
]