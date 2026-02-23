from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

import requests

import chromadb
from chromadb.config import Settings
from chromadb.api.types import OneOrMany, Embedding
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

import json

from typing import Any, Sequence

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "qwen2.5:7b"
MAX_ITERATIONS = 6


with open('tools.json') as file:
    TOOLS = json.load(file)


SYSTEM_PROMPT = """You are an expert scientific research assistant with access to
a curated ChromaDB knowledge base of scientific papers.

Behaviour rules:
1. Reason briefly before deciding whether retrieval is necessary.
2. Retrieve documents only when the answer genuinely requires content from the
   papers (specific findings, methods, data, citations).
3. You may call retrieve_documents more than once with different queries if the
   first retrieval was insufficient.
4. When synthesising retrieved passages, always cite the source filename.
5. If the retrieved content is not relevant, say so rather than hallucinating.
6. For conversational or general-knowledge questions, answer directly without
   retrieval.
"""


def search_web(inputs: dict) -> str:
    r = requests.get(f"https://duckduckgo.com/?q={inputs['term']}&ia=web")
    if r.status_code == 200:
        return r.text
    return "Could'nt find anything"


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

    def search(self, inputs: dict) -> str:

        query, top_k = inputs['query'], inputs.get('top_k', 5)
        query = query if isinstance(query, list) else [query]
        embedding: OneOrMany[Embedding] = self.embedder(query)
        n_results: int = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results
        )

        hits = [
            {
                "text": doc,
                "source": meta.get("filename", "unknown"),
                "chunk": meta.get("chunk_index"),
                "similarity": round(1 - dist, 4),
            } for (m, d, di) in zip(results["metadatas"], # type: ignore
                                    results["documents"], # type: ignore
                                    results["distances"]) # type: ignore
                for (meta, doc, dist) in zip(m, d, di)
        ]
        
        if not hits:
            return "No relevant documents found."
        lines = [f"[Retrieved {len(hits)} chunk(s) for query: '{inputs['query']}']"]
        for h in hits:
            lines.append(
                f"\n--- source: {h['source']} "
                f"| similarity: {h['similarity']} ---\n{h['text']}"
            )
        return "\n".join(lines)


class RAGAgent:
    def __init__(self,
                 collection: str,
                 tools_description: Sequence[dict[str, Any]]):
        self.retriever = Retriever(collection)
        self.client = ChatOllama(model=EMBED_MODEL).bind_tools(tools_description)
        self.history: list[dict] = []

    def _run_tool(self, name: str, inputs: dict) -> str:
        if name == "retrieve_documents":
            return self.retriever.search(inputs)
            
        if name == "search_web":
            return search_web(inputs)
        return f"Unknown tool: {name}"

    def ask(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = list(self.history)

        for _ in range(MAX_ITERATIONS):
            lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"]) if m['role'] == 'assistant'
                else ToolMessage(content=m['content'], tool_call_id=m['tool_call_id'])
                for m in messages
            ]
            response = self.client.invoke(lc_messages)

            assistant_content = response.content
            messages.append(
                {"role": "assistant", "content": assistant_content})
            
            if not response.tool_calls: # pyright: ignore[reportAttributeAccessIssue]
                self.history.append(
                    {"role": "assistant", "content": assistant_content})
                return assistant_content # pyright: ignore[reportReturnType]


            for block in response.tool_calls: # pyright: ignore[reportAttributeAccessIssue]
                if block['type'] != "tool_call":
                    continue
                print(
                    f"  [Agent → tool] {block['name']}({json.dumps(block['args'])})")
                result_text = self._run_tool(block['name'], block['args'])

                messages.append(
                    {"role": "tool_call", "content": result_text, "tool_call_id": block['id']})

        return "Max iterations reached without a final answer."

    def chat(self):
        print("\n🔬  Scientific RAG Agent ready. Type 'quit' to exit.\n")
        red = "\033[31m"
        green = "\033[32m"
        end = "\033[0m"
        while True:
            try:
                user_input = input(f"{green}You{end}: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                continue
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            answer = self.ask(user_input)
            print(f"\n{red}Assistant{end}: {answer}\n")


if __name__ == "__main__":

    agent = RAGAgent(collection='research_docs', tools_description=TOOLS)
    agent.chat()
