# LLMChat | A frontend for Ollama with tool calling

A fully local **Retrieval-Augmented Generation (RAG)** research assistant with **tool calling** support. Ask questions about your documents and get intelligent, context-aware answers — all running on your own machine, with no data sent to the cloud.

---

## Features

- **Local RAG pipeline** — ingest and query your own documents privately
- **Tool calling** — the agent can invoke tools to enhance retrieval and reasoning
- **Flask web interface** — clean, browser-based UI for interacting with the assistant
- **Fully local** — your data never leaves your machine
- **Python-powered** — built with a modern Python backend

---

## Project Structure

```
research_assistant/
├── flask_app/        # Web application (Flask server, HTML/CSS/JS frontend)
└── rag_agent/        # Core RAG agent logic and tool-calling implementation
```

---

## Getting Started
### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/kaifhm/research_assistant.git
   cd research_assistant
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

    You may need to install SQL drivers for PostgreSQL or MySQL if using those, separately.

### Running the App

To run the app, you need two things:
1. An SQL database.
2. A `.env` file with the following variables set.
   - `CHROMA_PATH`: Path to the ChromaDB. DB gets created if not present.
   - `CHUNK_SIZE`: Document chunk size when ingesting into ChromaDB
   - `CHUNK_OVERLAP`: Overlap size between two successive chunks.
   - `COLLECTION`: The collection name in ChromaDB.
   - `SYSTEM_PROMPT`: System prompt for the the LLM.
   - `LOGGING_LEVEL`: Defaults to `INFO`.
   - `SQLALCHEMY_DATABASE_URI`: URL to the SQL database.

```bash
# make sure SQL database is running if you're using MySQL or PostgreSQL
flask --app flask_app run
```

Then open your browser and navigate to `http://localhost:5000`.

#### Alternatively
You can also run the app on your terminal by running,

```bash
python -m rag_agent
```

---
## VectorDB ingestion

To add files to the vector db, run

```bash
python rag_agent/database.py --add-files path/to/file.txt.
``` 
Only .txt, .md, and .pdb files are supported as of now.

For more information, run

```bash
python rag_agent/database.py -h
```

## Defining new tools

You can define new tools for the LLM writing new functions in the `rag_agent.tools.py` file and adding the function object to `TOOLS` list. There are already examples in the file on how to write a tool function. Checkout [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools) for more information. You will have to restart the server if to see the updates.

---

## How It Works

1. **Ingestion** — Documents are loaded, chunked, and embedded into a local vector store.
2. **Retrieval** — When a query is received, the most relevant chunks are retrieved via semantic search.
3. **Tool Calling** — The agent can call tools (e.g., lookup, summarize) to augment its response.
4. **Generation** — The LLM uses the retrieved context and tool outputs to generate a grounded answer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, SQLAlchemy |
| Frontend | HTML, CSS, JavaScript |
| RAG / Agent | LangChain, Ollama |
| Vector Store | ChromaDB |

---

## Author

**kaifhm** — [GitHub](https://github.com/kaifhm)