import argparse
import hashlib
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter


CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
COLLECTION = 'research_docs'

def get_chroma_collection(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def doc_id(file_path: Path, chunk_index: int) -> str:
    """Stable, unique ID for a chunk."""
    h = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    return f"{file_path.stem}_{h}_chunk{chunk_index:04d}"


def ingest_file(file_path: Path, collection: chromadb.Collection) -> int:
    """Chunk, embed, and upsert one document. Returns number of chunks stored."""
    print(f"  ↳ {file_path.name}", end=" ", flush=True)

    doc_loader: Dict[str, type[TextLoader] | type[PDFPlumberLoader]] = {'.txt': TextLoader,
                                                                        '.md': TextLoader,
                                                                        '.pdf': PDFPlumberLoader}

    try:
        documents = doc_loader[file_path.suffix](file_path).load()
    except KeyError:
        print(
            f"Invalid path. Expected file of type: {tuple(doc_loader.keys())}. Got {file_path.suffix}")

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_documents = text_splitter.split_documents(documents)

    ids = [doc_id(file_path, i) for i in range(len(chunked_documents))]

    total_chunked_documents = len(chunked_documents)
    metadata: List[Dict] = [
        {
            "source":      str(file_path),
            "filename":    file_path.name,
            "chunk_index": i,
            "page": doc.metadata.get('page', 0),
            "total_chunks": total_chunked_documents,
        }
        for i, doc in enumerate(chunked_documents)
    ]

    collection.upsert(
        ids=ids,
        documents=[x.page_content for x in chunked_documents],  # type: ignore
        metadatas=metadata,  # type: ignore
    )
    print(f"→ {len(chunked_documents)} chunks")
    return len(chunked_documents)


def ingest_files(files: List[str], collection_name: str) -> None:

    filepaths = list(map(Path, files))

    collection = get_chroma_collection(collection_name)

    total_chunks = 0
    print(
        f"\nIngesting {len(filepaths)} file(s) into collection '{collection_name}':")
    for f in filepaths:
        total_chunks += ingest_file(f, collection)

    print(f"\nDone. {total_chunks} chunks stored in '{CHROMA_PATH}'.")


def view_files():
    collection = get_chroma_collection(COLLECTION)
    count = collection.count()
    limit = 10
    names = set()
    for offset in range(0, count, limit):
        items = collection.get(include=["metadatas"], limit=limit, offset=offset)
        names |= set(item['filename'] for item in items['metadatas'])
    print("Included files in DB:\n---------------------", *sorted(names), sep='\n')


def delete_files(files: Path | str):
    collection = get_chroma_collection(COLLECTION)
    collection.delete(where={"filename": {"$in": files}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="""
database.py - Load, chunk, embed, and store scientific documents into ChromaDB.

Usage:
    python database.py [--add-files file1.pdf [file2.txt, ] | --remove-files file1.pdf [file2.txt, ] | --view-files]
""",
        description="Interact with the ChromaDB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--add-files", nargs='+',
                        help="Folder of documents")
    group.add_argument("--remove-files", nargs='+')
    group.add_argument("--view-files", action='store_true')


    args = parser.parse_args()
    if args.add_files:
        ingest_files(args.add_files, COLLECTION)
    elif args.remove_files:
        pass
    elif args.view_files:
        view_files()
