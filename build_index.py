import os
import uuid

# Chroma imports
from chromadb import PersistentClient
from langchain_experimental.text_splitter import SemanticChunker

# LangChain imports
from langchain_ollama import OllamaEmbeddings

from app.utils.files_utils import find_files_with_extension
from app.utils.opensearch_utils import (
    create_index,
    get_opensearch_client,
    index_document,
)

# -------------------
# Make the directory for Chroma if it doesn't exist
# -------------------
persist_dir = "./chroma_db"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# -------------------
# OpenSearch setup
# -------------------
client = get_opensearch_client()
index_name = "documents"

index_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "metadata": {"type": "object"},
        }
    },
}

print(f"Creating index '{index_name}' if it doesn't exist...")
response = create_index(client, index_name, index_body)

# -------------------
# Chroma setup
# -------------------
chroma_client = PersistentClient(path=persist_dir)
chroma_collection = chroma_client.get_or_create_collection("documents")

# Embeddings + semantic chunker
embedding_model = OllamaEmbeddings(model="embeddinggemma")
chunker = SemanticChunker(embedding_model)

# -------------------
# Process documents
# -------------------
# Collect both .md and .txt files for Chroma
files_md = find_files_with_extension("./docs/", ".md")
files_txt = find_files_with_extension("./docs/", ".txt")
all_files = files_md + files_txt

print(f"Found {len(files_md)} markdown files and {len(files_txt)} text files.")

for file_path in all_files:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Generate a stable unique document ID
    doc_id = str(uuid.uuid4())

    # -------------------
    # Index full doc in OpenSearch
    # -------------------
    content = {
        "content": raw_text,
        "metadata": {"path": file_path, "doc_id": doc_id},
    }
    index_document(client, index_name, content)
    print(f"Indexed in OpenSearch: {file_path}")

    # -------------------
    # Index chunks in Chroma (only for .md/.txt)
    # -------------------
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in [".md", ".txt"]:
        print(f"Skipping Chroma for {file_path} (unsupported format)")
        continue

    # Chunk the doc
    chunks = chunker.split_text(raw_text)

    # Add chunks to Chroma
    for i, chunk in enumerate(chunks):
        chroma_id = f"{doc_id}_chunk_{i}"
        chroma_collection.add(
            ids=[chroma_id],
            documents=[chunk],
            embeddings=embedding_model.embed_documents([chunk]),
            metadatas=[
                {
                    "parent_doc_id": doc_id,
                    "chunk_index": i,
                    "path": file_path,
                }
            ],
        )
    print(f"Indexed {len(chunks)} chunks in Chroma for: {file_path}")
