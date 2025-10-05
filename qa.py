from typing import List, Literal

import logging
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.rag import RAG
from app.retriver import MultiModalRetriever
from app.utils.retiver_utils import get_opensearch_retriever, get_chroma_retriever
from app.workflow import Workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    embedding = OllamaEmbeddings(model="embeddinggemma")
    llm = OllamaLLM(model="gpt-oss:20b", temperature=0)

    # Create retrievers
    opensearch_retriever = get_opensearch_retriever()
    chroma_retriever = get_chroma_retriever(embedding, "./chroma_db")

    # Initialize dual-mode retriever
    dual_retriever = MultiModalRetriever(
        opensearch_retriever=opensearch_retriever,
        chroma_retriever=chroma_retriever,
        mode="opensearch"
    )

    rag = RAG(embedding, llm, dual_retriever)

    graph = Workflow(rag).build_graph()

    while True:
        query = input("Start Your Query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        inputs = {"original_query": query}
        for event in graph.stream(inputs):
            for key, value in event.items():
                logger.info(f"Event {key}: {value}")
