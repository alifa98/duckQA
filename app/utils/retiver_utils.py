from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opensearchpy import OpenSearch


def get_opensearch_retriever():
    from app.utils.opensearch_utils import get_opensearch_client

    class OpenSearchBM25Retriever(BaseRetriever):
        """OpenSearch retriever using BM25 keyword search."""

        def __init__(self, client: OpenSearch, index_name: str, k: int = 5):
            super().__init__()
            # self.client: OpenSearch = client
            # self.index_name: str = index_name
            # self.k: int = k

        def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            """Retrieve documents using BM25 keyword matching."""

            # BM25 query using match
            search_body = {"query": {"match": {"content": query}}, "size": self.k}

            response = self.client.search(index=self.index_name, body=search_body)

            documents = []
            for hit in response["hits"]["hits"]:
                documents.append(
                    Document(
                        page_content=hit["_source"][self.text_field],
                        metadata={
                            "_id": hit["_id"],
                            "_score": hit["_score"],
                            "_index": hit["_index"],
                            **hit["_source"],
                        },
                    )
                )

            return documents

    client = get_opensearch_client()
    return OpenSearchBM25Retriever(client, index_name="documents", k=5)


def get_chroma_retriever(embedding_fn, persistent_path):
    from langchain_chroma import Chroma
    from chromadb import PersistentClient

    chroma_client = PersistentClient(path=persistent_path)
    chroma_collection = chroma_client.get_or_create_collection("documents")

    # Create Chroma retriever
    chroma_vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embedding_fn,
        client=chroma_client,
        persist_directory=persistent_path,
    )
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 5})
    return chroma_retriever
