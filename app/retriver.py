from typing import Any, Dict, List, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MultiModalRetriever(BaseRetriever):
    """Custom retriever with OpenSearch and Chroma DB modes."""

    opensearch_retriever: BaseRetriever
    chroma_retriever: BaseRetriever
    mode: Literal["opensearch", "chroma"] = "opensearch"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents based on the selected mode."""
        if self.mode == "opensearch":
            return self.opensearch_retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
        elif self.mode == "chroma":
            return self.chroma_retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async implementation for better performance."""
        if self.mode == "opensearch":
            return await self.opensearch_retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
        elif self.mode == "chroma":
            return await self.chroma_retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
