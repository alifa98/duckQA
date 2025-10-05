import operator
from typing import Annotated, TypedDict

from app.utils.query_type import QueryType


class AgentState(TypedDict):
    current_iteration: int
    query_type: QueryType
    original_query: str
    sub_queries: Annotated[list[str], operator.add]
    all_elastic_documents: Annotated[list, operator.add]
    all_chroma_documents: Annotated[list, operator.add]
    selected_documents: Annotated[list, operator.add]
    confidence_score: float
    final_answer: str
