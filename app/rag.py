from langchain.output_parsers import EnumOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from .utils.query_type import QueryType
from .utils.state import AgentState


class RAG:
    def __init__(
        self,
        embedding_fn,
        llm,
        retriever: BaseRetriever,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.llm = llm
        self.retriever = retriever

    def query_classifier(self, state: AgentState) -> dict:
        output_parser = EnumOutputParser(enum=QueryType)
        prompt = PromptTemplate.from_template(
            """Classify the following query into the categories that are defined in the Instructions section.
            You must choose one of the categories that fits the query the best.
            
            Query: {query}
            
            Instructions: {instructions}
            """
        ).partial(instructions=output_parser.get_format_instructions)

        chain = prompt | self.llm | output_parser

        result = chain.invoke({"query": state["original_query"]})

        return {"query_type": result}
