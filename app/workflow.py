from sre_parse import State

from langgraph.graph import END, START, StateGraph

from app.utils.state import AgentState


class Workflow:
    def __init__(self, rag):
        self.rag = rag

    def build_graph(self):
        """Build the LangGraph workflow here"""

        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("query_classifier", self.rag.query_classifier)

        # Entry point
        workflow.set_entry_point("query_classifier")

        # Edges
        workflow.add_edge("query_classifier", END)

        return workflow.compile()
