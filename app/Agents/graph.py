from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.Agents.Nodes.planner import planner_node
from app.Agents.Nodes.responder import generate_node
from app.Agents.Nodes.retrieval import retrieve_node
from app.Agents.state import AgentState


def _route_after_planning(state: AgentState) -> Literal["retrieve", "respond"]:
    if state["current_query"] == "CONVERSATIONAL":
        return "respond"
    return "retrieve"


workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("respond", generate_node)
workflow.add_edge(START, "planner")
workflow.add_conditional_edges(
    "planner",
    _route_after_planning,
    {"retrieve": "retrieve", "respond": "respond"},
)
workflow.add_edge("retrieve", "respond")
workflow.add_edge("respond", END)

rag_agent = workflow.compile(checkpointer=MemorySaver())
