from typing import Annotated, TypedDict
import operator


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    current_query: str
    documents: list[str]
    plan: list[str]
    status: str
    final_answer: str