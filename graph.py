from langgraph.graph.state import END, StateGraph

from extraction_agent import extraction_agent
from schemas import State

builder = StateGraph(State)
builder.add_node("extract", extraction_agent)
builder.set_entry_point("extract")
builder.add_edge("extract", END)
graph = builder.compile()
