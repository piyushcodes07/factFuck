from langgraph.graph.state import END, StateGraph

from claim_identifier import claim_identifier
from deep_cross_reference import deep_cross_reference
from extraction_agent import extraction_agent
from schemas import State

builder = StateGraph(State)

builder.set_entry_point("extract")
builder.add_node("extract", extraction_agent)
builder.add_node("claim_identifier", claim_identifier)

# FIX: prevent_default_writes must be enabled
builder.add_node("deep_cross_reference", deep_cross_reference)

builder.add_edge("extract", "claim_identifier")
builder.add_edge("claim_identifier", "deep_cross_reference")
builder.add_edge("deep_cross_reference", END)

graph = builder.compile()
