"""
graph.py - LangGraph State Graph Builder

Constructs the conversational agent's state graph with three nodes:
  1. detect_intent — entry point, classifies user intent
  2. respond — generates RAG-augmented response for greetings/inquiries
  3. collect_lead — collects lead info for high-intent users

Routing:
  - detect_intent → respond (if greeting or inquiry)
  - detect_intent → collect_lead (if high_intent)
  - respond → END
  - collect_lead → END
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import detect_intent, respond, collect_lead


def build_graph(retriever):
    """
    Build and compile the LangGraph state graph for the agent.

    The graph flow:
        START → detect_intent → (routing) → respond OR collect_lead → END

    Args:
        retriever: The FAISS retriever for knowledge base queries,
                   passed to the respond node.

    Returns:
        A compiled LangGraph graph ready for invocation.
    """
    # Create the state graph with AgentState schema
    graph = StateGraph(AgentState)

    # --- Add Nodes ---

    # Node 1: Intent detection (entry point)
    graph.add_node("detect_intent", detect_intent)

    # Node 2: RAG-augmented response generation
    # Wrap the respond function to pass the retriever
    graph.add_node("respond", lambda state: respond(state, retriever))

    # Node 3: Lead information collection
    graph.add_node("collect_lead", collect_lead)

    # --- Set Entry Point ---
    graph.set_entry_point("detect_intent")

    # --- Add Conditional Routing ---
    # Route based on the detected intent after the detect_intent node
    def route_by_intent(state: dict) -> str:
        """Route to the appropriate node based on detected intent."""
        intent = state.get("intent", "inquiry")
        if intent == "high_intent":
            return "collect_lead"
        else:
            # Both "greeting" and "inquiry" go to respond
            return "respond"

    graph.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "collect_lead": "collect_lead",
            "respond": "respond"
        }
    )

    # --- Add Terminal Edges ---
    # Both respond and collect_lead terminate the current graph run
    graph.add_edge("respond", END)
    graph.add_edge("collect_lead", END)

    # Compile and return the graph
    compiled_graph = graph.compile()
    return compiled_graph
