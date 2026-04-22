"""
state.py - Agent State Definition

Defines the AgentState TypedDict used by all LangGraph nodes.
This state is passed through the graph and updated at each node.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    State schema for the AutoStream conversational agent.

    Fields:
        messages: Full conversation history as a list of dicts
                  with 'role' ('user' or 'assistant') and 'content'.
        intent: Detected intent of the current user message.
                One of: "greeting", "inquiry", "high_intent".
        lead_name: Collected lead name (None if not yet collected).
        lead_email: Collected lead email (None if not yet collected).
        lead_platform: Collected lead platform (None if not yet collected).
        lead_captured: Whether the lead has been successfully captured.
        collecting_lead: True once high_intent is detected; remains True
                         for all subsequent turns to continue lead collection.
    """
    messages: list
    intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool
