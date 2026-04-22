"""
nodes.py - LangGraph Node Functions

Contains the three core nodes of the AutoStream agent:
  1. detect_intent - Classifies user intent via Gemini
  2. respond - Generates RAG-augmented responses via Gemini
  3. collect_lead - Collects lead info step by step
"""

import os
import re
import time
from google import genai

# Initialize the Gemini client with API key from environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Model name — Gemini 3.0 Flash (as per assignment requirement)
MODEL_NAME = "gemini-flash-latest"


def _call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"   ⏳ Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def detect_intent(state: dict) -> dict:
    """
    Node 1: Detect user intent from the latest message.

    If lead collection is already in progress (collecting_lead=True),
    the intent is kept as "high_intent" without re-classification.

    Otherwise, uses Gemini to classify the intent as one of:
      - "greeting": A casual hello or greeting
      - "inquiry": A question about AutoStream (pricing, features, etc.)
      - "high_intent": User shows interest in signing up, buying,
                       starting a trial, or getting a demo

    Args:
        state: The current agent state dictionary.

    Returns:
        Updated state with the 'intent' field set.
    """
    # If already collecting lead info, skip classification
    if state.get("collecting_lead", False):
        state["intent"] = "high_intent"
        return state

    # Get the last user message
    last_message = state["messages"][-1]["content"]

    # Build a strict classification prompt
    classification_prompt = f"""You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the following user message into EXACTLY one of these three categories:
- "greeting" — if the user is saying hello, hi, hey, good morning, or any casual greeting
- "inquiry" — if the user is asking a question about AutoStream's features, pricing, plans, policies, support, or capabilities
- "high_intent" — if the user shows intent to sign up, subscribe, purchase, start a free trial, get a demo, or is ready to buy

IMPORTANT RULES:
1. Respond with ONLY the single word: greeting, inquiry, or high_intent
2. Do NOT include quotes, punctuation, or any other text
3. If unsure between inquiry and high_intent, choose inquiry unless they explicitly mention wanting to sign up, buy, subscribe, or try

User message: "{last_message}"

Your classification:"""

    # Call Gemini for classification
    raw_intent = _call_gemini(classification_prompt).lower()

    # Clean the response to extract only the valid intent
    # Remove any quotes, periods, or extra whitespace
    raw_intent = re.sub(r'[^a-z_]', '', raw_intent)

    # Validate and assign the intent
    valid_intents = {"greeting", "inquiry", "high_intent"}
    if raw_intent in valid_intents:
        state["intent"] = raw_intent
    else:
        # Default to inquiry if classification is unclear
        state["intent"] = "inquiry"

    return state


def respond(state: dict, retriever) -> dict:
    """
    Node 2: Generate a RAG-augmented response using Gemini.

    Retrieves relevant context from the FAISS knowledge base,
    builds a prompt with conversation history, and generates
    a helpful assistant response.

    Args:
        state: The current agent state dictionary.
        retriever: The FAISS retriever for knowledge base lookup.

    Returns:
        Updated state with the assistant's response appended to messages.
    """
    # Get the last user message for retrieval
    last_message = state["messages"][-1]["content"]

    # Retrieve relevant context from the knowledge base
    retrieved_docs = retriever.invoke(last_message)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build the conversation history string
    conversation_history = ""
    for msg in state["messages"]:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history += f"{role}: {msg['content']}\n"

    # Build the full prompt for Gemini
    response_prompt = f"""You are a friendly and professional customer support assistant for AutoStream, 
a SaaS platform that provides AI-powered automated video editing tools.

Your job is to answer user questions accurately based on the provided context.
If the user's question is a greeting, respond warmly and offer to help.
If the information is not in the context, say you don't have that information 
and suggest contacting support.

IMPORTANT RULES:
1. Be concise but helpful
2. Only use information from the provided context - do NOT make up features or pricing
3. If the user seems interested in purchasing, mention the available plans and encourage them
4. Always maintain a friendly, professional tone

--- KNOWLEDGE BASE CONTEXT ---
{context}

--- CONVERSATION HISTORY ---
{conversation_history}

Generate a helpful response to the user's latest message. Do NOT prefix your response with "Assistant:" or any role label."""

    # Call Gemini to generate the response
    assistant_reply = _call_gemini(response_prompt)

    # Append the assistant's reply to the conversation history
    state["messages"].append({
        "role": "assistant",
        "content": assistant_reply
    })

    return state


def collect_lead(state: dict) -> dict:
    """
    Node 3: Collect lead information step by step.

    On first entry (high_intent just detected, collecting_lead was False):
      - Sets collecting_lead = True
      - Asks for the user's name

    On subsequent entries:
      - Extracts the value from the last user message
      - Saves it to the appropriate state field
      - Asks for the next missing field

    Collection order: name → email → platform

    Once all 3 fields are collected:
      - Calls mock_lead_capture()
      - Sets lead_captured = True

    Args:
        state: The current agent state dictionary.

    Returns:
        Updated state with lead info and assistant message appended.
    """
    from agent.tools import mock_lead_capture

    last_message = state["messages"][-1]["content"]

    # First entry: high_intent was just detected
    if not state.get("collecting_lead", False):
        state["collecting_lead"] = True
        assistant_reply = (
            "That's great to hear you're interested in AutoStream! 🎉 "
            "I'd love to help you get started. Let me collect a few details.\n\n"
            "Could you please share your **full name**?"
        )
        state["messages"].append({
            "role": "assistant",
            "content": assistant_reply
        })
        return state

    # Subsequent entries: extract info from last user message
    # Check fields in order: name → email → platform
    if state.get("lead_name") is None:
        # Extract name from the last message
        extracted_name = _extract_field_with_gemini(last_message, "name")
        state["lead_name"] = extracted_name

        # Ask for email next
        assistant_reply = (
            f"Thanks, {extracted_name}! 😊\n\n"
            "Could you please share your **email address** so we can send you "
            "the setup details?"
        )

    elif state.get("lead_email") is None:
        # Extract email from the last message
        extracted_email = _extract_field_with_gemini(last_message, "email")
        state["lead_email"] = extracted_email

        # Ask for platform next
        assistant_reply = (
            f"Got it! 📧\n\n"
            "One last question — what **platform** do you primarily create content for? "
            "(e.g., YouTube, Instagram, TikTok, etc.)"
        )

    elif state.get("lead_platform") is None:
        # Extract platform from the last message
        extracted_platform = _extract_field_with_gemini(last_message, "platform")
        state["lead_platform"] = extracted_platform

        # All fields collected — capture the lead
        capture_result = mock_lead_capture(
            name=state["lead_name"],
            email=state["lead_email"],
            platform=state["lead_platform"]
        )

        state["lead_captured"] = True

        assistant_reply = (
            f"Awesome! 🚀 I've got everything I need.\n\n"
            f"Here's a summary of your details:\n"
            f"- **Name:** {state['lead_name']}\n"
            f"- **Email:** {state['lead_email']}\n"
            f"- **Platform:** {state['lead_platform']}\n\n"
            f"Our team will reach out to you shortly with next steps. "
            f"Welcome to AutoStream! 🎬"
        )

    else:
        # Edge case: all fields already collected but lead_captured not set
        assistant_reply = "It looks like we already have your details. Our team will be in touch soon!"

    state["messages"].append({
        "role": "assistant",
        "content": assistant_reply
    })

    return state


def _extract_field_with_gemini(user_message: str, field_type: str) -> str:
    """
    Use Gemini to extract a specific field value from the user's message.

    Args:
        user_message: The raw user message to extract from.
        field_type: One of "name", "email", or "platform".

    Returns:
        The extracted field value as a clean string.
    """
    extraction_prompt = f"""Extract the {field_type} from the following user message.

RULES:
1. Return ONLY the extracted {field_type} value, nothing else
2. Do NOT add any explanation, quotes, or extra text
3. If the message IS the {field_type} itself, return it as-is
4. Clean up any unnecessary whitespace

User message: "{user_message}"

Extracted {field_type}:"""

    extracted = _call_gemini(extraction_prompt)

    # Remove any surrounding quotes
    extracted = extracted.strip('"').strip("'")

    return extracted
