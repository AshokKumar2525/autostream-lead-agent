"""
main.py - AutoStream Conversational AI Agent Entry Point

Initializes the RAG retriever and LangGraph agent, then runs an
interactive CLI loop where the user can converse with the agent.
The loop continues until lead information is captured or the user exits.
"""

import os
import sys
import warnings
import logging

# ── Suppress ALL noisy warnings BEFORE any library imports ──
# These env vars must be set before transformers/HF libraries load
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Suppress Python-level warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Validate that the Gemini API key is set
if not os.getenv("GEMINI_API_KEY"):
    print("=" * 55)
    print("  ❌  ERROR: GEMINI_API_KEY environment variable not set!")
    print("  Please set it before running by saving the key in env file in root directory as shown below:")
    print('    GEMINI_API_KEY=your_api_key_here')
    print("=" * 55)
    sys.exit(1)

from agent.retriever import build_retriever
from agent.graph import build_graph


def main():
    """
    Main function to run the AutoStream conversational agent.

    Steps:
        1. Build the FAISS retriever from the knowledge base.
        2. Build and compile the LangGraph state graph.
        3. Initialize an empty agent state.
        4. Run an interactive loop:
           - Get user input
           - Append to conversation history
           - Invoke the graph
           - Print the assistant's response
        5. Exit when lead is captured or user presses Ctrl+C.
    """
    print("\n" + "=" * 55)
    print("  🎬  AutoStream Conversational AI Agent")
    print("  Powered by Google Gemini 3.0 Flash + LangGraph")
    print("=" * 55)
    print("\n📦 Loading knowledge base and building retriever...")

    # Step 1: Build the RAG retriever
    retriever = build_retriever()
    print("✅ Retriever ready!")

    # Step 2: Build the agent graph
    print("🔧 Building agent graph...")
    graph = build_graph(retriever)
    print("✅ Agent graph compiled!")

    # Step 3: Initialize empty agent state
    state = {
        "messages": [],
        "intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False
    }

    print("\n💬 Chat with AutoStream! Type 'quit' or 'exit' to leave.\n")
    print("-" * 55)

    # Step 4: Interactive conversation loop
    try:
        while not state.get("lead_captured", False):
            # Get user input
            user_input = input("\n🧑 You: ").strip()

            # Handle exit commands
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                print("\n👋 Thanks for chatting with AutoStream! Goodbye.\n")
                break

            # Skip empty input
            if not user_input:
                print("   (Please type a message)")
                continue

            # Append user message to state
            state["messages"].append({
                "role": "user",
                "content": user_input
            })

            # Invoke the graph with the current state
            state = graph.invoke(state)

            # Print the last assistant message
            if state["messages"]:
                last_msg = state["messages"][-1]
                if last_msg["role"] == "assistant":
                    print(f"\n🤖 AutoStream: {last_msg['content']}")

        # Check if lead was captured
        if state.get("lead_captured", False):
            print("\n" + "=" * 55)
            print("  🎉 Lead collection complete! Thank you.")
            print("=" * 55 + "\n")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\n👋 Session interrupted. Thanks for chatting with AutoStream!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
