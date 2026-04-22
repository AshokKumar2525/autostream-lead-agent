# 🎬 AutoStream Conversational AI Agent

A production-ready conversational AI agent built for **AutoStream**, a fictional SaaS company offering AI-powered automated video editing tools. The agent uses **LangGraph** for stateful conversation management, **Google Gemini 3.0 Flash** as the LLM backbone, and **FAISS + HuggingFace** for RAG-based knowledge retrieval.

---

## 📋 Table of Contents

- [Setup \& How to Run](#setup--how-to-run)
- [Architecture Explanation](#architecture-explanation)
- [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)
- [Data Privacy Note](#data-privacy-note)

---

## 🚀 Setup & How to Run

### Prerequisites
- Python 3.9 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/autostream-lead-agent.git
   cd autostream-lead-agent
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set the Gemini API key**
   ```bash
   save the gemini api key in env file in root directory as follows:
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run the agent**
   ```bash
   python main.py
   ```

### Sample Interaction
```
🧑 You: Hi there!
🤖 AutoStream: Hello! Welcome to AutoStream! How can I help you today?

🧑 You: What are your pricing plans?
🤖 AutoStream: We have two plans — Basic at $29/month (10 videos, 720p) and Pro at $79/month (unlimited videos, 4K, AI captions).

🧑 You: I want to sign up for the Pro plan
🤖 AutoStream: That's great! I'd love to help you get started. Could you please share your full name?

🧑 You: M.Ashok Kumar
🤖 AutoStream: Thanks, M.Ashok Kumar! Could you please share your email address?

🧑 You: ashok@email.com
🤖 AutoStream: Got it! What platform do you primarily create content for?

🧑 You: YouTube
✅ Lead captured successfully!
🤖 AutoStream: Awesome! Our team will reach out to you shortly. Welcome to AutoStream! 🎬
```

---

## 🏗️ Architecture Explanation

### Why LangGraph?

LangGraph was chosen as the orchestration framework because it provides **stateful, graph-based agent execution** — a critical requirement for multi-turn conversational flows. Unlike simple chain-based approaches in LangChain, LangGraph allows us to define explicit nodes (intent detection, response generation, lead collection) and conditional edges (routing based on detected intent), giving us fine-grained control over conversation flow while maintaining state across turns.

### State Management

The agent state is defined as a `TypedDict` and is passed through the graph on every invocation. It carries the full conversation history (`messages`), the current intent classification, collected lead fields (`lead_name`, `lead_email`, `lead_platform`), and boolean flags (`lead_captured`, `collecting_lead`). The `collecting_lead` flag ensures that once a user shows high intent, subsequent messages continue through the lead collection path without re-classification.

### RAG with FAISS

The knowledge base (`autostream_kb.md`) is loaded, split into chunks using `RecursiveCharacterTextSplitter`, and embedded using the lightweight `all-MiniLM-L6-v2` sentence-transformer model. These embeddings are stored in a FAISS vector store. At query time, the top-2 most relevant chunks are retrieved and injected into the Gemini prompt as context, ensuring responses are grounded in factual product information.

### Intent-Based Routing

After every user message, the `detect_intent` node classifies the intent into one of three categories: `greeting`, `inquiry`, or `high_intent`. Conditional edges in the graph then route to either the `respond` node (for greetings and inquiries) or the `collect_lead` node (for high-intent users). This routing mechanism ensures the agent adapts its behavior dynamically based on user signals.

```
┌──────────────────┐
│   User Message    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  detect_intent   │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌─────────────┐
│ respond│ │ collect_lead│
│ (RAG)  │ │ (CRM flow)  │
└────┬───┘ └──────┬──────┘
     │            │
     ▼            ▼
   [END]        [END]
```

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent to WhatsApp, you can use the **WhatsApp Cloud API** from Meta with a **FastAPI** webhook server.

### Step 1: Set Up WhatsApp Cloud API

1. Create a Meta Developer account at [developers.facebook.com](https://developers.facebook.com/)
2. Create a new app and add the **WhatsApp** product
3. Set up a test phone number in the WhatsApp sandbox
4. Note your **Phone Number ID**, **Access Token**, and **Verify Token**

### Step 2: Create the Webhook Server

```python
from fastapi import FastAPI, Request, Response
import httpx
import os

from agent.retriever import build_retriever
from agent.graph import build_graph

app = FastAPI()

# Initialize the agent
retriever = build_retriever()
graph = build_graph(retriever)

# Store per-user state (in production, use Redis or a database)
user_sessions = {}

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Verify the webhook with Meta's challenge."""
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=params.get("hub.challenge"), status_code=200)
    return Response(status_code=403)


@app.post("/webhook")
async def handle_message(request: Request):
    """Process incoming WhatsApp messages."""
    body = await request.json()

    # Extract the message
    entry = body.get("entry", [{}])[0]
    changes = entry.get("changes", [{}])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [])

    if not messages:
        return Response(status_code=200)

    message = messages[0]
    sender = message["from"]
    text = message.get("text", {}).get("body", "")

    # Get or create user session state
    if sender not in user_sessions:
        user_sessions[sender] = {
            "messages": [], "intent": "",
            "lead_name": None, "lead_email": None,
            "lead_platform": None,
            "lead_captured": False, "collecting_lead": False,
        }

    state = user_sessions[sender]
    state["messages"].append({"role": "user", "content": text})

    # Invoke the agent graph
    state = graph.invoke(state)
    user_sessions[sender] = state

    # Get the assistant's reply
    reply = state["messages"][-1]["content"]

    # Send reply back via WhatsApp API
    await send_whatsapp_message(sender, reply)

    return Response(status_code=200)


async def send_whatsapp_message(to: str, text: str):
    """Send a message back to the user via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

### Step 3: Deploy and Configure

1. Deploy the FastAPI app to a public server (e.g., Heroku, AWS, Railway)
2. Set the webhook URL in Meta Developer Console to `https://yourdomain.com/webhook`
3. Subscribe to the `messages` webhook field
4. Set environment variables: `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `VERIFY_TOKEN`, `GEMINI_API_KEY`

---

## 🔒 Data Privacy Note

### User Consent

This agent collects personal information (name, email, preferred platform) **only after the user has expressed explicit intent** to sign up or learn more. The agent clearly communicates what information is being collected and why before asking for each field. Users can exit the conversation at any time without providing data.

### Regulatory Compliance

- **DPDP Act (India):** The Digital Personal Data Protection Act, 2023 mandates that personal data must be processed only for a lawful purpose with the individual's consent. This agent adheres to these principles by collecting only the minimum necessary data and only upon explicit user interest. In a production deployment, a formal consent notice should be presented before data collection begins.

- **GDPR (EU):** For users in the European Union, the General Data Protection Regulation requires transparent data processing, purpose limitation, and data minimization. The agent's design aligns with these requirements. In production, the following would be added:
  - A clear privacy notice before data collection
  - Right to erasure (ability to delete collected data)
  - Data processing agreements with third-party services
  - Data retention policies

### Production Recommendations

1. Store collected leads in an encrypted database
2. Implement explicit opt-in consent flows
3. Provide users with data access and deletion requests
4. Conduct periodic Data Protection Impact Assessments (DPIA)
5. Log all data processing activities for audit compliance

---

## 📁 Project Structure

```
autostream-lead-agent/
├── knowledge_base/
│   └── autostream_kb.md        # Product knowledge base (pricing, features, policies)
├── agent/
│   ├── __init__.py             # Package initialization
│   ├── state.py                # AgentState TypedDict definition
│   ├── retriever.py            # FAISS + HuggingFace retriever setup
│   ├── nodes.py                # LangGraph node functions (intent, respond, lead)
│   ├── tools.py                # Mock lead capture tool
│   └── graph.py                # LangGraph state graph builder
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 3.0 Flash |
| Agent Framework | LangGraph |
| RAG Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Language | Python 3.9+ |
| Interface | CLI (Terminal) |


