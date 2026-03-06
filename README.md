# ⚡ Agentic RAG System

An **Agentic Retrieval-Augmented Generation (RAG) system** built with **LangGraph and LLM agents** that dynamically retrieves, evaluates, and generates responses using a multi-step reasoning workflow.

Unlike traditional RAG pipelines, this project introduces **AI agents that decide when to retrieve, verify, or refine answers**, improving accuracy and reliability.

---

# 🧠 Architecture

```
User Query
    |
    ▼
Router Agent
    |
    ├── Retrieve Documents
    │
    ▼
Retriever Agent
    |
    ▼
Document Evaluation
    |
    ├── Relevant → Pass to Generator
    └── Not Relevant → Re-retrieve
    |
    ▼
Generator Agent
    |
    ▼
Answer Evaluation
    |
    ├── Good Answer → Return to User
    └── Poor Answer → Retry / Improve
```

---

# ✨ Key Features

* **Agentic Workflow** — AI agents decide how to retrieve and generate responses.
* **Dynamic Retrieval** — Automatically retrieves the most relevant documents.
* **Self-Correction Loop** — Improves responses through evaluation and retries.
* **Multi-Agent Collaboration** — Router, Retriever, Generator, and Evaluator agents.
* **LangGraph Workflow** — Structured graph-based agent orchestration.
* **Dual Interface** — CLI and optional Streamlit UI.

---

# 📂 Project Structure

```
Agentic_RAG/
│
├── .env                     # API keys
├── requirements.txt         # Dependencies
├── main.py                  # CLI entry point
├── app.py                   # Streamlit UI
│
├── data/                    # Source documents
│
├── vectorstore/
│   └── store.py             # Vector DB setup
│
├── agents/
│   ├── router_agent.py
│   ├── retriever_agent.py
│   ├── generator_agent.py
│   └── evaluator_agent.py
│
├── graph/
│   └── workflow.py          # LangGraph workflow
│
└── utils/
    └── helpers.py
```

---

# 🚀 Getting Started

## 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Agentic_RAG.git
cd Agentic_RAG
```

---

## 2️⃣ Create & activate virtual environment

```bash
python -m venv venv
```

### Windows

```bash
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Set up API Keys

Create a `.env` file in the project root.

```
OPENAI_API_KEY=your_api_key
```

or

```
GROQ_API_KEY=your_groq_api_key
```

---

## 5️⃣ Run the application

### CLI Mode

```
python main.py
```

### Streamlit UI

```
streamlit run app.py
```

---

# 🔄 Workflow

1. User asks a question.
2. Router agent determines the required action.
3. Retriever agent fetches relevant documents.
4. Generator agent creates the response.
5. Evaluator agent checks response quality.
6. If needed, the system retries or refines the answer.

---

# 🛠 Tech Stack

* Python
* LangChain
* LangGraph
* Vector Database (FAISS / Chroma)
* LLM APIs (Groq / OpenAI)
* Streamlit

---

# 📊 Example Use Cases

* AI document assistants
* Research paper analysis
* Enterprise knowledge base
* Customer support automation

---

# 📜 License

MIT License
