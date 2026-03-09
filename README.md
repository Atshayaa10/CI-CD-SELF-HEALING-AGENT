# 🏆 Opalite OS — Autonomous CI/CD Self-Healing Engine

<div align="center">
  <h2>🌟 Top 10 Finalist (Out of 75 Teams) 🌟</h2>
  <p><i>An enterprise-grade, multi-agent AI orchestration engine that doesn't just detect broken CI/CD pipelines—it <b>heals</b> them seamlessly.</i></p>
</div>

---

## 🚀 The Vision: Zero-Touch Remediation

Continuous Integration (CI) is broken. We automated deployment but left debugging entirely manual, costing enterprises billions in lost developer velocity. 

**Opalite OS** intercepts failed GitHub pipelines and deploys a synchronized swarm of **seven autonomous AI agents**. These agents ingest the repository, diagnose the mathematical fault, cross-reference previous organizational fixes (RAG), synthesize a code patch, prove it works in an isolated Sandbox environment, and deploy a ready-to-merge Pull Request.

**All in under 60 seconds. Zero human intervention.**

---

## 🧠 The 7-Agent LangGraph Swarm Architecture

Unlike basic LLM wrappers that hallucinate code, Opalite operates on a strictly serialized **LangGraph state-machine**, forcing the AI to prove its work deterministically before touching production infrastructure.

1. **Scanner Agent:** Employs intelligent semantic-truncation to compress massive legacy codebases (like 3,000+ line Python files) into high-speed API context windows without crashing.
2. **Diagnostician Agent:** Parses the Abstract Syntax Tree (AST) to isolate mathematical faults, streaming deterministic JSON payloads tracing syntax errors, missing dependencies, or logic limits (e.g., div-by-zero bounds).
3. **Memory Crystal Agent (RAG):** Opalite isn't amnesiac. It queries a locally hosted Vector Database for Retrieval-Augmented Generation, referencing how the organization historically solved topologically similar bugs.
4. **Solver Agent:** Powered by **Groq LPU (Llama-3.3-70b)**, it synthesizes a precise syntactic patch to address all anomalies simultaneously.
5. **Verifier Agent (The Sandbox):** *The ultimate defensive moat.* The Verifier spins up an ephemeral, containerized Sandbox environment, injects the Solver's AI patch, and empirically executes the integration test suite (`pytest`) locally.
6. **Critic Agent:** Scrutinizes algorithmic fixes against the deterministic test results. Only an absolute 'PASSED' consensus allows the graph to advance, eliminating blind hallucinations.
7. **Deployer Agent:** Bundles the diffs, signs a secure branch, and injects a fully documented Pull Request directly into GitHub.

---

## 🛠️ Tech Stack & Interoperability

| Layer | Technology |
|---|---|
| **Backend Engine** | Python 3, FastAPI, Uvicorn (Asynchronous State Handling) |
| **AI Substrate** | Groq LPU (Llama 3.3-70B), LangChain, LangGraph |
| **Data Streaming** | Advanced Async Server-Sent Events (SSE) with HTTP Keep-Alive protections |
| **Integration** | Secure GitHub API Federation (HTTPX) |
| **Frontend UI** | Vanilla CSS/JS, Tailwind CSS, Marked.js, Highlight.js for Real-Time Markdown rendering |

---

## 🏆 Hackathon Journey & Recognition

Built under insane time constraints, Opalite OS was selected as a **Top 10 Finalist out of 75 highly competitive teams**. 

While we didn't take home the ultimate grand prize, the overwhelming feedback from Judges, CTOs, and technical architects validated our core thesis: the future of DevOps is fully autonomous, agentic validation. We successfully demonstrated that AI can move beyond just "code completion" into closed-loop, deterministic infrastructure management.

---

## ⚙️ Setup & Run Local Engine

### 1. Clone the repository
```bash
git clone https://github.com/PDK45/CI-CD-Self-Healing-AI-Agent.git
cd CI-CD-Self-Healing-AI-Agent/backend
```

### 2. Environment Variables
Create a `.env` in the `backend` directory mapping your API keys:
```env
GROQ_API_KEY=gsk_your_groq_key_here
GITHUB_TOKEN=ghp_your_personal_access_token_here
PORT=8000
```

### 3. Initialize the Swarm
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 4. Connect the Control Plane
Open your browser to `http://localhost:8000` to interact with the frontend console, map your federated repositories, and trigger the AI Healing loop.

---
*Built with ❤️ for the future of developer velocity.*
