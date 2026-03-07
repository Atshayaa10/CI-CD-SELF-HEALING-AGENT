# 🤖 Opalite — CI/CD Self-Healing AI Agent

An AI-powered SaaS platform that autonomously detects, diagnoses, and fixes failing CI/CD pipelines. When a GitHub Actions build fails, the system intercepts the webhook, analyzes logs using a multi-agent AI debate loop, generates a code patch, and opens a Pull Request — all without human intervention.

> **AI Model:** Groq · Llama 3.3-70B · Free Tier · ~14,400 requests/day

---

## 🎯 How It Works

```
GitHub Webhook (failure detected)
        ↓
  Diagnostician  →  Reads CI logs, identifies root cause
        ↓
   Researcher    →  Fetches the failing source files via GitHub API
        ↓
    Solver       →  Generates a code patch (diff) to fix the issue
        ↓
    Critic       →  Reviews patch for correctness & security
        ↓
  [APPROVED] ──→  Creates branch → Opens Pull Request on GitHub
  [REJECTED]  ──→  Loops back to Solver (max 3 iterations)
```

---

## ✅ Current Status (MVP Complete)

| Phase | Status |
|---|---|
| System Architecture & Design | ✅ Complete |
| FastAPI Backend & GitHub Webhook Engine | ✅ Complete |
| LangGraph Multi-Agent Orchestration | ✅ Complete |
| Conversational Chat API (`/chat`) | ✅ Complete |
| SaaS Dashboard (dark-mode UI) | ✅ Complete |
| End-to-End Test with Real Repo | 🔲 Next |
| Sandbox Testing (Docker) | 🔲 Upcoming |
| Production Deployment | 🔲 Upcoming |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x, FastAPI, Uvicorn |
| AI Model | Groq — Llama 3.3-70B (free) |
| AI Orchestration | LangChain, LangGraph |
| GitHub Integration | GitHub REST API (HTTPX) |
| Frontend | Tailwind CSS, Vanilla JS, Jinja2 |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/PDK45/CI-CD-Self-Healing-AI-Agent.git
cd CI-CD-Self-Healing-AI-Agent/backend
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Mac/Linux
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file inside the `backend/` folder:
```env
GROQ_API_KEY=your_groq_api_key_here
GITHUB_TOKEN=your_github_personal_access_token_here
PORT=8000
```

Get your **free Groq API key** at: [console.groq.com](https://console.groq.com) (takes 30 seconds)

### 4. Run the server
```bash
python main.py
```

Open **[http://localhost:8000](http://localhost:8000)** — the dashboard will load with a live AI chat.

---

## 📁 Project Structure

```
backend/
├── main.py                    # FastAPI server — /chat (streaming) and /webhook endpoints
├── requirements.txt
├── .env                       # API keys (not committed)
├── templates/
│   └── index.html             # SaaS Dashboard UI
└── agent/
    ├── graph.py               # LangGraph state machine wiring
    ├── nodes.py               # AI agents: Diagnostician, Researcher, Solver, Critic
    ├── state.py               # Shared agent state schema
    ├── context_builder.py     # CI log parser — extracts exact error traces
    └── tools/
        └── github_service.py  # GitHub API: fetch logs, create branches, open PRs
```

---

## 🔗 Connect a Real GitHub Repository

1. Run the server with a public URL (e.g. via [ngrok](https://ngrok.com): `ngrok http 8000`)
2. Go to your GitHub repo → **Settings → Webhooks → Add webhook**
3. Set Payload URL to: `https://your-ngrok-url/webhook`
4. Content type: `application/json`
5. Select event: **Workflow runs**

When a CI build fails, the agent loop triggers automatically and opens a PR with the fix.
