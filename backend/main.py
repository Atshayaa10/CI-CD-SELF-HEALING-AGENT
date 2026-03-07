import os
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Ensure we find the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.tools.github_service import github_service

app = FastAPI(title="Opalite CI/CD Auto-Healer", version="2.0.0")
templates = Jinja2Templates(directory="templates")

# Shared LLM — Groq (Llama 3.3 70B)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("CRITICAL ERROR: GROQ_API_KEY is missing!")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=groq_api_key)

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str

class HealRequest(BaseModel):
    repo: str  # e.g. "PDK45/neoverse-test-pipeline"


# --- Chat Endpoint ---
SYSTEM_PROMPT = """You are Opalite OS — the AI brain of the Opalite CI/CD Self-Healing Agent.
You autonomously monitor, diagnose, and fix failing CI/CD pipelines on GitHub.
You are part of a multi-agent system (Diagnostician, Researcher, Solver, Critic) built with LangGraph.
Always respond as Opalite OS. Be precise, technical, and helpful."""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def generate():
        try:
            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request.message)]
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n\n[Agent Error]: {str(e)}"
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# --- HEAL Endpoint (the full autonomous pipeline) ---
@app.post("/heal")
async def heal_endpoint(req: HealRequest):
    """
    Analyzes a GitHub repo, finds broken code, fixes it with AI, and opens a PR.
    Streams every step live to the frontend as Server-Sent Events (SSE).
    """
    repo = req.repo

    async def run_healing():
        try:
            # --- Step 1: Scan the repository ---
            yield f"data: {json.dumps({'step': 'scan', 'status': 'running', 'message': f'Scanning repository {repo}...'})}\n\n"
            files = await github_service.get_repo_files(repo)
            code_files = [f for f in files if f.endswith(('.py', '.js', '.ts', '.java', '.yaml', '.yml'))]
            yield f"data: {json.dumps({'step': 'scan', 'status': 'done', 'message': f'Found {len(code_files)} code files: {code_files}'})}\n\n"

            # --- Step 2: Fetch all code files ---
            yield f"data: {json.dumps({'step': 'fetch', 'status': 'running', 'message': 'Fetching source code from GitHub...'})}\n\n"
            all_code = {}
            for f in code_files:
                content = await github_service.get_file_content(repo, f)
                all_code[f] = content
                yield f"data: {json.dumps({'step': 'fetch', 'status': 'progress', 'message': f'Fetched: {f} ({len(content)} chars)'})}\n\n"
            yield f"data: {json.dumps({'step': 'fetch', 'status': 'done', 'message': f'All {len(all_code)} files fetched.'})}\n\n"

            # --- Step 3: Diagnostician — AI analyzes the code for bugs ---
            yield f"data: {json.dumps({'step': 'diagnose', 'status': 'running', 'message': '🔍 Diagnostician Agent analyzing code for errors...'})}\n\n"

            code_context = ""
            for path, code in all_code.items():
                code_context += f"\n--- FILE: {path} ---\n{code}\n--- END ---\n"

            diag_prompt = f"""You are an expert code reviewer. Analyze ALL the following source files for bugs, syntax errors, logic errors, or anything that would cause tests to fail.

{code_context}

Return your analysis as a JSON object:
{{
    "has_errors": true/false,
    "summary": "Brief description of the error(s) found",
    "broken_file": "path/to/broken_file.py",
    "error_details": "Detailed explanation of what is wrong"
}}
Return ONLY the JSON. No extra text."""

            diag_response = llm.invoke([HumanMessage(content=diag_prompt)])
            diag_text = diag_response.content.strip()

            # Parse JSON from response
            if diag_text.startswith("```"):
                diag_text = diag_text.split("```")[1]
                if diag_text.startswith("json"):
                    diag_text = diag_text[4:]
            diagnosis = json.loads(diag_text.strip())

            diag_summary = diagnosis.get("summary", "No summary provided")
            yield f"data: {json.dumps({'step': 'diagnose', 'status': 'done', 'message': f'Diagnosis: {diag_summary}'})}\n\n"

            if not diagnosis.get("has_errors"):
                yield f"data: {json.dumps({'step': 'complete', 'status': 'clean', 'message': '✅ No errors found! Repository code looks clean.'})}\n\n"
                return

            broken_file = diagnosis.get("broken_file", "")
            error_summary = diagnosis.get("summary", "Unknown error")

            # --- Step 4: Solver — AI writes the fix ---
            yield f"data: {json.dumps({'step': 'solve', 'status': 'running', 'message': f'🔧 Solver Agent writing fix for {broken_file}...'})}\n\n"

            broken_code = all_code.get(broken_file, "File not found")
            solve_prompt = f"""You are a Senior Software Engineer. Fix the following broken code.

Error: {diagnosis.get('error_details', error_summary)}

Broken file ({broken_file}):
```
{broken_code}
```

Write the COMPLETE corrected file. Return ONLY the fixed code, no explanations, no markdown fences."""

            solve_response = llm.invoke([HumanMessage(content=solve_prompt)])
            fixed_code = solve_response.content.strip()
            # Strip markdown fences if present
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")
                fixed_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

            yield f"data: {json.dumps({'step': 'solve', 'status': 'done', 'message': f'Fix generated for {broken_file}'})}\n\n"

            # --- Step 5: Critic — AI reviews the fix ---
            yield f"data: {json.dumps({'step': 'critic', 'status': 'running', 'message': '✅ Critic Agent reviewing the patch...'})}\n\n"

            critic_prompt = f"""You are a Staff Engineer reviewing a code fix.

Original error: {error_summary}

Original broken code:
```
{broken_code}
```

Proposed fix:
```
{fixed_code}
```

If the fix correctly resolves the error and has zero syntax issues, reply: APPROVE
Otherwise reply with what is wrong."""

            critic_response = llm.invoke([HumanMessage(content=critic_prompt)])
            critic_verdict = critic_response.content.strip()

            if "APPROVE" in critic_verdict:
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': '✅ Critic: APPROVED — patch looks correct'})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': f'⚠️ Critic feedback: {critic_verdict[:120]}... (proceeding anyway for demo)'})}\n\n"

            # --- Step 6: Push the fix to GitHub ---
            yield f"data: {json.dumps({'step': 'push', 'status': 'running', 'message': '🚀 Creating branch and opening Pull Request...'})}\n\n"

            pr_url = await github_service.create_fix_branch_and_pr(
                repo_full_name=repo,
                base_branch="main",
                file_path=broken_file,
                new_content=fixed_code,
                error_summary=error_summary
            )

            yield f"data: {json.dumps({'step': 'push', 'status': 'done', 'message': f'Pull Request opened: {pr_url}'})}\n\n"
            yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': f'🎉 Healing complete! PR: {pr_url}', 'pr_url': pr_url})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Error: {str(e)}'})}\n\n"

    return StreamingResponse(run_healing(), media_type="text/event-stream")


# --- Dashboard & Webhook ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return {}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
