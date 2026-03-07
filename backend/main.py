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
from agent.tools.test_runner import run_integration_tests

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
            code_files = [f for f in files if f.endswith(('.py', '.js', '.ts', '.java', '.yaml', '.yml', '.json', '.toml', '.conf', '.ini', '.txt')) or f in ('Dockerfile', 'Makefile')]
            yield f"data: {json.dumps({'step': 'scan', 'status': 'done', 'message': f'Found {len(code_files)} source & config files: {code_files}'})}\n\n"

            # --- Step 2: Fetch all code files ---
            yield f"data: {json.dumps({'step': 'fetch', 'status': 'running', 'message': 'Fetching source code from GitHub...'})}\n\n"
            all_code = {}
            for f in code_files:
                content = await github_service.get_file_content(repo, f)
                all_code[f] = content
                yield f"data: {json.dumps({'step': 'fetch', 'status': 'progress', 'message': f'Fetched: {f} ({len(content)} chars)'})}\n\n"
            yield f"data: {json.dumps({'step': 'fetch', 'status': 'done', 'message': f'All {len(all_code)} files fetched.', 'details': 'Fetched files:\\n' + chr(10).join(code_files)})}\n\n"

            # --- Step 3: Diagnostician — AI analyzes the code for bugs ---
            yield f"data: {json.dumps({'step': 'diagnose', 'status': 'running', 'message': '🔍 Diagnostician Agent analyzing code for errors...'})}\n\n"

            code_context = ""
            for path, code in all_code.items():
                code_context += f"\n--- FILE: {path} ---\n{code}\n--- END ---\n"

            diag_prompt = f"""You are an expert DevOps engineer and code reviewer. Analyze ALL the following source and configuration files for bugs, syntax errors, missing dependencies, or bad infrastructure configurations (like Dockerfile/requirements.txt errors) that would cause tests or deployments to fail.

{code_context}

Return your analysis as a JSON object:
{{
    "has_errors": true/false,
    "error_category": "Code Bug", "Infrastructure Config", or "Dependency Error",
    "summary": "Brief description of the error(s) found",
    "broken_file": "path/to/broken_file.ext",
    "error_details": "Detailed explanation of what is wrong"
}}
Return ONLY the JSON. No extra text."""

            diag_response = llm.invoke([HumanMessage(content=diag_prompt)])
            diag_text = diag_response.content.strip()

            # Parse JSON from response more robustly
            import re
            print(f"--- RAW LLM DIAGNOSIS OUTPUT ---\n{diag_text}\n--------------------------------")
            json_match = re.search(r'\{[\s\S]*\}', diag_text)
            if json_match:
                diag_text = json_match.group(0)
            else:
                raise ValueError("Could not find a valid JSON object in the LLM response.")
                
            diagnosis = json.loads(diag_text.strip())

            diag_summary = diagnosis.get("summary", "No summary provided")
            diag_category = diagnosis.get("error_category", "Code Bug")
            yield f"data: {json.dumps({'step': 'diagnose', 'status': 'done', 'message': f'[{diag_category}] {diag_summary}', 'details': json.dumps(diagnosis, indent=2)})}\n\n"

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

            yield f"data: {json.dumps({'step': 'solve', 'status': 'done', 'message': f'Fix generated for {broken_file}', 'details': f'--- ORIGINAL ---\\n{broken_code}\\n\\n--- FIXED ---\\n{fixed_code}'})}\n\n"

            # --- Step 4.5: Verifier — Running local tests ---
            yield f"data: {json.dumps({'step': 'verify', 'status': 'running', 'message': '🧪 Verifier Agent duplicating repo to run local tests...'})}\n\n"
            
            is_success, test_output = await run_integration_tests(repo, "main", {broken_file: fixed_code})
            
            if test_output is None:
                short_test_output = "No output"
            else:
                short_test_output = str(test_output)[-1000:]
            
            if is_success:
                yield f"data: {json.dumps({'step': 'verify', 'status': 'done', 'message': '✅ Local Sandbox Tests PASSED!', 'details': short_test_output})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'verify', 'status': 'error', 'message': '❌ Local Sandbox Tests FAILED!', 'details': short_test_output})}\n\n"

            # --- Step 5: Critic — AI reviews the fix and test results ---
            yield f"data: {json.dumps({'step': 'critic', 'status': 'running', 'message': '✅ Critic Agent reviewing the patch & test outcome...'})}\n\n"

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

Local Test Execution Result: [{'PASSED' if is_success else 'FAILED'}]
Test Output Trimmed:
{short_test_output}

If the fix correctly resolves the error, has zero syntax issues, AND Local Tests PASSED, reply: APPROVE
Otherwise reply with what is wrong and include the failing test trace."""

            critic_response = llm.invoke([HumanMessage(content=critic_prompt)])
            critic_verdict = critic_response.content.strip()

            if "APPROVE" in critic_verdict:
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': '✅ Critic: APPROVED — patch looks correct', 'details': critic_verdict})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': f'⚠️ Critic feedback: {critic_verdict[:120]}... (proceeding anyway for demo)', 'details': critic_verdict})}\n\n"

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

            # --- Step 7: CD Deployer — AI Auto-Merges and Deploys ---
            if "APPROVE" in critic_verdict and is_success:
                yield f"data: {json.dumps({'step': 'deploy', 'status': 'running', 'message': '🚢 Executing Continuous Deployment (Auto-merge & Webhook)...'})}\n\n"
                
                # We replicate the deployer_node logic here for the stream
                merged = await github_service.merge_pull_request(pr_url)
                if merged:
                    status_text = "PR successfully merged. "
                    webhook_url = os.getenv("DEPLOYMENT_WEBHOOK")
                    if webhook_url:
                        deployed = await github_service.trigger_deployment(webhook_url)
                        if deployed:
                            status_text += "Deployment webhook triggered successfully!"
                            yield f"data: {json.dumps({'step': 'deploy', 'status': 'done', 'message': '✅ CD Success: ' + status_text})}\n\n"
                        else:
                            status_text += "Failed to trigger deployment webhook."
                            yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': '⚠️ CD Warning: ' + status_text})}\n\n"
                    else:
                        status_text += "No DEPLOYMENT_WEBHOOK configured."
                        yield f"data: {json.dumps({'step': 'deploy', 'status': 'done', 'message': '✅ CD Success: ' + status_text})}\n\n"
                else:
                    yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': '⚠️ CD Warning: Failed to auto-merge PR.'})}\n\n"

            yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': f'🎉 Healing & Deployment complete! PR: {pr_url}', 'pr_url': pr_url})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Error: {repr(e)}'})}\n\n"

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
