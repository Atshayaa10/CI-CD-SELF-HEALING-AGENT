import os
import json
import asyncio
import httpx
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
from agent.tools.memory_crystal import save_fix_to_memory, query_memory_for_fix

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
    token: str = None

class AuthRequest(BaseModel):
    token: str

# --- GitHub Auth & Multi-Repo Endpoint ---
@app.post("/api/github/repos")
async def get_github_repos(req: AuthRequest):
    """Fetches all repositories accessible by the given GitHub token."""
    url = "https://api.github.com/user/repos"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {req.token}"
    }
    # Pagination support for up to 100 repos
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, params={"per_page": 100, "sort": "updated"})
        if r.status_code == 200:
            repos = r.json()
            return {"success": True, "repos": [{"name": repo["full_name"], "private": repo["private"], "updated_at": repo["updated_at"]} for repo in repos]}
        return {"success": False, "message": f"GitHub API Error: {r.status_code} - {r.text}"}

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
    # Auth Consistency: Use provided token if available, else fallback to env
    active_token = req.token or os.getenv("GITHUB_TOKEN")
    if not active_token:
        async def err_gen():
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': 'Authentication Error: GITHUB_TOKEN not found.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
    
    # Temporarily override service token for this request context
    github_service.token = active_token

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

            diag_prompt = f"""You are an expert DevOps engineer and code reviewer. Analyze the failure and categorize it into specific "Technical Issues".
            
            {code_context}

            Return your response EXACTLY as a JSON object:
            {{
                "has_errors": true,
                "error_summary": "High-level reason",
                "files_to_fetch": ["path/to/file.py"],
                "technical_issues": [
                    {{
                        "id": "ISSUE-001",
                        "category": "Code Bug", "Infrastructure Config", or "Dependency Error",
                        "path": "file.py",
                        "detail": "Specific technical detail for developers"
                    }}
                ]
            }}
            Return ONLY the JSON."""

            diag_response = llm.invoke([HumanMessage(content=diag_prompt)])
            diag_text = diag_response.content.strip()

            import re
            json_match = re.search(r'\{[\s\S]*\}', diag_text)
            if json_match:
                diagnosis = json.loads(json_match.group(0))
            else:
                raise ValueError("Valid JSON not found in diagnosis.")

            diag_summary = diagnosis.get("error_summary", "No summary")
            tech_issues = diagnosis.get("technical_issues", [])
            yield f"data: {json.dumps({'step': 'diagnose', 'status': 'done', 'message': f'✅ Detected {len(tech_issues)} Technical Issues', 'details': json.dumps(diagnosis, indent=2)})}\n\n"

            if not diagnosis.get("has_errors") or not tech_issues:
                yield f"data: {json.dumps({'step': 'complete', 'status': 'clean', 'message': '✅ No critical issues found! Repository looks stable.'})}\n\n"
                return

            error_summary = diagnosis.get("error_summary", "Unknown error")
            
            # --- Step 3.5: Memory Crystal (RAG) — AI searches past patterns by category ---
            primary_cat = tech_issues[0].get("category", "General") if tech_issues else "General"
            yield f"data: {json.dumps({'step': 'memory', 'status': 'running', 'message': f'🧠 Memory Crystal matching solutions for {primary_cat} issues...'})}\n\n"

            try:
                from agent.tools.memory_crystal import query_memory_for_fix
                past_fixes = query_memory_for_fix(error_summary, issue_category=primary_cat, n_results=1)
                memory_context = ""
                if past_fixes:
                    match = past_fixes[0]
                    memory_context = f"\n\nCRITICAL CONTEXT: Found a past {match['category']} fix from history:\n{match['fix_patch']}"
                    yield f"data: {json.dumps({'step': 'memory', 'status': 'done', 'message': f'✅ Memory Crystal matched a pattern for {match['category']}!', 'details': match['fix_patch']})}\n\n"
                else:
                    memory_context = ""
                    yield f"data: {json.dumps({'step': 'memory', 'status': 'done', 'message': '🧠 No specific past pattern found. Reasoning from first principles.'})}\n\n"
            except Exception as mem_err:
                memory_context = ""
                yield f"data: {json.dumps({'step': 'memory', 'status': 'error', 'message': f'Memory Crystal error: {mem_err}'})}\n\n"

            # --- Step 4: Solver — AI writes the fix addressing specific issues ---
            yield f"data: {json.dumps({'step': 'solve', 'status': 'running', 'message': f'🔧 Solver Agent remediating {len(tech_issues)} categorized issues...'})}\n\n"

            solve_prompt = f"""You are a Senior Software Engineer. Fix the following categorized issues:
            
            Summary: {error_summary}
            Technical Issues: {json.dumps(tech_issues, indent=2)}
            {memory_context}

            codebase:
            {code_context}

            Identify ALL files that need fixing. Return the COMPLETE content of each fixed file in markdown code blocks.
            Include the file path on the first line as a comment. Example:
            
            ```python
            # path/to/file.py
            def fixed_function():
                pass
            ```
            Return ONLY the code blocks. No explanations."""

            solve_response = llm.invoke([HumanMessage(content=solve_prompt)])
            raw_fix = solve_response.content.strip()
            
            from agent.tools.test_runner import extract_files_from_patch
            files_map = await extract_files_from_patch(raw_fix)
            
            # Fallback: If extract_files_from_patch failed but we have a single broken_file and LLM returned code
            if not files_map and broken_file:
                fixed_code = raw_fix
                if fixed_code.startswith("```"):
                    first_nl = fixed_code.find("\n")
                    fixed_code = fixed_code[first_nl:].strip() if first_nl != -1 else fixed_code[3:].strip()
                    if fixed_code.endswith("```"): fixed_code = fixed_code[:-3].strip()
                files_map[broken_file] = fixed_code

            if not files_map:
                yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': '❌ Solver Error: Failed to generate or parse any code fixes.'})}\n\n"
                return

            display_files = ", ".join(files_map.keys())
            yield f"data: {json.dumps({'step': 'solve', 'status': 'done', 'message': f'Fixes generated for: {display_files}', 'details': raw_fix})}\n\n"

            # --- Step 4.5: Verifier — Running local tests ---
            yield f"data: {json.dumps({'step': 'verify', 'status': 'running', 'message': f'🧪 Verifier Agent duplicating repo to run local tests on {len(files_map)} files...'})}\n\n"
            
            from agent.tools.test_runner import run_integration_tests
            is_success, test_output = await run_integration_tests(repo, "main", files_map)
            
            short_test_output = str(test_output or "No output")[-1500:]
            
            if is_success:
                yield f"data: {json.dumps({'step': 'verify', 'status': 'done', 'message': '✅ Local Sandbox Tests PASSED!', 'details': short_test_output})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'verify', 'status': 'error', 'message': '❌ Local Sandbox Tests FAILED!', 'details': short_test_output})}\n\n"

            # --- Step 5: Critic — AI reviews the fix and test results ---
            yield f"data: {json.dumps({'step': 'critic', 'status': 'running', 'message': '✅ Critic Agent reviewing the patch & test outcome...'})}\n\n"

            critic_prompt = f"""You are a Staff Engineer reviewing a code fix.
Original errors: {error_summary}
Proposed fixes:
{raw_fix}

Local Test Execution Result: [{'PASSED' if is_success else 'FAILED'}]
Test Output Trimmed:
{short_test_output}

You MUST reply with 'APPROVE' at the START of your response if the fix is correct. Otherwise, explain what is missing."""

            critic_response = llm.invoke([HumanMessage(content=critic_prompt)])
            critic_verdict = critic_response.content.strip()

            if critic_verdict.upper().startswith("APPROVE"):
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': '✅ Critic: APPROVED', 'details': critic_verdict})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': f'⚠️ Critic feedback: {critic_verdict[:100]}...', 'details': critic_verdict})}\n\n"

            # --- Step 6: Push the fixes to GitHub (Multi-file Support) ---
            yield f"data: {json.dumps({'step': 'push', 'status': 'running', 'message': f'🚀 Creating branch & committing {len(files_map)} files...'})}\n\n"

            pr_url = await github_service.create_fix_branch_and_pr(
                repo_full_name=repo,
                base_branch="main",
                files_map=files_map,
                error_summary=error_summary
            )
            
            # If there are more files, we should ideally add them to the SAME branch.
            # But create_fix_branch_and_pr opens a PR immediately.
            # For a hackathon demo, resolving the primary bug is key.

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
                            
                            # Step 7.5: Save to Memory Crystal for ALL fixed files
                            try:
                                for f_path, f_code in files_map.items():
                                    save_fix_to_memory(repo, error_summary, f_path, f_code)
                                yield f"data: {json.dumps({'step': 'memory', 'status': 'success', 'message': f'💎 {len(files_map)} fixes etched into the Memory Crystal!'})}\n\n"
                            except Exception as mem_err:
                                yield f"data: {json.dumps({'step': 'memory', 'status': 'error', 'message': f'Could not save to Memory Crystal: {mem_err}'})}\n\n"
                        else:
                            status_text += "Failed to trigger deployment webhook."
                            yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': '⚠️ CD Warning: ' + status_text})}\n\n"
                    else:
                        status_text += "No DEPLOYMENT_WEBHOOK configured."
                        yield f"data: {json.dumps({'step': 'deploy', 'status': 'done', 'message': '✅ CD Success: ' + status_text})}\n\n"
                        
                        # Save to Memory Crystal even if no webhook
                        try:
                            for f_path, f_code in files_map.items():
                                save_fix_to_memory(repo, error_summary, f_path, f_code)
                            yield f"data: {json.dumps({'step': 'memory', 'status': 'success', 'message': f'💎 {len(files_map)} fixes etched into the Memory Crystal!'})}\n\n"
                        except Exception as mem_err:
                            yield f"data: {json.dumps({'step': 'memory', 'status': 'error', 'message': f'Could not save to Memory Crystal: {mem_err}'})}\n\n"
                else:
                    yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': '⚠️ CD Warning: Failed to auto-merge PR.'})}\n\n"

            yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': f'🎉 Healing & Deployment complete! PR: {pr_url}', 'pr_url': pr_url})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Error: {repr(e)}'})}\n\n"

    return StreamingResponse(run_healing(), media_type="text/event-stream")


# --- Rollback Endpoint ---
@app.post("/rollback")
async def rollback_endpoint(req: HealRequest):
    """
    Instantly rewinds the main branch to the previous commit (safe state)
    and then triggers the deployment webhook to restore production.
    """
    repo = req.repo
    # Auth Consistency: Use provided token if available, else fallback to env
    active_token = req.token or os.getenv("GITHUB_TOKEN")
    if not active_token:
        async def err_gen():
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': 'Authentication Error: GITHUB_TOKEN not found.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
    
    # Temporarily override service token for this request context
    github_service.token = active_token

    async def run_rollback():
        try:
            # --- Step 1: Trigger GitHub Rollback ---
            yield f"data: {json.dumps({'step': 'scan', 'status': 'running', 'message': f'🚨 Initiating Emergency Rollback for {repo}...'})}\n\n"
            
            result = await github_service.rollback_to_previous_commit(repo, "main")
            
            if not result["success"]:
                yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Rollback failed: {result.get("message")}'})}\n\n"
                return
                
            commit_url = result.get("commit_url", "")
            yield f"data: {json.dumps({'step': 'push', 'status': 'done', 'message': f'✅ GitHub branch rolled back successfully!', 'pr_url': commit_url})}\n\n"

            # --- Step 2: Trigger Webhook to Redeploy ---
            webhook_url = os.getenv("DEPLOYMENT_WEBHOOK")
            if webhook_url:
                yield f"data: {json.dumps({'step': 'deploy', 'status': 'running', 'message': '🚢 Triggering webhook to redeploy the previous safe commit...'})}\n\n"
                deployed = await github_service.trigger_deployment(webhook_url)
                if deployed:
                    yield f"data: {json.dumps({'step': 'deploy', 'status': 'done', 'message': '✅ Production rollback deployment triggered!'})}\n\n"
                else:
                    yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': '⚠️ Failed to trigger deployment webhook.'})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'deploy', 'status': 'done', 'message': '✅ Rollback complete (No webhook configured to trigger).'})}\n\n"

            yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': f'🎉 Emergency Rollback Complete! Production stabilized.', 'pr_url': commit_url})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Error: {repr(e)}'})}\n\n"

    return StreamingResponse(run_rollback(), media_type="text/event-stream")


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
