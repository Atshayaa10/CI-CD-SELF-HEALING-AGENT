import os
import json
import time
import secrets
import asyncio
import httpx
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Cookie
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Ensure we find the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.tools.github_service import github_service, set_request_token
from agent.tools.gitlab_service import gitlab_service, set_request_token as set_gitlab_token
from agent.tools.test_runner import run_integration_tests
from agent.tools.memory_crystal import save_fix_to_memory, query_memory_for_fix


def _service_for(provider: str):
    """Return the SCM service + its token-setter for the given provider."""
    if provider == "gitlab":
        return gitlab_service, set_gitlab_token
    return github_service, set_request_token

app = FastAPI(title="Opalite CI/CD Auto-Healer", version="2.0.0")
templates = Jinja2Templates(directory="templates")

# --- GitHub OAuth (Login with GitHub) config ---
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", f"{APP_BASE_URL}/auth/github/callback")
OAUTH_SCOPES = "repo workflow read:user"

# --- GitLab OAuth ---
GITLAB_URL = os.getenv("GITLAB_URL", "https://gitlab.com").rstrip("/")
GITLAB_CLIENT_ID = os.getenv("GITLAB_CLIENT_ID", "")
GITLAB_CLIENT_SECRET = os.getenv("GITLAB_CLIENT_SECRET", "")
GITLAB_REDIRECT_URI = os.getenv("GITLAB_REDIRECT_URI", f"{APP_BASE_URL}/auth/gitlab/callback")
GITLAB_SCOPES = "api"  # needed to read pipelines and open/merge MRs
SESSION_COOKIE = "opalite_session"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # set true behind HTTPS

# In-memory stores. Fine for a single-instance demo; swap for Redis in production.
# The GitHub token lives ONLY here, server-side — it's never sent to the browser.
SESSIONS: dict = {}       # session_id -> {"token": str, "user": dict, "created": float}
_OAUTH_STATES: dict = {}  # state -> expiry timestamp (CSRF protection)


def _get_session(session_id: Optional[str]) -> Optional[dict]:
    return SESSIONS.get(session_id) if session_id else None


def _session_token(session_id: Optional[str]) -> Optional[str]:
    sess = _get_session(session_id)
    return sess["token"] if sess else None


def _session_provider(session_id: Optional[str]) -> str:
    sess = _get_session(session_id)
    return sess.get("provider", "github") if sess else "github"

# Shared LLM — Groq (Llama 3.3 70B)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("CRITICAL ERROR: GROQ_API_KEY is missing!")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=groq_api_key)

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    repo: Optional[str] = None   # the repo the user is asking about (for grounded answers)

class HealRequest(BaseModel):
    repo: str  # e.g. "PDK45/neoverse-test-pipeline"


# ======================================================================
#  GitHub OAuth — "Login with GitHub"
# ======================================================================
@app.get("/auth/github/login")
async def github_login():
    """Kick off the OAuth web flow: redirect the user to GitHub to authorize."""
    if not GITHUB_CLIENT_ID:
        return JSONResponse(
            {"error": "GitHub OAuth not configured. Set GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET in .env."},
            status_code=500,
        )
    # CSRF protection: one-time state, valid for 10 minutes.
    state = secrets.token_urlsafe(24)
    _OAUTH_STATES[state] = time.time() + 600
    from urllib.parse import urlencode
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": OAUTH_SCOPES,
        "state": state,
        "allow_signup": "false",
    }
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{urlencode(params)}")


@app.get("/auth/github/callback")
async def github_callback(code: str = None, state: str = None):
    """GitHub redirects back here with a code. Exchange it for a token, start a session."""
    # Validate + consume the CSRF state.
    expiry = _OAUTH_STATES.pop(state, None)
    if not code or not state or expiry is None or expiry < time.time():
        return RedirectResponse("/?auth=invalid_state")

    async with httpx.AsyncClient() as client:
        # 1. Exchange the code for an access token (secret stays server-side).
        token_r = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": OAUTH_REDIRECT_URI,
            },
        )
        token_data = token_r.json()
        access_token = token_data.get("access_token")
        if not access_token:
            return RedirectResponse("/?auth=failed")

        # 2. Identify the user for display purposes.
        user_r = await client.get(
            "https://api.github.com/user",
            headers={"Accept": "application/vnd.github.v3+json", "Authorization": f"token {access_token}"},
        )
        user = user_r.json() if user_r.status_code == 200 else {}

    # 3. Create a server-side session; only an opaque id goes to the browser.
    session_id = secrets.token_urlsafe(32)
    SESSIONS[session_id] = {
        "provider": "github",
        "token": access_token,
        "user": {"login": user.get("login"), "avatar_url": user.get("avatar_url"), "name": user.get("name")},
        "created": time.time(),
    }
    resp = RedirectResponse("/?auth=success")
    resp.set_cookie(
        SESSION_COOKIE, session_id,
        httponly=True, secure=COOKIE_SECURE, samesite="lax", max_age=60 * 60 * 8,
    )
    return resp


# ======================================================================
#  GitLab OAuth — "Login with GitLab"
# ======================================================================
@app.get("/auth/gitlab/login")
async def gitlab_login():
    if not GITLAB_CLIENT_ID:
        return JSONResponse(
            {"error": "GitLab OAuth not configured. Set GITLAB_CLIENT_ID / GITLAB_CLIENT_SECRET in .env."},
            status_code=500,
        )
    from urllib.parse import urlencode
    state = secrets.token_urlsafe(24)
    _OAUTH_STATES[state] = time.time() + 600
    params = {
        "client_id": GITLAB_CLIENT_ID,
        "redirect_uri": GITLAB_REDIRECT_URI,
        "response_type": "code",
        "scope": GITLAB_SCOPES,
        "state": state,
    }
    return RedirectResponse(f"{GITLAB_URL}/oauth/authorize?{urlencode(params)}")


@app.get("/auth/gitlab/callback")
async def gitlab_callback(code: str = None, state: str = None):
    expiry = _OAUTH_STATES.pop(state, None)
    if not code or not state or expiry is None or expiry < time.time():
        return RedirectResponse("/?auth=invalid_state")

    async with httpx.AsyncClient() as client:
        token_r = await client.post(
            f"{GITLAB_URL}/oauth/token",
            data={
                "client_id": GITLAB_CLIENT_ID,
                "client_secret": GITLAB_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GITLAB_REDIRECT_URI,
            },
        )
        access_token = token_r.json().get("access_token")
        if not access_token:
            return RedirectResponse("/?auth=failed")

        user_r = await client.get(f"{GITLAB_URL}/api/v4/user",
                                  headers={"Authorization": f"Bearer {access_token}"})
        user = user_r.json() if user_r.status_code == 200 else {}

    session_id = secrets.token_urlsafe(32)
    SESSIONS[session_id] = {
        "provider": "gitlab",
        "token": access_token,
        "user": {"login": user.get("username"), "avatar_url": user.get("avatar_url"), "name": user.get("name")},
        "created": time.time(),
    }
    resp = RedirectResponse("/?auth=success")
    resp.set_cookie(SESSION_COOKIE, session_id,
                    httponly=True, secure=COOKIE_SECURE, samesite="lax", max_age=60 * 60 * 8)
    return resp


@app.get("/api/session")
async def get_session_info(opalite_session: str = Cookie(default=None)):
    """Report whether the current browser is logged in, as whom, and via which provider."""
    sess = _get_session(opalite_session)
    if sess:
        return {"authenticated": True, "provider": sess.get("provider", "github"), "user": sess["user"]}
    return {
        "authenticated": False,
        "github_configured": bool(GITHUB_CLIENT_ID),
        "gitlab_configured": bool(GITLAB_CLIENT_ID),
    }


@app.post("/logout")
async def logout(opalite_session: str = Cookie(default=None)):
    """End the session and clear the cookie."""
    SESSIONS.pop(opalite_session, None)
    resp = JSONResponse({"success": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/api/repos")
async def get_repos(opalite_session: str = Cookie(default=None)):
    """List repos/projects for the logged-in user, dispatching on their provider."""
    token = _session_token(opalite_session)
    if not token:
        return JSONResponse({"success": False, "message": "Not authenticated. Please log in."}, status_code=401)
    provider = _session_provider(opalite_session)
    if provider == "gitlab":
        set_gitlab_token(token)
        return {"success": True, "provider": "gitlab", "repos": await gitlab_service.list_projects()}
    # GitHub
    return await get_github_repos(opalite_session)


@app.get("/api/github/repos")
async def get_github_repos(opalite_session: str = Cookie(default=None)):
    """List repositories the logged-in user can access. Token is read from the session."""
    token = _session_token(opalite_session)
    if not token:
        return JSONResponse({"success": False, "message": "Not authenticated. Log in with GitHub."}, status_code=401)

    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.github.com/user/repos",
            headers=headers,
            params={"per_page": 100, "sort": "updated", "affiliation": "owner,collaborator,organization_member"},
        )
        if r.status_code == 200:
            repos = r.json()
            return {"success": True, "repos": [
                {"name": repo["full_name"], "private": repo["private"], "updated_at": repo["updated_at"]}
                for repo in repos
            ]}
        return JSONResponse({"success": False, "message": f"GitHub API Error: {r.status_code}"}, status_code=r.status_code)

# --- Chat Endpoint ---
SYSTEM_PROMPT = """You are Opalite OS — the AI brain of the Opalite CI/CD Self-Healing Agent.
You help developers understand, debug, and improve their repositories and CI/CD pipelines.

FORMATTING RULES — always follow, no exceptions:
- Reply in short, scannable **Markdown bullet points** (`- `), not paragraphs.
- One idea per bullet; start each with a **bold lead phrase** when it helps.
- Wrap file names, commands, and identifiers in `backticks`.
- Use a short fenced code block ONLY when showing actual code.
- Use a numbered list when describing an ordered sequence of steps.
- Keep the whole answer tight — aim for 3–7 bullets unless more detail is asked for.

When repository code is provided below, ground every answer in it and cite specific files
(e.g. `app.py`). If the provided context is not enough to answer, say so in one bullet and
state exactly what's missing."""

# Cache built repo contexts so we don't re-fetch every message. Keyed by "provider:repo".
_CHAT_CONTEXT_CACHE: dict = {}
_CHAT_CODE_EXT = ('.py', '.js', '.ts', '.java', '.go', '.rb', '.yaml', '.yml', '.json', '.toml', '.cfg', '.ini', '.txt', '.md')


async def _build_chat_context(provider: str, repo: str, svc) -> str:
    """Fetch a capped snapshot of the repo's code to ground chat answers. Cached per repo."""
    key = f"{provider}:{repo}"
    if key in _CHAT_CONTEXT_CACHE:
        return _CHAT_CONTEXT_CACHE[key]

    files = await svc.get_repo_files(repo)
    code_files = [f for f in files if f.endswith(_CHAT_CODE_EXT) or f in ("Dockerfile", "Makefile", "Procfile")]

    context = f"Repository: {repo}\nFiles in root: {', '.join(files) or 'unknown'}\n"
    total = 0
    for f in code_files:
        if total > 16000:
            context += "\n--- [context truncated] ---\n"
            break
        content = await svc.get_file_content(repo, f)
        snippet = content[:4000]
        context += f"\n--- FILE: {f} ---\n{snippet}\n"
        total += len(snippet)

    _CHAT_CONTEXT_CACHE[key] = context
    return context


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, opalite_session: str = Cookie(default=None)):
    provider = _session_provider(opalite_session)
    token = _session_token(opalite_session)
    svc, set_token = _service_for(provider)
    repo = request.repo

    async def generate():
        try:
            system = SYSTEM_PROMPT
            if repo and token:
                set_token(token)
                try:
                    ctx = await _build_chat_context(provider, repo, svc)
                    system += f"\n\n=== REPOSITORY CONTEXT ({repo}) ===\n{ctx}\n=== END CONTEXT ==="
                except Exception as ctx_err:
                    system += f"\n\n(Note: could not load code for '{repo}': {ctx_err})"
            elif repo and not token:
                system += f"\n\n(The user asked about '{repo}' but is not logged in, so no code is available — tell them to log in to get repo-specific answers.)"

            messages = [SystemMessage(content=system), HumanMessage(content=request.message)]
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n\n[Agent Error]: {str(e)}"
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# --- HEAL Endpoint (the full autonomous pipeline) ---
@app.post("/heal")
async def heal_endpoint(req: HealRequest, opalite_session: str = Cookie(default=None)):
    """
    Analyzes a GitHub repo, finds broken code, fixes it with AI, and opens a PR.
    Streams every step live to the frontend as Server-Sent Events (SSE).
    """
    repo = req.repo
    provider = _session_provider(opalite_session)
    svc, set_token = _service_for(provider)
    # Auth: use the logged-in user's OAuth token; fall back to env for local/dev.
    active_token = _session_token(opalite_session) or os.getenv("GITHUB_TOKEN")
    if not active_token:
        async def err_gen():
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': 'Not authenticated. Log in to heal your repos.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def run_healing():
        import asyncio
        # Bind this user's token to every SCM API call made during the stream.
        set_token(active_token)
        async def ainvoke_with_keepalive(messages):
            task = asyncio.create_task(llm.ainvoke(messages))
            while not task.done():
                yield ": keepalive\n\n"
                await asyncio.sleep(2)
            yield task.result()

        try:
            # --- Step 1: Scan the repository ---
            yield f"data: {json.dumps({'step': 'scan', 'status': 'running', 'message': f'Scanning repository {repo}...'})}\n\n"
            files = await svc.get_repo_files(repo)
            code_files = [f for f in files if f.endswith(('.py', '.js', '.ts', '.java', '.yaml', '.yml', '.json', '.toml', '.conf', '.ini', '.txt')) or f in ('Dockerfile', 'Makefile')]
            yield f"data: {json.dumps({'step': 'scan', 'status': 'done', 'message': f'Found {len(code_files)} source & config files: {code_files}'})}\n\n"

            # --- Step 2: Fetch all code files ---
            yield f"data: {json.dumps({'step': 'fetch', 'status': 'running', 'message': 'Fetching source code from GitHub...'})}\n\n"
            all_code = {}
            for f in code_files:
                content = await svc.get_file_content(repo, f)
                all_code[f] = content
                yield f"data: {json.dumps({'step': 'fetch', 'status': 'progress', 'message': f'Fetched: {f} ({len(content)} chars)'})}\n\n"
            yield f"data: {json.dumps({'step': 'fetch', 'status': 'done', 'message': f'All {len(all_code)} files fetched.', 'details': 'Fetched files:\\n' + chr(10).join(code_files)})}\n\n"

            # --- Step 3: Diagnostician — AI analyzes the code for bugs ---
            yield f"data: {json.dumps({'step': 'diagnose', 'status': 'running', 'message': '🔍 Diagnostician Agent analyzing code for errors...'})}\n\n"

            code_context = ""
            MAX_CHARS_PER_FILE = 15000
            MAX_TOTAL_CHARS = 20000
            
            for path, code in all_code.items():
                if len(code_context) > MAX_TOTAL_CHARS:
                    code_context += f"\n--- [TRUNCATED: Max context size reached] ---\n"
                    break
                    
                file_content = code
                if len(file_content) > MAX_CHARS_PER_FILE:
                    file_content = file_content[:MAX_CHARS_PER_FILE] + f"\n... [TRUNCATED: File too large. Showing first {MAX_CHARS_PER_FILE} chars] ..."
                
                code_context += f"\n--- FILE: {path} ---\n{file_content}\n--- END ---\n"

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
            }}
            Return ONLY the JSON."""

            diag_response = None
            async for x in ainvoke_with_keepalive([HumanMessage(content=diag_prompt)]):
                if isinstance(x, str): yield x
                else: diag_response = x
                
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
            ```
            Return ONLY the code blocks. No explanations."""

            solve_response = None
            async for x in ainvoke_with_keepalive([HumanMessage(content=solve_prompt)]):
                if isinstance(x, str): yield x
                else: solve_response = x
                
            raw_fix = solve_response.content.strip()
            
            from agent.tools.test_runner import extract_files_from_patch
            files_map = await extract_files_from_patch(raw_fix)

            # Primary file to patch, derived from the diagnosis (used only for the fallback below).
            broken_file = tech_issues[0].get("path") if tech_issues else None

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
            is_success, test_output = await run_integration_tests(repo, "main", files_map, token=active_token, provider=provider)
            
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

            critic_response = None
            async for x in ainvoke_with_keepalive([HumanMessage(content=critic_prompt)]):
                if isinstance(x, str): yield x
                else: critic_response = x
                
            critic_verdict = critic_response.content.strip()

            if critic_verdict.upper().startswith("APPROVE"):
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': '✅ Critic: APPROVED', 'details': critic_verdict})}\n\n"
            else:
                yield f"data: {json.dumps({'step': 'critic', 'status': 'done', 'message': f'⚠️ Critic feedback: {critic_verdict[:100]}...', 'details': critic_verdict})}\n\n"

            # --- Step 5.9: Auto-install the CI/CD pipeline if the repo doesn't have one ---
            # A workflow file added in a commit runs on that commit's push, so this is
            # self-bootstrapping: this same PR turns the repo into a self-deploying one.
            if os.getenv("AUTO_INSTALL_CICD", "true").lower() == "true":
                installed = []
                if provider == "gitlab":
                    from agent.tools.cicd_templates import GITLAB_CI_YML
                    if not await svc.file_exists(repo, ".gitlab-ci.yml", "main"):
                        files_map[".gitlab-ci.yml"] = GITLAB_CI_YML
                        installed.append(".gitlab-ci.yml")
                else:
                    from agent.tools.cicd_templates import CI_YML, DEPLOY_YML
                    if not await svc.file_exists(repo, ".github/workflows/ci.yml", "main"):
                        files_map[".github/workflows/ci.yml"] = CI_YML
                        installed.append("ci.yml")
                    if not await svc.file_exists(repo, ".github/workflows/deploy.yml", "main"):
                        files_map[".github/workflows/deploy.yml"] = DEPLOY_YML
                        installed.append("deploy.yml")
                if installed:
                    installed_str = ", ".join(installed)
                    install_msg = f"🔧 No CI/CD pipeline found — installing {installed_str} in this PR so future merges auto-deploy."
                    yield f"data: {json.dumps({'step': 'push', 'status': 'progress', 'message': install_msg})}\n\n"

            # --- Step 6: Push the fixes to GitHub (Multi-file Support) ---
            yield f"data: {json.dumps({'step': 'push', 'status': 'running', 'message': f'🚀 Creating branch & committing {len(files_map)} files...'})}\n\n"

            pr_result = await svc.create_fix_branch_and_pr(
                repo_full_name=repo,
                base_branch="main",
                files_map=files_map,
                error_summary=error_summary
            )
            pr_url = pr_result["pr_url"]
            fix_head_sha = pr_result["head_sha"]

            yield f"data: {json.dumps({'step': 'push', 'status': 'done', 'message': f'Pull Request opened: {pr_url}'})}\n\n"

            # --- Step 7: CD Deployer — CI-gated auto-merge, then GitHub Actions deploys ---
            merged = False
            auto_deploy_eligible = ("APPROVE" in critic_verdict) and is_success
            if not auto_deploy_eligible:
                gate_reason = "local sandbox tests did not pass" if not is_success else "the Critic did not approve the fix"
                skip_msg = f"⛔ Auto-deploy gated: {gate_reason}. Fix delivered as a PR for human review — nothing was merged or deployed."
                yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': skip_msg, 'pr_url': pr_url})}\n\n"

            if auto_deploy_eligible:
                # 7a. CI GATE: wait for the fix branch's own GitHub Actions checks to pass.
                #     The agent only merges code that GitHub itself has marked green.
                ci_timeout = int(os.getenv("CI_GATE_TIMEOUT", "600"))
                ci_state = "none"
                if ci_timeout > 0:
                    gate_msg = f"⏳ CI Gate: waiting for GitHub Actions on the fix branch (timeout {ci_timeout}s)..."
                    yield f"data: {json.dumps({'step': 'ci_gate', 'status': 'running', 'message': gate_msg})}\n\n"
                    ci_result = await svc.wait_for_checks(repo, fix_head_sha, timeout=ci_timeout)
                    ci_state = ci_result["state"]
                    ci_summary = ci_result["summary"]

                    if ci_state == "failure":
                        blocked_msg = f"⛔ CD blocked — {ci_summary}. PR left open for human review."
                        done_msg = f"Healing done; deployment gated by failing CI. PR: {pr_url}"
                        yield f"data: {json.dumps({'step': 'ci_gate', 'status': 'error', 'message': blocked_msg, 'pr_url': pr_url})}\n\n"
                        yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': done_msg, 'pr_url': pr_url})}\n\n"
                        return
                    elif ci_state == "pending":
                        pending_msg = f"⏱️ {ci_summary} PR left open for human merge."
                        done_msg = f"Healing done; CI still running. PR: {pr_url}"
                        yield f"data: {json.dumps({'step': 'ci_gate', 'status': 'error', 'message': pending_msg, 'pr_url': pr_url})}\n\n"
                        yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': done_msg, 'pr_url': pr_url})}\n\n"
                        return
                    elif ci_state == "none":
                        none_msg = "⚠️ No CI on the repo — proceeding on the strength of the local sandbox verification."
                        yield f"data: {json.dumps({'step': 'ci_gate', 'status': 'done', 'message': none_msg})}\n\n"
                    else:
                        ok_msg = f"✅ {ci_summary}"
                        yield f"data: {json.dumps({'step': 'ci_gate', 'status': 'done', 'message': ok_msg})}\n\n"

                yield f"data: {json.dumps({'step': 'deploy', 'status': 'running', 'message': '🚢 CI green — auto-merging. GitHub Actions (deploy.yml) will ship it...'})}\n\n"

                # 7b. Merge. The push to main is what triggers real CD via .github/workflows/deploy.yml.
                merged = await svc.merge_pull_request(pr_url)
                if merged:
                    status_text = "PR successfully merged. "

                    # 7c. Agent-as-deployer: build + run the healed app locally, expose via ngrok.
                    if os.getenv("DEPLOY_ENABLED", "false").lower() == "true":
                        yield f"data: {json.dumps({'step': 'deploy', 'status': 'running', 'message': '🌐 Deploying the healed app and exposing it via ngrok...'})}\n\n"
                        try:
                            from agent.tools.deployer import deploy_repo
                            dep = await deploy_repo(repo, active_token, provider=provider)
                            if dep["success"]:
                                live_msg = f'✅ Live deployment ready: <a href="{dep["url"]}" target="_blank" class="underline text-emerald-300 font-bold">{dep["url"]}</a>'
                                yield f"data: {json.dumps({'step': 'deploy', 'status': 'done', 'message': live_msg, 'pr_url': dep['url']})}\n\n"
                            else:
                                yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': '⚠️ Auto-deploy failed: ' + dep['message']})}\n\n"
                        except Exception as dep_err:
                            yield f"data: {json.dumps({'step': 'deploy', 'status': 'error', 'message': f'⚠️ Auto-deploy error: {dep_err}'})}\n\n"

                    webhook_url = os.getenv("DEPLOYMENT_WEBHOOK")
                    if webhook_url:
                        deployed = await svc.trigger_deployment(webhook_url)
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

            if merged:
                final_msg = f'🎉 Healing & Deployment complete! Verified, auto-merged & shipped. PR: {pr_url}'
            else:
                final_msg = f'✅ Healing complete. Fix delivered as a PR for review — auto-deploy was gated (not verified/approved). PR: {pr_url}'
            yield f"data: {json.dumps({'step': 'complete', 'status': 'success', 'message': final_msg, 'pr_url': pr_url})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Error: {repr(e)}'})}\n\n"

    return StreamingResponse(run_healing(), media_type="text/event-stream")


# --- Rollback Endpoint ---
@app.post("/rollback")
async def rollback_endpoint(req: HealRequest, opalite_session: str = Cookie(default=None)):
    """
    Instantly rewinds the main branch to the previous commit (safe state)
    and then triggers the deployment webhook to restore production.
    """
    repo = req.repo
    provider = _session_provider(opalite_session)
    svc, set_token = _service_for(provider)
    # Auth: use the logged-in user's OAuth token; fall back to env for local/dev.
    active_token = _session_token(opalite_session) or os.getenv("GITHUB_TOKEN")
    if not active_token:
        async def err_gen():
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': 'Not authenticated. Log in to roll back.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def run_rollback():
        # Bind this user's token to every SCM API call made during the stream.
        set_token(active_token)
        try:
            # --- Step 1: Trigger SCM Rollback ---
            yield f"data: {json.dumps({'step': 'scan', 'status': 'running', 'message': f'🚨 Initiating Emergency Rollback for {repo}...'})}\n\n"

            result = await svc.rollback_to_previous_commit(repo, "main")
            
            if not result["success"]:
                yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Rollback failed: {result.get("message")}'})}\n\n"
                return
                
            commit_url = result.get("commit_url", "")
            yield f"data: {json.dumps({'step': 'push', 'status': 'done', 'message': f'✅ GitHub branch rolled back successfully!', 'pr_url': commit_url})}\n\n"

            # --- Step 2: Trigger Webhook to Redeploy ---
            webhook_url = os.getenv("DEPLOYMENT_WEBHOOK")
            if webhook_url:
                yield f"data: {json.dumps({'step': 'deploy', 'status': 'running', 'message': '🚢 Triggering webhook to redeploy the previous safe commit...'})}\n\n"
                deployed = await svc.trigger_deployment(webhook_url)
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
    # Starlette's current signature puts `request` first.
    return templates.TemplateResponse(request, "index.html")

@app.get("/favicon.ico")
async def favicon():
    return {}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
