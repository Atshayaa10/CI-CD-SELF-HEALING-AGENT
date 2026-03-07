import os
import json
from dotenv import load_dotenv

# Ensure we find the .env file in the backend/ directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState
from agent.context_builder import log_analyzer
from agent.tools.github_service import github_service
from agent.tools.test_runner import run_integration_tests, extract_files_from_patch

# Initialize Groq LLM — Llama 3.3-70B: excellent at code analysis and fixing
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("CRITICAL ERROR: GROQ_API_KEY is missing! The AI workflows will fail.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=groq_api_key)


async def diagnostician_node(state: AgentState):
    """
    Node 1: Receives the raw CI logs, extracts the error trace,
    and asks the LLM to identify the root cause and which files need to be fixed.
    """
    print(f"--- [DIAGNOSTICIAN] Analyzing Logs for Run {state['run_id']} ---")

    # Shrink massive logs down to the actual error trace
    trimmed_logs = log_analyzer.extract_error_trace(state["raw_logs"])

    prompt = f"""
    You are an expert CI/CD DevOps Engineer. A build pipeline just failed.
    Here is the exact error trace:

    <error_trace>
    {trimmed_logs}
    </error_trace>

    Your job is to:
    1. Briefly summarize why the build failed (1-2 sentences).
    2. Identify the exact file paths within the repository that likely need to be modified to fix this bug (e.g., source code, Dockerfile, requirements.txt, configuration files).

    Return your response EXACTLY as a JSON object with two keys:
    {{
        "summary": "Short explanation of the error",
        "files": ["path/to/file1.py", "path/to/Dockerfile"]
    }}
    Return ONLY the JSON object. No extra text.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Strip markdown code fences if the model wraps the JSON
        json_str = response.content.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        result_dict = json.loads(json_str.strip())
        summary = result_dict.get("summary", "Failed to parse summary")
        files_to_fetch = result_dict.get("files", [])
    except Exception as e:
        print(f"  -> JSON Parse error in Diagnostician: {e}")
        summary = response.content
        files_to_fetch = []

    print(f"  -> Summary: {summary}")
    print(f"  -> Files to fetch: {files_to_fetch}")

    return {
        "error_summary": summary,
        "files_to_fetch": files_to_fetch
    }


async def researcher_node(state: AgentState):
    """
    Node 2: Takes the files identified by the Diagnostician and fetches
    their actual raw content from the GitHub repository.
    """
    print(f"--- [RESEARCHER] Fetching {len(state['files_to_fetch'])} files from GitHub ---")

    file_contents = {}
    repo = state["repository"]
    commit = state["commit_sha"]

    for file_path in state["files_to_fetch"]:
        print(f"  -> Fetching: {file_path}")
        content = await github_service.get_file_content(repo, file_path, commit)
        file_contents[file_path] = content

    return {"file_contents": file_contents}


async def solver_node(state: AgentState):
    """
    Node 3: Takes the error summary and raw file contents, then writes a
    corrected version of each file to fix the bug.
    """
    print("--- [SOLVER] Drafting Code Patch ---")

    context = ""
    for path, code in state["file_contents"].items():
        context += f"\n\n--- START OF FILE: {path} ---\n{code}\n--- END OF FILE ---"

    critic_feedback = state.get("critic_feedback") or "N/A"

    prompt = f"""
    You are a Senior Software Engineer resolving a CI/CD build failure.

    Failure reason: {state['error_summary']}

    Here is the relevant source or configuration code:
    {context}

    Critic's previous feedback (if any): {critic_feedback}

    Write the corrected version of each file that fixes the issue.
    This could be a logic bug OR a server configuration error (like missing dependencies, Dockerfile syntax, environment setup).
    Wrap each file in a markdown code block with the file path on the first line as a comment. Example:

    ```python
    # path/to/filename.py
    def fixed_function():
        pass
    ```
    ```dockerfile
    # Dockerfile
    RUN pip install missing-package
    ```
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    print("  -> Patch drafted.")
    return {"proposed_patch": response.content}


async def verifier_node(state: AgentState):
    """
    Node 3.5: Executes the Solver's proposed patch locally in a sandbox
    and runs the test suite to verify if the fix actually works.
    """
    print("--- [VERIFIER] Running Local Test Sandbox ---")

    if not state.get("proposed_patch"):
        return {"test_results": "Error: No patch was proposed.", "is_test_passed": False}

    files = await extract_files_from_patch(state["proposed_patch"])
    
    is_success, output = await run_integration_tests(
        repo=state["repository"],
        commit_sha=state["commit_sha"],
        files_to_write=files
    )
    
    return {
        "is_test_passed": is_success,
        "test_results": output
    }

async def critic_node(state: AgentState):
    """
    Node 4: Reviews the Solver's proposed patch AND the Verifier's 
    local test execution results. Returns APPROVE or detailed feedback.
    """
    print("--- [CRITIC] Reviewing Solver's Patch & Test Results ---")

    test_status = "PASSED" if state.get("is_test_passed") else "FAILED"
    
    prompt = f"""
    You are a Staff Engineer doing a code review.
    A junior engineer proposed a fix for a CI/CD failure.
    We just ran the fix LOCALLY against the unit tests.

    Original Error: {state['error_summary']}

    Proposed Fix:
    {state['proposed_patch']}

    Local Test Execution Result: [{test_status}]
    Test Output Log:
    {state.get('test_results', 'No test results run')}

    If the fix correctly resolves the error, has zero syntax issues, AND the Local Tests PASSED, reply with exactly:
    APPROVE

    If it is wrong, introduces new bugs, or if the LOCAL TESTS FAILED, reply with specific feedback
    explaining what is wrong so the engineer can correct it. Include the failing test trace in your feedback.
    Do NOT just say REJECT.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if "APPROVE" in content:
        print("  -> Verdict: APPROVED ✅ (Tests Passed & Code Clean)")
        return {"is_patch_approved": True, "critic_feedback": None}
    else:
        print(f"  -> Verdict: REJECTED ❌\n  -> Feedback: {content[:120]}...")
        return {"is_patch_approved": False, "critic_feedback": content}

async def deployer_node(state: AgentState):
    """
    Node 5 (CD Hook): If Critic approves and tests pass, autonomously merges 
    the PR and hits a deployment trigger (Render/Vercel/etc).
    """
    print("--- [DEPLOYER] Triggering Continuous Deployment ---")
    if not state.get("pr_url"):
        return {"deployment_status": "No PR to merge."}
        
    merged = await github_service.merge_pull_request(state["pr_url"])
    if merged:
        print("  -> AI safely auto-merged the PR.")
        status = "PR successfully merged. "
        webhook_url = os.getenv("DEPLOYMENT_WEBHOOK")
        if webhook_url:
            deployed = await github_service.trigger_deployment(webhook_url)
            if deployed:
                status += "Deployment webhook triggered successfully!"
                print("  -> Production Deployment Trigger Hit!")
            else:
                status += "Failed to trigger deployment webhook."
                print("  -> Webhook failed to return 200.")
        else:
            status += "No DEPLOYMENT_WEBHOOK configured."
            print("  -> Skipped deploy webhook (not in .env).")
            
        return {"deployment_status": status}
    else:
        print("  -> Failed to merge PR via API.")
        return {"deployment_status": "Failed to auto-merge PR."}
