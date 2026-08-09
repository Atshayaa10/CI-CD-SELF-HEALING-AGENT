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
    1. Categorize the failure into specific "Technical Issues".
    2. For each issue, identify:
       - Category: (e.g., Syntax Error, Dependency Conflict, Logic Bug, IaC Misconfiguration)
       - Root Cause: (Specific technical reason)
       - File Path: (Which file is broken)
       - Detail: (Developer-centric technical detail)

    Return your response EXACTLY as a JSON object:
    {{
        "error_summary": "High-level reason for failure",
        "files_to_fetch": ["path/to/file1.py", "path/to/Dockerfile"],
        "technical_issues": [
            {{
                "id": "ISSUE-001",
                "category": "Syntax Error",
                "root_cause": "Unexpected indent",
                "path": "math_utils.py",
                "detail": "Indent error at line 42 in subtract function"
            }}
        ]
    }}
    Return ONLY the JSON object.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        json_str = response.content.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        result_dict = json.loads(json_str.strip())
        summary = result_dict.get("error_summary", "Failed to parse summary")
        files_to_fetch = result_dict.get("files_to_fetch", [])
        technical_issues = result_dict.get("technical_issues", [])
    except Exception as e:
        print(f"  -> JSON Parse error in Diagnostician: {e}")
        summary = "Error parsing diagnosis JSON"
        files_to_fetch = []
        technical_issues = [{"category": "Diagnosis Error", "detail": str(e)}]

    return {
        "error_summary": summary,
        "files_to_fetch": files_to_fetch,
        "technical_issues": technical_issues
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
    error_summary = state['error_summary']

    # --- Query Memory Crystal for past fixes ---
    try:
        from agent.tools.memory_crystal import query_memory_for_fix
        # Use the first issue's category if available for more targeted RAG
        primary_issue = state.get("technical_issues", [{}])[0]
        issue_cat = primary_issue.get("category", None)
        
        past_fixes = query_memory_for_fix(error_summary, issue_category=issue_cat, n_results=1)
        memory_context = ""
        if past_fixes:
            match = past_fixes[0]
            print(f"  -> [MEMORY] Found highly relevant past fix for category: {match['category']}!")
            memory_context = f"\n\nCRITICAL CONTEXT: I have seen a very similar error in the past. Here is how I successfully fixed it before. Try to adapt this past fix to the current code:\\nPast Error: {match['past_error']}\\nPast Fix Applied:\\n{match['fix_patch']}"
    except Exception as e:
        print(f"  -> [MEMORY] Warning: Could not query Memory Crystal: {e}")
        memory_context = ""

    issues_context = ""
    for issue in state.get("technical_issues", []):
        issues_context += f"- [{issue.get('category', 'Bug')}] File: {issue.get('path', 'N/A')} - {issue.get('detail', 'N/A')}\n"

    prompt = f"""
    You are a Senior Software Engineer resolving multiple technical issues in a CI/CD build failure.

    High-level Summary: {state['error_summary']}

    Specific Issues to Resolve:
    {issues_context}{memory_context}

    Here is the relevant source or configuration code:
    {context}

    Critic's previous feedback (if any): {critic_feedback}

    Write the corrected version of each file that fixes ALL the issues listed above.
    Wrap each file in a markdown code block with the file path on the first line as a comment. Example:

    ```python
    # path/to/filename.py
    def fixed_function():
        pass
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
            
        # --- Save ALL fixed files to Memory Crystal ---
        try:
            from agent.tools.memory_crystal import save_fix_to_memory
            repo = state.get("repository", "unknown/repo")
            error_details = state.get("error_summary", "")
            
            # Use the actual fixed files from the Solver's patch
            from agent.tools.test_runner import extract_files_from_patch
            fixed_files = await extract_files_from_patch(state.get("proposed_patch", ""))
            
            primary_issue = state.get("technical_issues", [{}])[0]
            issue_cat = primary_issue.get("category", "General")

            if fixed_files and error_details:
                for f_path, f_code in fixed_files.items():
                    save_fix_to_memory(repo, error_details, f_path, f_code, issue_category=issue_cat)
                print(f"  -> [MEMORY] {len(fixed_files)} fixes etched into the Memory Crystal under category: {issue_cat}!")
        except Exception as e:
            print(f"  -> [MEMORY] Warning: Failed to write to Memory Crystal: {e}")
            
        return {"deployment_status": status}
    else:
        print("  -> Failed to merge PR via API.")
        return {"deployment_status": "Failed to auto-merge PR."}
