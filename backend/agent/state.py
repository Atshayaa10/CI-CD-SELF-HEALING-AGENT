import operator
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    This class defines the shared state passed between the LangGraph nodes.
    Each node (Diagnostic, Solver, Critic) mutates or reads from this state.
    """
    
    # Core identifying info
    repository: str
    run_id: int
    commit_sha: str
    
    # 1. Diagnostician Context
    raw_logs: str
    error_summary: str | None
    files_to_fetch: List[str]
    technical_issues: List[dict] # Dedicated list of categorized issues
    
    # 2. Researcher Context
    file_contents: dict[str, str] # e.g., {'src/main.py': 'def hello()...'}
    
    # 3. Solver Output
    proposed_patch: str | None
    
    # 4. Verifier Output (Feature 1)
    test_results: str | None
    is_test_passed: bool | None
    
    # 5. Critic Output
    is_patch_approved: bool
    critic_feedback: str | None
    
    # LangChain messages for conversational history tracking 
    # (annotated to append messages rather than overwrite)
    messages: Annotated[list[BaseMessage], operator.add]
