from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import diagnostician_node, researcher_node, solver_node, verifier_node, critic_node, deployer_node

# 1. Initialize the Graph using our TypedDict State
workflow = StateGraph(AgentState)

# 2. Add our AI Nodes
workflow.add_node("diagnostician", diagnostician_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("solver", solver_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("critic", critic_node)
workflow.add_node("deployer", deployer_node)

# 3. Define the edges (Flow of logic)
workflow.set_entry_point("diagnostician")
workflow.add_edge("diagnostician", "researcher")
workflow.add_edge("researcher", "solver")
workflow.add_edge("solver", "verifier")
workflow.add_edge("verifier", "critic")

# 4. Define the Conditional Edge (The "Reasoning Loop")
def check_critic_approval(state: AgentState):
    """
    If the critic approved, we are done (moves to remediation in Phase 4).
    If rejected, loop back to the solver with the new feedback in state.
    """
    if state["is_patch_approved"]:
         return "deploy"
    else:
         return "retry_solver"
         

workflow.add_conditional_edges(
    "critic",
    check_critic_approval,
    {
        "deploy": "deployer",
        "retry_solver": "solver"
    }
)

workflow.add_edge("deployer", END)

# 5. Compile the graph with memory checkpointing
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Optional Debugging Tool: Run `python graph.py` to save an image of the state machine chart
if __name__ == "__main__":
    from IPython.display import Image
    import sys
    
    try:
         graph_png = app.get_graph().draw_mermaid_png()
         with open("agent_architecture.png", "wb") as f:
             f.write(graph_png)
         print("Successfully exported graph architecture to agent_architecture.png")
    except Exception as e:
         print(f"Could not export graph. Make sure you have graphviz installed. {e}")
