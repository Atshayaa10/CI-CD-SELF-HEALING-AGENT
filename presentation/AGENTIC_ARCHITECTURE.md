# 🤖 Opalite OS: Agentic Intelligence Deep-Dive

**Opalite OS is not an "Automation Script." It is a multi-agent autonomous system built on high-level reasoning and stateful orchestration.**

---

## 🧠 1. The Core: LangGraph Orchestration
Unlike traditional linear CI/CD pipelines, Opalite uses a **Directed Acyclic Graph (with Loops)** to manage reasoning.
*   **Stateful Memory:** We maintain a global `AgentState` that tracks logs, code context, and past verification failures.
*   **Decoupled Nodes:** Each agent (Diagnostician, Solver, Critic) is a specialized "Expert Node" with its own prompt-engineering and tool-access layer.
*   **Technical Issue Categorization (Jury Feedback):** Instead of simple "error parsing," the agent now identifies and labels granular **Technical Issues** (e.g., *Syntax Error*, *Dependency Conflict*, *IaC Misconfig*). This categorization drives smarter RAG retrieval and more targeted code generation.

---

## 🔄 2. The Self-Correction Loop (Agency Level 2)
The hallmark of a true agent is the ability to **self-correct**. Opalite implements a "Reasoning-Verification-Review" loop:
1.  **The Verifier (The Reality Check):** It executes the AI's proposal in a real shell. If the command fails, it doesn't just error out—it feeds the raw shell failure back into the graph.
2.  **The Critic (The Quality Gate):** A separate LLM personality reviews the code OR the test logs. If the Critic is unsatisfied, it instructs the **Solver** to "Try Again" with specific feedback.
3.  **Iteration:** The agent will autonomously loop between **Solving** and **Verifying** up to $N$ times until the Critic approves or a solution is reached.

---

## 📚 3. Memory & Learning (The Memory Crystal)
Opalite becomes smarter every time it fixes a repo. 
*   **Persistent RAG:** Successful "Heal Events" are etched into our **Opalite-Memory engine**.
*   **Contextual Retrieval:** During the `Solver` phase, the agent autonomously queries its own memory. It looks for similar past failures and uses "Few-Shot Learning" to apply those historical fixes to the current problem.

---

## 🛠️ 4. Autonomous Tool Use
The agent has "Hands" in the real world:
*   **GitHub REST Node:** For non-destructive operations (Fetching code, listing repos).
*   **Git Database Node:** For autonomous history management (Rollbacks, PR creation).
*   **Shell Node:** For running `pytest` or `pip install` in a sandboxed environment.

---

## ⚖️ 5. Why We Are Inter-Agentic
We don't just rely on one LLM call. We rely on the **Collaborative Intelligence** of:
*   **Agent A (Diagnostician):** Focuses on root cause analysis.
*   **Agent B (Solver):** Focuses on code generation.
*   **Agent C (Verifier):** Focuses on empirical testing.
*   **Agent D (Critic):** Focuses on architectural review.

**Opalite OS is a digital workforce, not a single tool.**
