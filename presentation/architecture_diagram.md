# 🏗️ Opalite OS: System Architecture

The following diagram visualizes the autonomous self-healing lifecycle, multi-agent orchestration, and the federated integration layer.

```mermaid
graph TD
    %% Layer 1: Perception
    subgraph PERCEPTION ["1. Perception Layer"]
        GH[GitHub Webhooks] --> BACKEND[Opalite Backend]
        UI[User Dashboard] --> BACKEND
    end

    %% Layer 2: Thinking (The Agentic Core)
    subgraph BRAIN ["2. Reasoning Hub (LangGraph)"]
        direction TB
        DIAG[Diagnostician] --> SOLVE[Solver]
        SOLVE --> VERIFY[Verifier Node]
        VERIFY -- "Feedback Loop" --> SOLVE
        VERIFY --> CRITIC[Critic Node]
        CRITIC -- "Rejection" --> SOLVE
    end

    %% Layer 3: Memory & Infrastructure
    subgraph INFRA ["3. Context & Memory"]
        CRYSTAL[(Memory Crystal RAG)] <--> SOLVE
        GROQ[[Groq LLM Engine]] <--> BRAIN
    end

    %% Layer 4: Action
    subgraph ACTION ["4. Execution Layer"]
        CRITIC -- "Approved" --> PR[Auto-PR / Merge]
        PR --> CD[Deployment]
        CD -- "Crash Detected" --> RB[Sentinel Rollback]
        RB -- "Fix" --> PR
    end

    BACKEND -- "Trigger" --> DIAG
```

![Opalite Premium Architecture Asset](C:\Users\Dk\.gemini\antigravity\brain\12d39119-a4e2-4c20-b734-81d25e5a4052\opalite_architecture_premium_1772899617857.png)


## 🧩 Architectural Component Breakdown

### 1. **Ingestion Layer (FastAPI)**
The high-performance entry point that manages authenticated user sessions (GitHub PAT) and listens for asynchronous failure events.

### 2. **Agentic Orchestration (LangGraph)**
The "Thinking" layer. Instead of a linear script, LangGraph allows for **cyclical reasoning**. The **Critic** and **Verifier** act as safety gates, forcing the **Solver** to self-correct until the code patch is perfect.

### 3. **Intelligence Engine (Groq + RAG)**
*   **Groq:** Provides ultra-fast inference for real-time healing.
*   **Memory Crystal:** A vector-similarity store that allows the agent to recall how it fixed similar bugs in other repositories, effectively "learning" over time.

### 4. **Remediation Layer**
*   **Verifier (Sandbox):** Executes the fix in an isolated environment to prevent breaking production further.
*   **Deployer:** Manages Git history (branches/PRs), handles auto-merges, and executes emergency rollbacks if health checks fail.
