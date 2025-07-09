# ZIP Hybrid System Flow Diagram

## System Architecture Overview

The ZIP hybrid system is a sophisticated AI-driven continuous integration system that operates 24/7 to discover work, conduct research, create projects, and manage tasks autonomously.

## System Components

### 1. Entry Points
- **run_from_zip.sh** - Shell script that sets up environment and changes to zip directory
- **run_hybrid_continuous.py** - Python entry point that initializes the hybrid runner
- **EnhancedHybridOrchestrator** - Enhanced orchestrator with research evolution and pattern learning

### 2. Core Orchestrator Flow

```mermaid
graph TB
    A[run_from_zip.sh] -->|Sets env vars & PYTHONPATH| B[run_hybrid_continuous.py]
    B -->|Creates| C[HybridRunner]
    C -->|Initializes| D[EnhancedHybridOrchestrator]
    
    D -->|Starts| E[Initialize Components]
    E --> F[Start Workers]
    E --> G[Start Research Engine]
    E --> H[Initialize Work Finder]
    
    F -->|Creates 5 workers| I[Worker Pool]
    I -->|Each worker runs| J[Worker Loop]
    
    D -->|Runs| K[Main Orchestration Loop]
```

## Detailed Flow Diagrams

### System Startup Flow

```mermaid
sequenceDiagram
    participant Shell as run_from_zip.sh
    participant Env as Environment
    participant Runner as run_hybrid_continuous.py
    participant Orch as ContinuousOrchestrator
    participant State as StateManager
    
    Shell->>Shell: Change to zip directory
    Shell->>Shell: Clean state files
    Shell->>Env: Export .env.local variables
    Shell->>Env: Set PYTHONPATH to zip dir
    Shell->>Runner: Execute with arch -arm64
    
    Runner->>Runner: Setup logging
    Runner->>Orch: Create EnhancedHybridOrchestrator
    Orch->>State: Load state with repository discovery
    Orch->>Orch: Initialize AI Brain
    Orch->>Orch: Initialize components
    Orch->>Orch: Start workers (default: 5)
    Orch->>Orch: Start main loop
```

### Work Discovery Process

```mermaid
graph TD
    A[Main Loop - Every 2 seconds] -->|Check queue size| B{Queue < Target?}
    B -->|Yes| C[IntelligentWorkFinder.discover_work]
    B -->|No| A
    
    C --> D[Check Active Projects]
    D -->|No projects| E[Generate NEW_PROJECT tasks only]
    D -->|Has projects| F[Multi-Source Discovery]
    
    F --> G[Repository Work Discovery]
    F --> H[Portfolio Gap Analysis]
    F --> I[Research Opportunities]
    F --> J[Maintenance Tasks]
    F --> K[Integration Opportunities]
    
    G --> L[ProjectLifecycleAnalyzer]
    L -->|Cached 15 min| M[Analyze each repo lifecycle]
    
    H --> N[PortfolioIntelligence]
    N -->|Cached 5 min| O[Find market gaps]
    
    E --> P[Generate Dynamic Project Ideas]
    P --> Q[Market Research]
    
    L --> R[Filter & Prioritize]
    O --> R
    I --> R
    J --> R
    K --> R
    
    R --> S[Check Duplicates]
    S -->|Not duplicate| T[Add to Work Queue]
    S -->|All duplicates| U[Generate Alternative Work]
```

### Research Engine Flow (Gemini Deep Research)

```mermaid
graph TB
    A[Research Triggered] --> B[GeminiDeepResearch]
    
    B --> C[Phase 0: Market Timing]
    C -->|Score < 7| D[Abort Research]
    C -->|Score >= 7| E[Phase 1: Problem Discovery]
    
    E -->|25-30 problems| F[Phase 2: Problem Validation]
    F -->|Score >= 9/10| G[Phase 2.5: User Research]
    
    G --> H[Phase 3: Competition Analysis]
    H --> I[Phase 3.5: Technical Validation]
    
    I --> J[Phase 4: Market Sizing]
    J --> K[Phase 4.5: Unit Economics]
    
    K --> L[Phase 5: Final Selection]
    L -->|15x opportunity| M[Create MarketOpportunity]
    
    M --> N[Save Research Knowledge]
    N --> O[Generate NEW_PROJECT Work Items]
```

### Project Creation Flow

```mermaid
sequenceDiagram
    participant Worker as Worker Thread
    participant PC as ProjectCreator
    participant AI as AI Brain
    participant Git as LocalGitHubSimulator
    participant FS as File System
    
    Worker->>PC: execute NEW_PROJECT task
    PC->>AI: Generate project details
    AI-->>PC: Project spec with architecture
    
    PC->>PC: Validate all required fields
    PC->>Git: Check if project exists
    Git-->>PC: Not exists
    
    PC->>Git: Fork starter kit
    Git->>FS: Git clone laravel/react-starter-kit
    Git->>FS: Create new repo directory
    Git-->>PC: LocalRepository object
    
    PC->>PC: Customize project
    PC->>Git: Update package.json
    PC->>Git: Create README.md
    PC->>Git: Create ARCHITECTURE.md
    PC->>Git: Create MONETIZATION_STRATEGY.md
    PC->>Git: Create metadata.json
    
    PC->>PC: Create initial structure
    PC->>Git: Create directory structure
    PC->>Git: Create initial components
    
    PC->>Git: Create initial issues (optional in local)
    PC->>PC: Configure project settings
    
    PC->>Worker: Return success with all validations
    Worker->>Worker: Generate follow-up setup tasks
```

### Worker Management Flow

```mermaid
graph TD
    A[Orchestrator Start] --> B[Calculate Worker Distribution]
    B --> C[30% System Workers]
    B --> D[60% Project Workers]
    B --> E[10% Flex Workers]
    
    C --> F[worker_1: system_tasks]
    C --> G[worker_2: general]
    
    D --> H[worker_3: project_A]
    D --> I[worker_4: project_B]
    
    E --> J[worker_5: general]
    
    F --> K[Worker Loop]
    G --> K
    H --> K
    I --> K
    J --> K
    
    K --> L{Find Work}
    L -->|Specialized match| M[Execute Work]
    L -->|No match & idle > 30s| N[Take general work]
    L -->|No work| O[Sleep 1s]
    
    M --> P[Update metrics]
    P --> Q[Check for follow-up]
    Q --> K
```

### State Persistence Flow

```mermaid
graph LR
    A[System State] --> B[StateManager]
    B --> C{Storage Mode}
    
    C -->|File-based| D[system_state.json]
    C -->|Redis| E[Redis State Adapter]
    
    F[Orchestrator State] --> G[continuous_orchestrator_state.json]
    
    H[Task State] --> I[TaskPersistence]
    I --> J[task_state.json]
    
    K[Work Queue] --> L{Queue Type}
    L -->|In-memory| M[List in orchestrator]
    L -->|Redis| N[RedisWorkQueue]
    
    O[Claude Tasks] --> P[claude_execution_queue.json]
    
    Q[Completed Tasks] --> R[work_finder_completed_tasks.json]
```

### Local Mode Operations

```mermaid
graph TD
    A[LocalGitHubSimulator] --> B[Local Repository Management]
    
    B --> C[Git Operations]
    C --> D[git clone]
    C --> E[git add/commit]
    C --> F[git status]
    
    B --> G[File Operations]
    G --> H[Create/Update files]
    G --> I[Read file contents]
    
    B --> J[Issue Simulation]
    J --> K[issues/ directory]
    J --> L[issue-{number}.json files]
    
    B --> M[Repository Structure]
    M --> N[local_repos/repositories/]
    M --> O[{org}/{repo}/]
```

## Key Decision Points

### 1. Work Discovery Decision
```
IF no active projects exist:
    ONLY generate NEW_PROJECT tasks
ELSE:
    Run full discovery across all sources
    
IF discovered work < target/2:
    Use enhanced work generator
    
IF queue empty:
    Generate emergency work
```

### 2. Worker Assignment Decision
```
IF worker.specialization matches work.repository:
    Assign work
ELIF worker is general:
    Assign any work
ELIF worker idle > 30s AND work is system/general:
    Allow cross-specialization assignment
ELSE:
    Skip this work item
```

### 3. Task Execution Decision
```
IF task is NEW_PROJECT:
    Execute project creation with FULL validation
    ALL steps MUST succeed or entire creation fails
ELSE:
    Queue GitHub issue creation
    Mark task complete immediately
```

### 4. Research Trigger Decision
```
IF market timing score < 7:
    Abort research
    
IF problem validation score < 9:
    Skip problem
    
IF opportunity < 15x:
    Skip opportunity
```

## System Features

### Continuous Operation
- Main loop runs every 2 seconds
- Workers check for work continuously
- No artificial delays except brief sleep when idle

### Intelligent Work Discovery
- Multiple discovery sources
- Caching to reduce API calls (15 min lifecycle, 5 min portfolio)
- Duplicate detection and filtering
- Emergency work generation when queue empty

### Research Evolution
- 9-phase deep market research
- No timeouts - research continues until complete
- All phases data stored for knowledge synthesis
- Validates opportunities with 15x potential

### Project Creation
- Atomic operation - all or nothing
- Full validation at each step
- Comprehensive documentation generation
- Initial project structure setup

### State Management
- Multiple state files for different components
- Periodic state saves
- State restoration on startup
- Support for both file and Redis storage

### Worker Specialization
- Dynamic distribution based on projects
- Specialization with flexibility
- Idle workers can take general tasks
- Balanced workload distribution

## Error Handling

### Failed Tasks
- Exponential backoff cooldown
- Different cooldown strategies by error type
- Task reassignment capability
- Failed task tracking

### Repository Cleanup
- Hourly checks for deleted repositories
- Automatic removal of related work items
- State reload after cleanup

### Graceful Shutdown
- All workers complete current tasks
- State saved before exit
- Clean worker deregistration