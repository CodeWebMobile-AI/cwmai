# ZIP Hybrid System Flow Diagram

## 🚀 System Startup

```mermaid
graph TD
    A[./run_from_zip.sh] --> B[Clean state files if --reset]
    B --> C[Export environment variables]
    C --> D[cd zip/]
    D --> E[python3 run_hybrid_continuous.py --workers 10]
    E --> F[HybridRunner.__init__]
    F --> G[Initialize ContinuousOrchestrator]
    G --> H[Load system state]
    H --> I[Initialize components]
    I --> J[Start main loop]
```

### Key Components Initialized:
1. **StateManager** - Handles system_state.json
2. **IntelligentWorkFinder** - Discovers new work
3. **GeminiDeepResearch** - Market research engine
4. **ProjectCreator** - Creates new projects
5. **LocalGitHubSimulator** - Local Git operations
6. **UnifiedTaskState** - Task tracking
7. **Worker Pool** - Parallel execution (10 workers)

## 🔄 Main Orchestration Loop

```mermaid
graph TD
    A[Main Loop - Every 30 seconds] --> B{Work Queue Empty?}
    B -->|Yes| C[Discover Work]
    B -->|No| D[Assign Work to Workers]
    
    C --> E{Active Projects Exist?}
    E -->|No| F[Generate NEW_PROJECT Only]
    E -->|Yes| G[Multi-Source Discovery]
    
    F --> H[Intensive Research Mode]
    G --> I[Repository Work]
    G --> J[Portfolio Gaps]
    G --> K[System Tasks]
```

## 🔬 Intensive Research Flow (NEW_PROJECT)

```mermaid
graph TD
    A[Start Research] --> B[Phase 0: Market Timing]
    B -->|Score < 7| ABORT[Abort - Bad Timing]
    B -->|Score >= 7| C[Phase 1: Problem Discovery]
    
    C --> D[25-30 Problems Found]
    D --> E[Phase 2: Validation]
    E -->|Score >= 9/10| F[~12 Problems Pass]
    
    F --> G[Phase 2.5: User Research]
    G -->|High Urgency| H[~7 Problems Pass]
    
    H --> I[Phase 3: Competition Analysis]
    I -->|15x Better| J[~7 Opportunities]
    
    J --> K[Phase 3.5: Technical Validation]
    K -->|MVP <= 3mo| L[~4 Pass]
    
    L --> M[Phase 4: Market Sizing]
    M -->|TAM >= $1B| N[~3 Pass]
    
    N --> O[Phase 4.5: Unit Economics]
    O -->|LTV:CAC >= 3:1| P[~2 Pass]
    
    P --> Q[Phase 5: Final Selection]
    Q --> R[1 BEST Project Selected]
    
    R --> S[Save ALL Research Data]
```

### Research Data Saved:
- All 25-30 problems discovered
- Validation scores for each phase
- User stories and workflows
- Competition analysis
- Technical requirements
- Market sizing data
- Unit economics calculations

## 📁 Project Creation Flow

```mermaid
graph TD
    A[NEW_PROJECT Work Item] --> B[ProjectCreator.create_project]
    B --> C[Generate Project Details]
    
    C --> D[Fork Laravel/React Starter]
    D --> E[Create Local Repository]
    
    E --> F[Generate Documentation]
    F --> G[README.md - Quick Start Guide]
    F --> H[ARCHITECTURE.md - System Design]
    F --> I[MONETIZATION_STRATEGY.md - Business Model]
    
    G --> J[Save metadata.json]
    H --> J
    I --> J
    
    J --> K[Create Setup Tasks]
    K --> L[Database Setup]
    K --> M[Environment Config]
    K --> N[Dependencies Install]
    K --> O[Initial Tests]
    
    L --> P[Register in repositories.json]
    M --> P
    N --> P
    O --> P
    
    P --> Q[Project Ready for Development]
```

## 👷 Worker Management

```mermaid
graph TD
    A[10 Workers Total] --> B[Worker Distribution]
    
    B --> C[30% System Tasks<br/>3 Workers]
    B --> D[60% Project Tasks<br/>6 Workers]
    B --> E[10% Flexible<br/>1 Worker]
    
    C --> F[Monitoring]
    C --> G[Maintenance]
    C --> H[Research]
    
    D --> I[Feature Development]
    D --> J[Bug Fixes]
    D --> K[Documentation]
    
    E --> L[Any Available Work]
    
    M[Idle > 30s] --> N[Can Take Any Task]
```

### Worker States:
- **IDLE** - Waiting for work
- **WORKING** - Processing task
- **ERROR** - Task failed
- **SHUTTING_DOWN** - Graceful shutdown

## 🗄️ State Management

```mermaid
graph TD
    A[State Files] --> B[system_state.json]
    A --> C[task_state.json]
    A --> D[continuous_orchestrator_state.json]
    A --> E[repositories.json]
    A --> F[task_hashes.json]
    
    B --> G[Projects & Config]
    C --> H[Task Tracking]
    D --> I[Metrics & Stats]
    E --> J[Local Repositories]
    F --> K[Duplicate Detection]
    
    L[Research Knowledge] --> M[research_TIMESTAMP.json]
    L --> N[research_summary.json]
```

## 🔍 Work Discovery Sources

```mermaid
graph TD
    A[IntelligentWorkFinder] --> B[Repository Work]
    A --> C[Portfolio Gaps]
    A --> D[System Monitoring]
    A --> E[Alternative Tasks]
    
    B --> F[Issues & Features]
    C --> G[Market Research]
    D --> H[Performance & Cleanup]
    E --> I[When Duplicates Found]
    
    F --> J[Cache: 15 min]
    G --> K[Cache: 5 min]
    H --> L[No Cache]
    I --> M[Dynamic Generation]
```

## 📊 Key Decision Points

1. **No Projects? → Only NEW_PROJECT tasks**
2. **Research Score < 7? → Abort**
3. **Problem Severity < 8? → Skip**
4. **Validation < 9/10? → Skip**
5. **Not 15x Better? → Skip**
6. **TAM < $1B? → Skip**
7. **LTV:CAC < 3:1? → Skip**
8. **Final Selection → 1 Project Only**

## 🚦 System Health Monitoring

```mermaid
graph TD
    A[Continuous Monitoring] --> B[Worker Utilization]
    A --> C[Queue Depth]
    A --> D[Task Completion Rate]
    A --> E[Error Tracking]
    
    B --> F[Target: 80%+]
    C --> G[Target: 10-30 items]
    D --> H[Success Rate: 90%+]
    E --> I[Auto-retry with backoff]
```

## 💾 Local Mode Operations

```mermaid
graph TD
    A[LocalGitHubSimulator] --> B[Git Operations]
    A --> C[Issue Management]
    A --> D[Repository Registry]
    
    B --> E[Fork → Copy Template]
    B --> F[Clone → Already Local]
    
    C --> G[Issues as JSON Files]
    C --> H[Comments in Same File]
    
    D --> I[repositories.json]
    D --> J[File:// URLs]
```

## 🔄 Complete Cycle Time

```
Research: 15-30 minutes (no timeouts)
    ↓
Project Creation: 2-5 minutes
    ↓
Setup Tasks: 10-20 minutes
    ↓
Development: Continuous
    ↓
Next Research: After current project progresses
```

## 🎯 System Goals

1. **Quality > Quantity**: 1 exceptional project > 5 mediocre ones
2. **Autonomous Operation**: 24/7 without human intervention
3. **Learning System**: Saves all research for future improvements
4. **Local First**: Everything runs locally, no external dependencies
5. **Intensive Validation**: 9 phases ensure only the best projects

This ZIP hybrid system creates a sustainable, high-quality project pipeline with comprehensive validation at every step.