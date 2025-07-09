# CWMAI Minimal Runtime Package

This folder contains only the essential files needed to run:
1. `run_hybrid_continuous.py` - Hybrid continuous AI system
2. `scripts/enhanced_conversational_assistant.py` - Enhanced conversational AI assistant

## Folder Structure

```
zip/
├── README.md                          # This file
├── .env.local                         # API keys and configuration (REQUIRED)
├── CLAUDE.md                          # System instructions
├── run_hybrid_continuous.py           # Main hybrid system entry point
├── scripts/
│   ├── enhanced_conversational_assistant.py  # Conversational assistant
│   ├── [100+ Python modules]          # All required dependencies
│   ├── custom_tools/                  # Generated custom tools
│   ├── redis_integration/             # Redis integration modules
│   ├── specialized_agents/            # Specialized agent implementations
│   ├── task_type_configs/             # Task type configurations
│   ├── improvements/                  # Improvement modules
│   └── workflows/                     # Workflow definitions
├── logs/                              # Log files directory
├── *.json                             # Configuration and state files
└── *.db                               # SQLite database files
```

## Running the Scripts

### 1. Run Hybrid Continuous System (10 workers)
```bash
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10
```

### 2. Run Enhanced Conversational Assistant
```bash
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 scripts/enhanced_conversational_assistant.py
```

## Important Notes

1. **Environment Variables**: The `.env.local` file MUST be present with all required API keys
2. **Python Path**: The PYTHONPATH must be set to the cwmai directory
3. **Architecture**: Uses arm64 architecture for M1/M2 Macs
4. **Dependencies**: All Python dependencies are included in the scripts/ directory

## Key Features

### Hybrid Continuous System
- 24/7 continuous operation
- Dynamic work generation
- Intelligent task distribution
- Claude executor pool
- Pattern learning and evolution
- Multi-objective optimization

### Enhanced Conversational Assistant
- Self-improvement capabilities
- Background intelligence daemon
- Tool creation and execution
- Conversation history tracking
- Context-aware responses

## Minimal Requirements
- Python 3.x
- API keys in .env.local
- SQLite (included with Python)
- No external pip packages required (all dependencies included)