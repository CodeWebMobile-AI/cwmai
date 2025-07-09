#!/bin/bash
# Comprehensive test runner for research-evolution system with ARM64

echo "================================================================================"
echo "🧪 TESTING RESEARCH-EVOLUTION INTEGRATION WITH ARM64"
echo "================================================================================"

# Function to run a command with proper ARM64 setup
run_arm64() {
    export $(cat .env.local | grep -v '^#' | xargs)
    export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        arch -arm64 python "$@"
    else
        arch -arm64 python3 "$@"
    fi
}

# Test 1: Verify consolidation
echo -e "\n📋 TEST 1: Verifying Research Consolidation"
echo "----------------------------------------"
run_arm64 test_research_simple.py

# Test 2: Quick integration test (1 minute)
echo -e "\n\n🔄 TEST 2: Quick Integration Test (1 minute)"
echo "----------------------------------------"
echo "Running knowledge-driven evolution for 1 minute..."
run_arm64 run_knowledge_driven_evolution.py --minutes 1

# Test 3: Research functionality test
echo -e "\n\n🔬 TEST 3: Testing Research Functionality"
echo "----------------------------------------"
run_arm64 -c "
import sys
sys.path.insert(0, '.')
try:
    from scripts.research_evolution_engine import ResearchEvolutionEngine
    from scripts.ai_brain_factory import AIBrainFactory
    
    # Test initialization
    ai_brain = AIBrainFactory.create_for_research()
    engine = ResearchEvolutionEngine(
        state_manager=None,
        ai_brain=ai_brain,
        task_generator=None,
        self_improver=None,
        outcome_learning=None
    )
    
    # Test get_research_state
    state = engine.get_research_state()
    print('✅ Research engine initialized successfully')
    print(f'   Cycles completed: {state[\"cycles_completed\"]}')
    print(f'   Is running: {state[\"is_running\"]}')
    
except Exception as e:
    print(f'❌ Research engine test failed: {e}')
    import traceback
    traceback.print_exc()
"

# Test 4: Full 15-minute test (if requested)
echo -e "\n\n================================================================================"
echo "📊 TEST SUMMARY"
echo "================================================================================"
echo "All quick tests completed. The system is configured to:"
echo "  • Use ARM64 architecture with virtual environment"
echo "  • Run research as part of knowledge-driven evolution"
echo "  • Redirect research tasks from continuous orchestrators"
echo ""
echo "To run a full 15-minute test, execute:"
echo "  ./run_arm64_test.sh"
echo ""
echo "Or directly:"
echo "  source venv/bin/activate"
echo "  export \$(cat .env.local | grep -v '^#' | xargs)"
echo "  export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai"
echo "  arch -arm64 python run_knowledge_driven_evolution.py --minutes 15"
echo "================================================================================"