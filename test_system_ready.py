#!/usr/bin/env python3
"""
Test System Readiness

Simple test to verify the self-amplifying system is ready to run.
"""

import os
import sys
import json

# Check if we can load the configuration
def test_configuration():
    """Test if configuration is properly set."""
    print("Checking configuration...")
    
    try:
        # Read the research evolution engine file
        config_found = {
            "enable_fixed_interval": False,
            "enable_proactive_research": False,
            "cycle_interval_seconds": 0,
            "max_research_per_cycle": 0
        }
        
        with open("scripts/research_evolution_engine.py", "r") as f:
            content = f.read()
            
            # Check key configurations
            if '"enable_fixed_interval": True' in content:
                config_found["enable_fixed_interval"] = True
            if '"enable_proactive_research": True' in content:
                config_found["enable_proactive_research"] = True
            if '"cycle_interval_seconds": 20 * 60' in content:
                config_found["cycle_interval_seconds"] = 20 * 60
            if '"max_research_per_cycle": 8' in content:
                config_found["max_research_per_cycle"] = 8
        
        print(f"  ✓ Continuous learning: {config_found['enable_fixed_interval']}")
        print(f"  ✓ Proactive research: {config_found['enable_proactive_research']}")
        print(f"  ✓ Cycle interval: {config_found['cycle_interval_seconds'] / 60} minutes")
        print(f"  ✓ Research per cycle: {config_found['max_research_per_cycle']}")
        
        all_good = all([
            config_found["enable_fixed_interval"],
            config_found["enable_proactive_research"],
            config_found["cycle_interval_seconds"] == 20 * 60,
            config_found["max_research_per_cycle"] == 8
        ])
        
        return all_good
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_directories():
    """Test if required directories exist."""
    print("\nChecking directories...")
    
    dirs = [
        "research_knowledge",
        "research_knowledge/raw_research", 
        "research_knowledge/processed_insights",
        "research_knowledge/metadata",
        "logs"
    ]
    
    all_exist = True
    for d in dirs:
        exists = os.path.exists(d)
        print(f"  {'✓' if exists else '✗'} {d}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_startup_script():
    """Test if startup script exists."""
    print("\nChecking startup script...")
    
    script = "start_self_amplifying_ai.py"
    exists = os.path.exists(script)
    is_executable = os.access(script, os.X_OK) if exists else False
    
    print(f"  {'✓' if exists else '✗'} {script} exists")
    print(f"  {'✓' if is_executable else '✗'} {script} is executable")
    
    return exists


def test_environment():
    """Test environment readiness."""
    print("\nChecking environment...")
    
    # Check for API keys (without exposing them)
    api_keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY")
    }
    
    any_key = False
    for key_name, key_value in api_keys.items():
        has_key = bool(key_value)
        if has_key:
            any_key = True
        print(f"  {'✓' if has_key else '✗'} {key_name} {'set' if has_key else 'not set'}")
    
    return any_key


def main():
    """Run all readiness tests."""
    print("=" * 60)
    print("CWMAI Self-Amplifying System Readiness Check")
    print("=" * 60)
    
    tests = {
        "Configuration": test_configuration(),
        "Directories": test_directories(),
        "Startup Script": test_startup_script(),
        "Environment": test_environment()
    }
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = sum(tests.values())
    total = len(tests)
    
    for test_name, result in tests.items():
        print(f"{test_name}: {'✅ PASS' if result else '❌ FAIL'}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SYSTEM READY!")
        print("Start with: python start_self_amplifying_ai.py")
    elif tests["Configuration"] and tests["Directories"] and tests["Startup Script"]:
        print("\n⚠️  SYSTEM READY (No API keys)")
        print("The system will run but AI features will be limited.")
        print("Start with: python start_self_amplifying_ai.py")
    else:
        print("\n❌ SYSTEM NOT READY")
        print("Fix the issues above before starting.")
    
    # Create a simple status file
    status = {
        "ready": passed >= 3,  # At least config, dirs, and script
        "timestamp": os.popen('date').read().strip(),
        "tests": tests
    }
    
    with open("self_amplifying_status.json", "w") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()