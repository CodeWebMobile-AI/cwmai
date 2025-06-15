#!/usr/bin/env python3
"""
Fix JSON serialization issues by identifying and removing function references.
"""

import json
import sys
sys.path.append('/workspaces/cwmai')

def check_state_file():
    """Check system_state.json for issues."""
    print("Checking system_state.json for function references...")
    
    try:
        with open('/workspaces/cwmai/system_state.json', 'r') as f:
            content = f.read()
            
        # Look for function references in the content
        if '<function' in content:
            print("⚠️  Found function references in system_state.json!")
            
            # Find all occurrences
            import re
            functions = re.findall(r'"[^"]*":\s*"<function[^>]+>', content)
            
            for func in functions:
                print(f"  - {func}")
                
            # Load and clean the JSON
            state = json.loads(content)
            cleaned_state = clean_functions(state)
            
            # Save cleaned version
            with open('/workspaces/cwmai/system_state_cleaned.json', 'w') as f:
                json.dump(cleaned_state, f, indent=2)
                
            print("\n✅ Cleaned state saved to system_state_cleaned.json")
            print("   Review the cleaned file and replace system_state.json if correct")
            
        else:
            print("✅ No function references found in system_state.json")
            
    except Exception as e:
        print(f"Error: {e}")


def clean_functions(obj):
    """Recursively clean function references from object."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if isinstance(value, str) and '<function' in value:
                print(f"  Removing function reference: {key} = {value[:50]}...")
                # Skip this entry or replace with placeholder
                cleaned[key] = "removed_function_reference"
            else:
                cleaned[key] = clean_functions(value)
        return cleaned
    elif isinstance(obj, list):
        return [clean_functions(item) for item in obj]
    else:
        return obj


def check_redis_issues():
    """Check for Redis-related issues."""
    print("\nChecking recent Redis errors...")
    
    import subprocess
    
    # Check circuit breaker issues
    circuit_breaker = subprocess.check_output(
        ['grep', '-c', 'Circuit breaker', '/workspaces/cwmai/continuous_ai.log'],
        text=True
    ).strip()
    
    # Check JSON serialization errors
    json_errors = subprocess.check_output(
        ['grep', '-c', 'not JSON serializable', '/workspaces/cwmai/continuous_ai.log'],
        text=True
    ).strip()
    
    print(f"Circuit breaker errors: {circuit_breaker}")
    print(f"JSON serialization errors: {json_errors}")
    
    if int(circuit_breaker) > 10:
        print("\n⚠️  High number of circuit breaker errors - Redis connection may be unstable")
        print("   Consider restarting Redis or checking connection settings")
        
    if int(json_errors) > 5:
        print("\n⚠️  Multiple JSON serialization errors detected")
        print("   This usually means functions or non-serializable objects in state")


if __name__ == "__main__":
    check_state_file()
    check_redis_issues()