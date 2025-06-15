#!/usr/bin/env python3
"""
Analyze Redis Connection Sources

This script analyzes which parts of the codebase are creating Redis connections.
"""

import os
import re
from collections import defaultdict
from pathlib import Path


def find_redis_connections():
    """Find all Redis connection creation points in the codebase."""
    connection_patterns = [
        r'get_redis_client\s*\(',
        r'RedisClient\s*\(',
        r'redis\.Redis\s*\(',
        r'RedisCluster\s*\(',
        r'\.pubsub\s*\(',
        r'get_connection_pool\s*\(',
    ]
    
    results = defaultdict(list)
    
    # Search Python files
    for root, dirs, files in os.walk('/workspaces/cwmai'):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines, 1):
                            for pattern in connection_patterns:
                                if re.search(pattern, line):
                                    results[str(filepath)].append({
                                        'line': i,
                                        'code': line.strip(),
                                        'pattern': pattern
                                    })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return results


def analyze_pubsub_usage():
    """Find all PubSub subscription patterns."""
    pubsub_patterns = [
        r'\.subscribe\s*\(',
        r'\.psubscribe\s*\(',
        r'\.unsubscribe\s*\(',
        r'\.punsubscribe\s*\(',
    ]
    
    results = defaultdict(list)
    
    for root, dirs, files in os.walk('/workspaces/cwmai/scripts'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines, 1):
                            for pattern in pubsub_patterns:
                                if re.search(pattern, line):
                                    results[str(filepath)].append({
                                        'line': i,
                                        'code': line.strip(),
                                        'pattern': pattern
                                    })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return results


def main():
    """Analyze connection sources."""
    print("Analyzing Redis connection sources...\n")
    
    # Find connection creation points
    connections = find_redis_connections()
    
    print("=== Redis Connection Creation Points ===")
    total_connections = 0
    for filepath, matches in sorted(connections.items()):
        if matches:
            print(f"\n{filepath}:")
            for match in matches:
                print(f"  Line {match['line']}: {match['code']}")
                total_connections += 1
    
    print(f"\nTotal connection creation points: {total_connections}")
    
    # Find PubSub usage
    pubsub = analyze_pubsub_usage()
    
    print("\n\n=== PubSub Usage ===")
    total_pubsub = 0
    for filepath, matches in sorted(pubsub.items()):
        if matches:
            print(f"\n{filepath}:")
            for match in matches:
                print(f"  Line {match['line']}: {match['code']}")
                total_pubsub += 1
    
    print(f"\nTotal PubSub operations: {total_pubsub}")
    
    # Summary
    print("\n\n=== Summary ===")
    print(f"Files with Redis connections: {len(connections)}")
    print(f"Files with PubSub operations: {len(pubsub)}")
    
    # Identify potential issues
    print("\n=== Potential Issues ===")
    
    # Check for multiple connections in same file
    for filepath, matches in connections.items():
        if len(matches) > 2:
            print(f"⚠️  {filepath} creates {len(matches)} connections - possible leak")
    
    # Check for PubSub without unsubscribe
    for filepath, matches in pubsub.items():
        subscribes = sum(1 for m in matches if 'subscribe(' in m['code'] and 'unsubscribe' not in m['code'])
        unsubscribes = sum(1 for m in matches if 'unsubscribe(' in m['code'])
        
        if subscribes > unsubscribes:
            print(f"⚠️  {filepath} has {subscribes} subscribes but only {unsubscribes} unsubscribes")


if __name__ == "__main__":
    main()