#!/usr/bin/env python3
"""Test direct control of continuous AI system."""

import asyncio
import sys
sys.path.insert(0, '.')

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def test_control():
    assistant = ConversationalAIAssistant()
    
    print("Testing Continuous AI Control Methods\n")
    
    # Test 1: Check status
    print("1. Checking system status...")
    status = await assistant.check_continuous_ai_status()
    print(f"   Running: {status['running']}")
    if status['running']:
        print(f"   PID: {status.get('pid', 'Unknown')}")
        print(f"   Workers: {status.get('workers', 'Unknown')}")
    print()
    
    # Test 2: Monitor health
    if status['running']:
        print("2. Monitoring system health...")
        health = await assistant.monitor_system_health()
        print(f"   Health Score: {health['health_score']}/100")
        print(f"   Status: {health['status']}")
        if health['issues']:
            print(f"   Issues: {', '.join(health['issues'])}")
        if health['recommendations']:
            print(f"   Recommendations: {health['recommendations'][0]}")

if __name__ == "__main__":
    asyncio.run(test_control())