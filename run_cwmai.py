#!/usr/bin/env python3
"""
Run CWMAI - Simple launcher for the continuous AI system
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.continuous_orchestrator import ContinuousOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)




async def run_orchestrator():
    """Run the continuous orchestrator."""
    print("\n🤖 Starting Continuous AI Orchestrator...")
    
    orchestrator = ContinuousOrchestrator(
        max_workers=2,
        enable_parallel=True,
        enable_research=False,
        enable_round_robin=True
    )
    
    print("📊 Configuration:")
    print(f"   - Max workers: {orchestrator.max_workers}")
    print(f"   - Parallel processing: {orchestrator.enable_parallel}")
    print(f"   - Round-robin AI: {orchestrator.enable_round_robin}")
    print(f"   - Repositories: {len(orchestrator.system_state.get('repositories', []))}")
    
    print("\n🚀 Running continuous orchestration (press Ctrl+C to stop)...")
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping orchestrator...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


async def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                    CWMAI - 24/7 AI System                 ║
║              Continuous Work Management AI                ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check environment
    providers = []
    if os.getenv('ANTHROPIC_API_KEY'): providers.append('Anthropic')
    if os.getenv('OPENAI_API_KEY'): providers.append('OpenAI')
    if os.getenv('GEMINI_API_KEY'): providers.append('Gemini')
    if os.getenv('DEEPSEEK_API_KEY'): providers.append('DeepSeek')
    
    print(f"✅ Available AI providers: {', '.join(providers) if providers else 'None'}")
    print(f"✅ GitHub integration: {'Configured' if os.getenv('GITHUB_TOKEN') else 'Not configured'}")
    print(f"✅ MCP support: {'Available' if os.path.exists('mcp_config.json') else 'Not configured'}")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "mcp":
            print("\n🔧 MCP Status:")
            print("   - Infrastructure: ✅ Installed")
            print("   - Fallback mode: ✅ Available")
            print("   - Full integration: ⏳ Requires MCP protocol setup")
        else:
            print(f"\nUnknown command: {sys.argv[1]}")
            print("Usage: python run_cwmai.py [mcp]")
    else:
        await run_orchestrator()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)