"""
Verify MCP-Redis setup is complete
"""

import os
import subprocess
import shutil
import json

def check_installation():
    """Check if MCP-Redis is properly installed"""
    print("=== MCP-Redis Setup Verification ===\n")
    
    # 1. Check npm/npx
    npx_path = shutil.which("npx")
    if npx_path:
        print(f"✓ npx found at: {npx_path}")
    else:
        print("✗ npx not found in PATH")
        return False
    
    # 2. Check MCP-Redis server installation
    try:
        result = subprocess.run(
            ["npx", "-y", "@modelcontextprotocol/server-redis", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # The server doesn't have a --version flag, so we check if it tries to connect
        if "REDIS_URL" in result.stderr or "TypeError" in result.stderr:
            print("✓ MCP-Redis server is installed")
        else:
            print("✗ MCP-Redis server not found")
            return False
    except Exception as e:
        print(f"✗ Error checking MCP-Redis: {e}")
        return False
    
    # 3. Check integration files
    files_to_check = [
        ("scripts/mcp_redis_integration.py", "MCP-Redis integration module"),
        ("scripts/mcp_config.py", "MCP configuration (with Redis)"),
        ("MCP_REDIS_INTEGRATION.md", "Documentation")
    ]
    
    all_files_present = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description} missing: {file_path}")
            all_files_present = False
    
    # 4. Check redis_work_queue.py integration
    with open("scripts/redis_work_queue.py", "r") as f:
        content = f.read()
        if "MCPRedisIntegration" in content:
            print("✓ redis_work_queue.py has MCP-Redis integration")
        else:
            print("✗ redis_work_queue.py missing MCP-Redis integration")
            all_files_present = False
    
    # 5. Check environment setup
    print("\n=== Environment Configuration ===")
    use_mcp = os.getenv("USE_MCP_REDIS", "false")
    redis_url = os.getenv("REDIS_URL", "not set")
    
    print(f"USE_MCP_REDIS: {use_mcp}")
    print(f"REDIS_URL: {redis_url}")
    
    if use_mcp.lower() == "true":
        print("✓ MCP-Redis is enabled")
    else:
        print("✗ MCP-Redis is disabled (set USE_MCP_REDIS=true to enable)")
    
    # 6. Show example usage
    print("\n=== Usage Instructions ===")
    print("1. Start Redis server:")
    print("   docker run -d -p 6379:6379 redis:latest")
    print("   # or: redis-server")
    print("\n2. Enable MCP-Redis in .env.local:")
    print("   USE_MCP_REDIS=true")
    print("   REDIS_URL=redis://localhost:6379")
    print("\n3. Run your application:")
    print("   export $(cat .env.local | grep -v '^#' | xargs)")
    print("   python run_continuous_ai.py")
    
    print("\n=== Integration Features ===")
    print("When MCP-Redis is enabled, you get:")
    print("- find_similar_tasks(): AI-powered similarity search")
    print("- optimize_task_assignment(): Intelligent task routing")
    print("- get_intelligent_queue_insights(): Advanced analytics")
    print("- Natural language Redis operations")
    
    # 7. Summary
    print("\n=== Setup Status ===")
    if all_files_present and npx_path:
        print("✓ MCP-Redis integration is properly set up!")
        print("✓ The system will use MCP-Redis when USE_MCP_REDIS=true and Redis is running")
        return True
    else:
        print("✗ Some components are missing. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = check_installation()
    exit(0 if success else 1)