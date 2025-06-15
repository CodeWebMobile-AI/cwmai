#!/usr/bin/env python3
"""
System Reset Script

Completely cleans up all system data and state for fresh testing.
Removes all logs, state files, cache data, and research knowledge while preserving code.
"""

import os
import shutil
import sys
import time
import signal
import subprocess
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime


def print_banner(preserve_cache=False, preserve_knowledge=False):
    """Print reset banner."""
    preserve_mode = ""
    if preserve_cache or preserve_knowledge:
        preserve_mode = f"\n║  🔒 PRESERVE MODE:                                              ║"
        if preserve_cache:
            preserve_mode += f"\n║     • AI Response Cache                                         ║"
        if preserve_knowledge:
            preserve_mode += f"\n║     • Research Knowledge Base                                   ║"
            preserve_mode += f"\n║     • Worker Capabilities                                       ║"
            preserve_mode += f"\n║     • Intelligence Hub Data                                     ║"
            preserve_mode += f"\n║     • External Agent Cache                                      ║"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    CWMAI SYSTEM RESET TOOL                      ║
║                                                                  ║
║  Cleans up all system data for fresh testing:                   ║
║  • System state files                                            ║
║  • Execution logs and results                                    ║
║  • Research knowledge base                                       ║
║  • AI response cache                                             ║
║  • Task history and analysis                                     ║
║  • Intelligence hub data                                         ║
║  • Redis data and queues                                         ║
║  • Python cache files                                            ║
║  • All temporary and generated files                             ║{preserve_mode}
║                                                                  ║
║  ⚠️  This will permanently delete all accumulated data!          ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def kill_running_processes():
    """Kill any running CWMAI processes."""
    print("🔪 Stopping any running CWMAI processes...")
    
    processes_to_kill = [
        "run_dynamic_ai.py",
        "run_continuous_ai.py",
        "run_self_improver.py",
        "python.*scripts",
        "god_mode",
        "main_cycle",
        "task_manager",
        "continuous_orchestrator",
        "research_evolution",
        "worker_intelligence"
        # Note: Removed redis-server to keep Redis running
    ]
    
    killed_count = 0
    for process_pattern in processes_to_kill:
        try:
            result = subprocess.run(
                ["pkill", "-f", process_pattern], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                killed_count += 1
        except Exception:
            pass  # pkill not available or no processes found
    
    if killed_count > 0:
        print(f"✅ Stopped {killed_count} running processes")
        time.sleep(2)  # Give processes time to shut down
    else:
        print("✅ No running processes found")


async def reset_redis_async(preserve_cache=False, preserve_knowledge=False):
    """Async Redis reset with proper circuit breaker handling."""
    try:
        from scripts.redis_integration.redis_client import RedisClient
        
        # Create Redis client
        redis_client = RedisClient()
        
        # Connect the client
        await redis_client.connect()
        print("✅ Redis client connected")
        
        # Reset circuit breaker if present
        if hasattr(redis_client, 'circuit_breaker'):
            if redis_client.circuit_breaker.state == 'open':
                print("  ⚠️  Circuit breaker is OPEN - resetting...")
                redis_client.circuit_breaker.state = 'closed'
                redis_client.circuit_breaker.failure_count = 0
                print("  ✅ Circuit breaker reset to CLOSED")
        
        # Patterns to preserve
        preserve_patterns = []
        if preserve_cache:
            preserve_patterns.append("ai_response_cache:*")
        if preserve_knowledge:
            preserve_patterns.extend([
                "intelligence_hub_*",
                "worker_capabilities:*",
                "research_knowledge:*",
                "external_agent_cache:*"
            ])
        
        # Clear data selectively
        async with redis_client.get_connection() as conn:
            if preserve_patterns:
                # Selective clear - preserve certain patterns
                print("🔒 Preserving selected Redis data...")
                
                # Get all keys
                all_keys = []
                async for key in conn.scan_iter():
                    # Check if key matches any preserve pattern
                    should_preserve = False
                    for pattern in preserve_patterns:
                        if pattern.endswith('*'):
                            prefix = pattern[:-1]
                            if key.startswith(prefix):
                                should_preserve = True
                                break
                    
                    if not should_preserve:
                        all_keys.append(key)
                
                # Delete non-preserved keys in batches
                if all_keys:
                    for i in range(0, len(all_keys), 1000):
                        batch = all_keys[i:i+1000]
                        await conn.delete(*batch)
                    print(f"✅ Cleared {len(all_keys)} Redis keys while preserving {len(preserve_patterns)} patterns")
                else:
                    print("✅ No Redis keys to clear (all preserved)")
            else:
                # Full clear
                await conn.flushall()
                print("✅ Redis data cleared")
            
            # Always clear work queue streams
            work_queues = [
                "cwmai:work_queue:critical",
                "cwmai:work_queue:high", 
                "cwmai:work_queue:medium",
                "cwmai:work_queue:low",
                "cwmai:work_queue:background"
            ]
            
            for queue in work_queues:
                try:
                    await conn.delete(queue)
                except:
                    pass
            
            # Clear execution state patterns (always clear these)
            execution_patterns = [
                "tasks:*",
                "cwmai:circuit_breaker:*",
                "redis:*",
                "worker:status:*",
                "worker:current_task:*"
            ]
            
            for pattern in execution_patterns:
                try:
                    async for key in conn.scan_iter(match=pattern):
                        await conn.delete(key)
                except:
                    pass
        
        print("✅ Redis queues and execution state cleared")
        
        # Test the connection
        try:
            pong = await redis_client.ping()
            if pong:
                print("✅ Redis connection verified")
        except Exception as e:
            print(f"⚠️  Connection test failed: {e}")
        
        await redis_client.disconnect()
        return True
        
    except ImportError:
        print("ℹ️  Redis async client not available - falling back to sync reset")
        return False
    except Exception as e:
        print(f"⚠️  Async Redis reset failed: {e}")
        return False


def reset_redis(preserve_cache=False, preserve_knowledge=False):
    """Reset Redis data and state."""
    print("🔄 Resetting Redis...")
    
    # Try async reset first
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # Run async reset
        success = loop.run_until_complete(reset_redis_async(preserve_cache, preserve_knowledge))
        if success:
            return
    except Exception as e:
        print(f"⚠️  Async reset failed: {e}")
    
    # Fall back to sync Redis reset
    try:
        import redis
        
        # Try to connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test connection
        r.ping()
        
        # Patterns to preserve
        preserve_patterns = []
        if preserve_cache:
            preserve_patterns.append("ai_response_cache:*")
        if preserve_knowledge:
            preserve_patterns.extend([
                "intelligence_hub_*",
                "worker_capabilities:*",
                "research_knowledge:*",
                "external_agent_cache:*"
            ])
        
        if preserve_patterns:
            # Selective clear
            print("🔒 Preserving selected Redis data...")
            all_keys = []
            for key in r.scan_iter():
                should_preserve = False
                for pattern in preserve_patterns:
                    if pattern.endswith('*'):
                        prefix = pattern[:-1]
                        if key.startswith(prefix):
                            should_preserve = True
                            break
                if not should_preserve:
                    all_keys.append(key)
            
            # Delete in batches
            if all_keys:
                for i in range(0, len(all_keys), 1000):
                    batch = all_keys[i:i+1000]
                    r.delete(*batch)
                print(f"✅ Cleared {len(all_keys)} Redis keys while preserving patterns")
        else:
            # Clear all Redis data
            r.flushall()
            print("✅ Redis data cleared")
        
        # Reset circuit breaker if it exists
        circuit_breaker_key = "cwmai:circuit_breaker:state"
        r.delete(circuit_breaker_key)
        
        # Clear work queue streams
        work_queues = [
            "cwmai:work_queue:critical",
            "cwmai:work_queue:high", 
            "cwmai:work_queue:medium",
            "cwmai:work_queue:low",
            "cwmai:work_queue:background"
        ]
        
        for queue in work_queues:
            try:
                r.delete(queue)
            except:
                pass
                
        print("✅ Redis queues cleared")
        
    except ImportError:
        print("ℹ️  Redis library not installed - skipping Redis reset")
    except Exception as e:
        print(f"⚠️  Could not reset Redis: {e}")
        print("   Redis may not be running or accessible")


def create_backup(backup_important=False):
    """Create backup of important data if requested."""
    if not backup_important:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    print(f"💾 Creating backup in {backup_dir}/...")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup important files
    important_files = [
        "system_state.json",
        "task_history.json", 
        "task_analysis.json",
        "context.json"
    ]
    
    backed_up = 0
    for file in important_files:
        if os.path.exists(file):
            try:
                shutil.copy2(file, backup_dir)
                backed_up += 1
            except Exception as e:
                print(f"⚠️  Failed to backup {file}: {e}")
    
    if backed_up > 0:
        print(f"✅ Backed up {backed_up} files to {backup_dir}/")
        return backup_dir
    else:
        # Remove empty backup directory
        try:
            os.rmdir(backup_dir)
        except:
            pass
        print("ℹ️  No important files to backup")
        return None


def clean_python_cache():
    """Remove all Python cache files and directories."""
    print("🧹 Cleaning Python cache files...")
    
    removed_dirs = 0
    removed_files = 0
    
    # Find and remove all __pycache__ directories
    for root, dirs, files in os.walk("."):
        # Skip .git directory
        if ".git" in root:
            continue
            
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                removed_dirs += 1
            except Exception as e:
                print(f"⚠️  Error removing {pycache_path}: {e}")
    
    # Remove any stray .pyc files
    for pyc_file in Path(".").rglob("*.pyc"):
        if ".git" not in str(pyc_file):
            try:
                pyc_file.unlink()
                removed_files += 1
            except Exception as e:
                print(f"⚠️  Error removing {pyc_file}: {e}")
    
    if removed_dirs > 0 or removed_files > 0:
        print(f"✅ Removed {removed_dirs} __pycache__ directories and {removed_files} .pyc files")
    else:
        print("ℹ️  No Python cache files found")


def remove_files_safely(files_list, description):
    """Safely remove a list of files."""
    print(f"🗑️  Removing {description}...")
    removed_count = 0
    
    for file_pattern in files_list:
        if '*' in file_pattern:
            # Handle wildcards using Path.glob
            try:
                parent = Path(file_pattern).parent
                pattern = Path(file_pattern).name
                for file_path in parent.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        removed_count += 1
            except Exception as e:
                print(f"⚠️  Error removing {file_pattern}: {e}")
        else:
            # Handle specific files
            if os.path.exists(file_pattern):
                try:
                    if os.path.isfile(file_pattern):
                        os.remove(file_pattern)
                        removed_count += 1
                except Exception as e:
                    print(f"⚠️  Error removing {file_pattern}: {e}")
    
    if removed_count > 0:
        print(f"✅ Removed {removed_count} files")
    else:
        print("ℹ️  No files found to remove")


def remove_directories_safely(dirs_list, description):
    """Safely remove a list of directories."""
    print(f"📁 Removing {description}...")
    removed_count = 0
    
    for dir_path in dirs_list:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                removed_count += 1
                print(f"  ✓ Removed {dir_path}")
            except Exception as e:
                print(f"⚠️  Error removing {dir_path}: {e}")
    
    if removed_count > 0:
        print(f"✅ Removed {removed_count} directories")
    else:
        print("ℹ️  No directories found to remove")


def reset_system_data(preserve_cache=False, preserve_knowledge=False):
    """Remove all system data files."""
    # Root directory files
    root_files = [
        "*.json",
        "*.log", 
        "*.pkl",
        "*.cache",
        "*.html",
        "test_state.json",
        "orchestrator_state.json",
        "workflow_context.json",
        "production_context.json",
        "continuous_orchestrator_state.json",
        "system_state.json",
        "task_history.json",
        "task_state.json",
        "self_amplifying_status.json",
        "context.json",
        "continuous_ai.log",
        "continuous_ai_output.log",
        "continuous_ai_output_new.log",
        "self_improvement_*.log"
    ]
    remove_files_safely(root_files, "root directory data files")
    
    # Scripts directory files  
    scripts_files = [
        "scripts/*.json",
        "scripts/*.log",
        "scripts/*.pkl",
        "scripts/*.cache",
        "scripts/*.html",
        "scripts/research_intelligence_dashboard.html",
        "scripts/research_activation_report.json",
        "scripts/self_amplifying_activation_summary.txt"
    ]
    remove_files_safely(scripts_files, "scripts directory data files")
    
    # Directories to remove
    directories = [
        "state_backups",
        "logs",
        ".integration_cache",
        ".self_improver",
        "docker",  # Empty directory
        "config"   # Empty directory
    ]
    
    # Conditionally add directories based on preserve flags
    if not preserve_knowledge:
        directories.extend([
            "research_knowledge",
            "test_research_knowledge",
            "worker_capabilities",
            ".external_agent_cache"
        ])
    else:
        print("🔒 Preserving knowledge directories:")
        print("   • research_knowledge")
        print("   • test_research_knowledge") 
        print("   • worker_capabilities")
        print("   • .external_agent_cache")
    
    # Handle timestamped backup directories
    for backup_dir in Path(".").glob("backup_20*"):
        if backup_dir.is_dir():
            directories.append(str(backup_dir))
    
    remove_directories_safely(directories, "data directories")


def verify_cleanup():
    """Verify the cleanup was successful."""
    print("🔍 Verifying cleanup...")
    
    # Check for remaining data files
    remaining_files = []
    
    # Check root directory
    for ext in ["*.json", "*.log", "*.pkl", "*.cache", "*.html"]:
        files = list(Path(".").glob(ext))
        # Exclude essential files
        files = [f for f in files if f.name not in ["requirements.txt", "LICENSE", ".gitignore"]]
        remaining_files.extend(files)
    
    # Check scripts directory
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        for ext in ["*.json", "*.log", "*.pkl", "*.cache", "*.html", "*.txt"]:
            remaining_files.extend(list(scripts_dir.glob(ext)))
    
    # Check for data directories
    data_dirs = [
        "research_knowledge", 
        "test_research_knowledge",
        "state_backups", 
        "logs",
        ".external_agent_cache",
        ".integration_cache",
        ".self_improver"
    ]
    remaining_dirs = [d for d in data_dirs if os.path.exists(d)]
    
    # Check for backup directories
    backup_dirs = list(Path(".").glob("backup_20*"))
    remaining_dirs.extend([str(d) for d in backup_dirs if d.is_dir()])
    
    # Check for __pycache__ directories
    pycache_dirs = list(Path(".").rglob("__pycache__"))
    pycache_dirs = [d for d in pycache_dirs if ".git" not in str(d)]
    
    if remaining_files:
        print(f"⚠️  Found {len(remaining_files)} remaining data files:")
        for f in remaining_files[:10]:  # Show first 10
            print(f"    {f}")
        if len(remaining_files) > 10:
            print(f"    ... and {len(remaining_files) - 10} more")
    
    if remaining_dirs:
        print(f"⚠️  Found {len(remaining_dirs)} remaining data directories:")
        for d in remaining_dirs[:10]:
            print(f"    {d}")
        if len(remaining_dirs) > 10:
            print(f"    ... and {len(remaining_dirs) - 10} more")
    
    if pycache_dirs:
        print(f"⚠️  Found {len(pycache_dirs)} remaining __pycache__ directories")
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        key_count = r.dbsize()
        if key_count > 0:
            print(f"⚠️  Redis still has {key_count} keys")
    except:
        pass  # Redis not available
    
    if not remaining_files and not remaining_dirs and not pycache_dirs:
        print("✅ Cleanup verification successful - system is clean!")
        return True
    else:
        print("⚠️  Some files/directories remain - manual cleanup may be needed")
        return False


def verify_code_intact():
    """Verify that code files are still intact."""
    print("🔍 Verifying code files are intact...")
    
    # Check for essential Python files
    essential_files = [
        "run_dynamic_ai.py",
        "scripts/ai_brain.py",
        "scripts/production_orchestrator.py",
        "scripts/swarm_intelligence.py",
        "scripts/ai_response_cache.py",
        "scripts/async_state_manager.py",
        "scripts/intelligence_hub.py"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    else:
        print("✅ All essential code files are intact!")
        return True


def main():
    """Main reset function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CWMAI System Reset Tool - Clean up system data for fresh testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full reset (removes all data)
  python reset_system.py
  
  # Preserve learned data but clear execution state
  python reset_system.py --preserve-cache --preserve-knowledge
  
  # Preserve only AI response cache
  python reset_system.py --preserve-cache
  
  # Non-interactive mode with auto-yes
  python reset_system.py -y
  
  # Non-interactive with backup
  python reset_system.py -y --backup
        """
    )
    
    parser.add_argument(
        '--preserve-cache',
        action='store_true',
        help='Preserve AI response cache to save API costs'
    )
    
    parser.add_argument(
        '--preserve-knowledge',
        action='store_true',
        help='Preserve research knowledge, worker capabilities, and intelligence data'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompts (non-interactive mode)'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of important files before reset'
    )
    
    args = parser.parse_args()
    
    # Print banner with preserve flags
    print_banner(args.preserve_cache, args.preserve_knowledge)
    
    # Confirmation prompt (unless -y flag is used)
    if not args.yes:
        if args.preserve_cache or args.preserve_knowledge:
            print("This will delete system execution state while preserving selected data.")
        else:
            print("This will permanently delete all system data and accumulated intelligence.")
        print("Code files will be preserved.")
        print()
        
        response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Reset cancelled")
            return
        
        # Optional backup (if not specified via --backup)
        if not args.backup:
            backup_response = input("Create backup of important files first? (y/n): ").strip().lower()
            create_backup_flag = backup_response in ['y', 'yes']
        else:
            create_backup_flag = True
    else:
        # Non-interactive mode
        create_backup_flag = args.backup
    
    print("\n" + "="*60)
    print("STARTING SYSTEM RESET")
    if args.preserve_cache or args.preserve_knowledge:
        print("MODE: Selective Reset (Preserving Data)")
    else:
        print("MODE: Full Reset")
    print("="*60)
    
    # Track statistics
    start_time = time.time()
    
    try:
        # Step 1: Stop running processes
        kill_running_processes()
        
        # Step 2: Clean up deleted repositories first
        print("🧹 Running cleanup for deleted repositories...")
        try:
            cleanup_result = subprocess.run(
                ["python3", "cleanup_startup.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if cleanup_result.returncode == 0:
                print("✅ Deleted repositories cleaned up")
            else:
                print("⚠️  Cleanup script failed, continuing with reset...")
        except subprocess.TimeoutExpired:
            print("⚠️  Cleanup script timed out, continuing with reset...")
        except Exception as e:
            print(f"⚠️  Could not run cleanup script: {e}")
        
        # Step 3: Create backup if requested
        backup_dir = create_backup(create_backup_flag)
        
        # Step 4: Reset Redis
        reset_redis(args.preserve_cache, args.preserve_knowledge)
        
        # Step 5: Clean Python cache
        clean_python_cache()
        
        # Step 6: Reset system data
        reset_system_data(args.preserve_cache, args.preserve_knowledge)
        
        # Step 7: Verify cleanup
        cleanup_success = verify_cleanup()
        
        # Step 8: Verify code is intact
        code_intact = verify_code_intact()
        
        # Calculate duration
        duration = time.time() - start_time
        
        print("\n" + "="*60)
        print("RESET COMPLETE")
        print("="*60)
        
        if cleanup_success and code_intact:
            print("🎉 System reset successful!")
            print(f"⏱️  Reset completed in {duration:.2f} seconds")
            
            if args.preserve_cache or args.preserve_knowledge:
                print("\n🔒 Preserved data:")
                if args.preserve_cache:
                    print("  • AI Response Cache")
                if args.preserve_knowledge:
                    print("  • Research Knowledge Base")
                    print("  • Worker Capabilities")
                    print("  • Intelligence Hub Data")
                    print("  • External Agent Cache")
            
            print("\nYour system is now clean and ready for fresh testing.")
            
            if backup_dir:
                print(f"\n💾 Backup available in: {backup_dir}/")
            
            print("\n📝 Summary of cleaned items:")
            print("  • All system state files")
            print("  • Redis execution state and queues") 
            print("  • Python cache files")
            if not args.preserve_knowledge:
                print("  • Research knowledge base")
                print("  • External agent cache")
            if not args.preserve_cache:
                print("  • AI response cache")
            print("  • Logs and temporary files")
            
            print("\n🚀 To start testing:")
            print("  python run_dynamic_ai.py --mode test")
            print("  python run_continuous_ai.py")
            print("  python test_system_ready.py")
            
        else:
            print("⚠️  Reset completed with warnings - check output above")
            
    except KeyboardInterrupt:
        print("\n❌ Reset interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Reset failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()