#!/usr/bin/env python3
"""
Patch Redis client to handle common errors more gracefully.

This script patches the Redis client to:
1. Handle "no such key" errors without logging as errors
2. Improve timeout handling
3. Add better error context
"""

import os
import re
import shutil
from datetime import datetime


def patch_redis_client():
    """Patch the Redis client for better error handling."""
    
    print("Patching Redis Client")
    print("=" * 50)
    
    redis_client_path = "scripts/redis_integration/redis_client.py"
    
    # Backup the original file
    backup_path = f"{redis_client_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(redis_client_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    # Read the current file
    with open(redis_client_path, 'r') as f:
        content = f.read()
    
    # Patch 1: Improve error handling in execute_with_retry
    old_execute_with_retry = '''            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"Redis operation failed after {max_retries} retries: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)'''
    
    new_execute_with_retry = '''            except Exception as e:
                # Handle specific Redis errors more gracefully
                error_msg = str(e).lower()
                
                # Don't log "no such key" as an error - it's expected behavior
                if "no such key" in error_msg:
                    return None
                
                # Handle timeout errors with more context
                if "timeout" in error_msg:
                    self.logger.warning(f"Redis timeout on attempt {attempt + 1}/{max_retries + 1}")
                
                if attempt == max_retries:
                    # Only log as error if it's not an expected condition
                    if "no such key" not in error_msg:
                        self.logger.error(f"Redis operation failed after {max_retries} retries: {e}")
                    raise
                    
                await asyncio.sleep(2 ** attempt)'''
    
    if old_execute_with_retry in content:
        content = content.replace(old_execute_with_retry, new_execute_with_retry)
        print("✓ Patched execute_with_retry error handling")
    else:
        print("⚠ Could not find execute_with_retry pattern to patch")
    
    # Patch 2: Add better error context to operations
    # Find the get method and improve it
    old_get_method = '''    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return await self.execute_with_retry(lambda conn, k: conn.get(k), key)'''
    
    new_get_method = '''    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            result = await self.execute_with_retry(lambda conn, k: conn.get(k), key)
            return result.decode('utf-8') if result and isinstance(result, bytes) else result
        except Exception as e:
            # Silently return None for "no such key" errors
            if "no such key" in str(e).lower():
                return None
            raise'''
    
    if old_get_method in content:
        content = content.replace(old_get_method, new_get_method)
        print("✓ Patched get method for better error handling")
    else:
        print("⚠ Could not find get method pattern to patch")
    
    # Patch 3: Add graceful handling for scan operations
    scan_pattern = r'async def scan\(self,[^}]+\}'
    scan_match = re.search(scan_pattern, content, re.DOTALL)
    
    if scan_match:
        old_scan = scan_match.group(0)
        # Check if error handling already exists
        if "no such key" not in old_scan:
            # Add error handling to scan method
            new_scan = old_scan.replace(
                "return await self.execute_with_retry(lambda conn, *a, **kw: conn.scan(*a, **kw), cursor, match=match, count=count)",
                """try:
            return await self.execute_with_retry(lambda conn, *a, **kw: conn.scan(*a, **kw), cursor, match=match, count=count)
        except Exception as e:
            # Return empty result for scan errors
            if "no such key" in str(e).lower():
                return (0, [])
            raise"""
            )
            
            content = content.replace(old_scan, new_scan)
            print("✓ Patched scan method for graceful error handling")
    else:
        print("⚠ Could not find scan method to patch")
    
    # Patch 4: Improve circuit breaker to handle expected errors
    old_circuit_breaker = '''        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                self.logger.info("Circuit breaker closed - connection restored")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e'''
    
    new_circuit_breaker = '''        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                self.logger.info("Circuit breaker closed - connection restored")
            return result
        except Exception as e:
            # Don't count "no such key" errors as failures
            error_msg = str(e).lower()
            if "no such key" not in error_msg:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e'''
    
    if old_circuit_breaker in content:
        content = content.replace(old_circuit_breaker, new_circuit_breaker)
        print("✓ Patched circuit breaker to ignore expected errors")
    else:
        print("⚠ Could not find circuit breaker pattern to patch")
    
    # Write the patched content
    with open(redis_client_path, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Successfully patched {redis_client_path}")
    print(f"  Original backed up to: {backup_path}")
    
    # Create a test script to verify the patches
    test_script = '''#!/usr/bin/env python3
"""Test the patched Redis client."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.redis_integration import get_redis_client


async def test_patches():
    """Test that patches work correctly."""
    print("Testing Redis client patches...")
    
    redis_client = await get_redis_client()
    
    # Test 1: Get non-existent key should return None without error
    result = await redis_client.get("definitely:does:not:exist:key:12345")
    assert result is None, f"Expected None, got {result}"
    print("✓ Non-existent key returns None")
    
    # Test 2: Set and get a value
    test_key = "patch:test:key"
    await redis_client.set(test_key, "test_value")
    result = await redis_client.get(test_key)
    assert result == "test_value", f"Expected 'test_value', got {result}"
    print("✓ Set/Get works correctly")
    
    # Clean up
    await redis_client.delete(test_key)
    
    print("\\nAll patch tests passed!")


if __name__ == "__main__":
    asyncio.run(test_patches())
'''
    
    with open("test_redis_patches.py", 'w') as f:
        f.write(test_script)
    
    os.chmod("test_redis_patches.py", 0o755)
    print("\n✓ Created test_redis_patches.py to verify the patches")
    
    print("\nNext steps:")
    print("1. Run: python test_redis_patches.py to verify patches")
    print("2. Restart the continuous AI system")
    print("3. Monitor logs - 'no such key' errors should no longer appear")


if __name__ == "__main__":
    patch_redis_client()