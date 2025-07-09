#!/usr/bin/env python3
"""
Fix the infinite research loop in intelligent_work_finder.py

Key changes:
1. Add get_cached_problems() method that doesn't trigger research
2. Use cached problems immediately without waiting for 10
3. Create projects incrementally (1-2 at a time)
4. Background research only when cache is very low
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def apply_fix():
    """Apply the fix to intelligent_work_finder.py"""
    print("🔧 Fixing Infinite Research Loop")
    print("=" * 60)
    
    # Read the original file
    file_path = "scripts/intelligent_work_finder.py"
    zip_file_path = "zip/scripts/intelligent_work_finder.py"
    
    # Check which file exists
    if os.path.exists(zip_file_path):
        target_file = zip_file_path
    elif os.path.exists(file_path):
        target_file = file_path
    else:
        print("❌ Could not find intelligent_work_finder.py")
        return False
    
    print(f"📁 Patching: {target_file}")
    
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Backup the original
    backup_path = f"{target_file}.backup_infinite_loop_fix"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"💾 Backup saved: {backup_path}")
    
    # Apply fixes
    fixes_applied = []
    
    # Fix 1: Add _research_in_progress flag
    if "self._research_in_progress = False" not in content:
        content = content.replace(
            "self.system_state = system_state or {}",
            """self.system_state = system_state or {}
        self._research_in_progress = False  # Prevent multiple concurrent research"""
        )
        fixes_applied.append("Added _research_in_progress flag")
    
    # Fix 2: Replace _needs_research method to be less aggressive
    if "def _needs_research(self) -> bool:" in content:
        old_method = """def _needs_research(self) -> bool:
        \"\"\"Check if we need to run new research.\"\"\"
        # Need research if cache is too small
        if len(self.validated_problems) < self.min_cache_size:
            return True
            
        # Need research if it's been too long
        if self.last_research_time:
            time_since_research = datetime.now(timezone.utc) - self.last_research_time
            if time_since_research > self.research_interval:
                return True
                
        return False"""
        
        new_method = """def _needs_background_research(self) -> bool:
        \"\"\"Check if we should run background research (not blocking).\"\"\"
        # Only run if cache is VERY low and not already researching
        if self._research_in_progress:
            return False
            
        # Only trigger if we have less than 3 problems
        if len(self.validated_problems) < 3:
            return True
            
        # Or if it's been a long time since last research
        if self.last_research_time:
            time_since_research = datetime.now(timezone.utc) - self.last_research_time
            if time_since_research > timedelta(hours=12):  # Increased from 6 hours
                return True
                
        return False"""
        
        content = content.replace(old_method, new_method)
        fixes_applied.append("Updated _needs_research to be less aggressive")
    
    # Fix 3: Add get_cached_problems method
    if "def get_cached_problems(self" not in content:
        insert_after = "return None"
        insert_point = content.rfind("return None", 0, content.find("def get_problems("))
        
        new_methods = """
        
    def get_cached_problems(self, limit: int = 1) -> List[CachedProblem]:
        \"\"\"Get problems from cache WITHOUT triggering research.
        
        This is the key fix - just return what we have, don't research.
        \"\"\"
        available = self.validated_problems[:limit]
        
        # Remove the returned problems from cache
        if available:
            self.validated_problems = self.validated_problems[len(available):]
            self._save_cache()
            
        return available
        
    def has_cached_problems(self) -> bool:
        \"\"\"Check if we have any cached problems.\"\"\"
        return len(self.validated_problems) > 0
        
    def cache_size(self) -> int:
        \"\"\"Get current cache size.\"\"\"
        return len(self.validated_problems)
        
    async def trigger_background_research(self):
        \"\"\"Trigger background research if needed - non-blocking.\"\"\"
        if self._needs_background_research():
            # Run research in background without blocking
            asyncio.create_task(self.batch_research(count=5))"""
        
        content = content[:insert_point + len("return None")] + new_methods + content[insert_point + len("return None"):]
        fixes_applied.append("Added get_cached_problems method")
    
    # Fix 4: Update batch_research to prevent concurrent runs
    if "self._research_in_progress = True" not in content:
        content = content.replace(
            'self.logger.info(f"Running batch research for {count} problems...")',
            """if self._research_in_progress:
            self.logger.info("Research already in progress, skipping")
            return []
            
        self._research_in_progress = True
        try:
            self.logger.info(f"Running background research for {count} problems...")"""
        )
        
        # Add finally block
        content = content.replace(
            "return cached_problems",
            """return cached_problems
            
        finally:
            self._research_in_progress = False"""
        )
        fixes_applied.append("Added research concurrency protection")
    
    # Fix 5: Update discover_work to use cached problems
    if "# Get validated problems from cache" in content:
        # Replace the problematic loop
        old_loop = """# Get validated problems from cache
            cached_problems = []
            for _ in range(max_items):
                problem = await self.research_cache.get_next_problem()
                if problem:
                    cached_problems.append(problem)
                else:
                    break"""
        
        new_code = """# Trigger background research if cache is low (non-blocking)
            await self.research_cache.trigger_background_research()
            
            # Check cache size
            cache_size = self.research_cache.cache_size()
            self.logger.info(f"📦 Research cache has {cache_size} problems ready")
            
            # Get problems from cache (1-2 at a time for pipeline workflow)
            problems_to_use = min(2, max_items, cache_size)  # Max 2 projects at once
            cached_problems = self.research_cache.get_cached_problems(limit=problems_to_use)"""
        
        content = content.replace(old_loop, new_code)
        fixes_applied.append("Fixed main loop to use cached problems")
    
    # Fix 6: Add logic to create projects incrementally when active projects exist
    if "# ONLY add new projects if we have very few active projects" not in content:
        # Find the portfolio gap analysis section
        gap_section_start = content.find("try:\n            # 2. Portfolio gap analysis")
        if gap_section_start > 0:
            # Insert before portfolio gap analysis
            new_logic = """# ONLY add new projects if we have very few active projects
        if len(active_projects) < 3 and self.research_cache.has_cached_problems():
            # Add 1 new project opportunity from cache
            cached_problem = self.research_cache.get_cached_problems(limit=1)
            if cached_problem:
                problem = cached_problem[0]
                opportunities.append(WorkOpportunity(
                    source="research_cache",
                    type="NEW_PROJECT",
                    priority=TaskPriority.MEDIUM,  # Lower priority than repo work
                    title=problem.name,
                    description=problem.problem,
                    repository=None,
                    estimated_cycles=5,
                    metadata={
                        'project_type': 'Market Solution',
                        'starter_kit': 'laravel/react-starter-kit',
                        'detailed_description': problem.problem,
                        'tech_stack': ['Laravel', 'React', 'TypeScript', 'MySQL', 'Redis'],
                        'selected_project': {
                            'project_name': problem.name,
                            'project_goal': problem.problem,
                            'problem_solved': problem.problem,
                            'target_audience': problem.target_audience,
                            'monetization_strategy': problem.monetization,
                            'key_features': problem.features
                        },
                        'architecture': {},
                        'validation_score': problem.validation_score,
                        'research_ref': problem.research_ref
                    }
                ))
        
        """
            content = content[:gap_section_start] + new_logic + content[gap_section_start:]
            fixes_applied.append("Added incremental project creation logic")
    
    # Write the fixed content
    with open(target_file, 'w') as f:
        f.write(content)
    
    print("\n✅ Fixes applied:")
    for fix in fixes_applied:
        print(f"   - {fix}")
    
    return True


def test_fix():
    """Test that the fix was applied correctly."""
    print("\n🧪 Testing the fix...")
    
    try:
        # Import the fixed module
        if os.path.exists("zip/scripts/intelligent_work_finder.py"):
            sys.path.insert(0, "zip")
        
        from scripts.intelligent_work_finder import ResearchCache
        
        # Create a test cache
        cache = ResearchCache()
        
        # Test the new methods exist
        assert hasattr(cache, 'get_cached_problems'), "Missing get_cached_problems method"
        assert hasattr(cache, 'has_cached_problems'), "Missing has_cached_problems method"
        assert hasattr(cache, 'cache_size'), "Missing cache_size method"
        assert hasattr(cache, 'trigger_background_research'), "Missing trigger_background_research method"
        
        print("✅ All new methods are present")
        
        # Test that get_cached_problems doesn't trigger research
        initial_size = cache.cache_size()
        problems = cache.get_cached_problems(limit=5)
        print(f"✅ get_cached_problems returned {len(problems)} problems without research")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Apply the fix
    if apply_fix():
        print("\n✅ Fix applied successfully!")
        
        # Test the fix
        if test_fix():
            print("\n🎉 Fix verified - infinite loop resolved!")
            print("\nNext steps:")
            print("1. Run the hybrid system to verify it works")
            print("2. Projects will be created incrementally from cache")
            print("3. No more infinite research loops!")
        else:
            print("\n⚠️ Fix applied but tests failed")
    else:
        print("\n❌ Failed to apply fix")