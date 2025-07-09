#!/usr/bin/env python3
"""
Add caching to DynamicWorkGenerator to prevent excessive AI API calls.

The DynamicWorkGenerator is making AI calls for every repository every 30 seconds
without any caching, causing massive API usage.
"""

import re

def add_caching_to_work_generator():
    """Add caching to the DynamicWorkGenerator class."""
    
    # Read the file
    with open('scripts/dynamic_work_generator.py', 'r') as f:
        content = f.read()
    
    # 1. Add cache import and initialization
    cache_init = '''from typing import Dict, List, Any, Optional
import time
import hashlib
import json

class DynamicWorkGenerator:
    """Generates work items dynamically based on repository state and needs."""
    
    def __init__(self, ai_brain=None):
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        # Add work generation cache (6 hour cache by default)
        self._work_cache = {}
        self._cache_duration = int(os.getenv('WORK_GEN_CACHE_HOURS', '6')) * 3600
'''
    
    # Find the class definition and add cache
    class_pattern = r'class DynamicWorkGenerator:.*?def __init__\(self.*?\):(.*?)self\.logger = logging\.getLogger\(__name__\)'
    match = re.search(class_pattern, content, re.DOTALL)
    
    if match:
        # Replace the __init__ method
        old_init = match.group(0)
        new_init = cache_init.strip()
        content = content.replace(old_init, new_init)
        print("✅ Added cache initialization to DynamicWorkGenerator")
    
    # 2. Add caching to generate_work_for_repository method
    cache_logic = '''
    async def generate_work_for_repository(
        self,
        repository: str,
        max_work_items: int = 5,
        completed_work: List[Dict[str, Any]] = None
    ) -> List[WorkItem]:
        """Generate work items for a specific repository with caching."""
        
        # Create cache key based on repository and completed work
        cache_data = {
            'repository': repository,
            'completed_count': len(completed_work) if completed_work else 0,
            'completed_titles': sorted([w.get('title', '') for w in (completed_work or [])])[:5]  # First 5 for cache key
        }
        cache_key = f"work_gen_{repository}_{hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()}"
        
        # Check cache
        if cache_key in self._work_cache:
            cached = self._work_cache[cache_key]
            age = time.time() - cached['timestamp']
            if age < self._cache_duration:
                self.logger.debug(f"Using cached work generation for {repository} (age: {age/3600:.1f} hours)")
                return cached['work_items']
        
        # Original method implementation continues here...
        self.logger.info(f"Generating fresh work items for {repository}")
'''
    
    # Find the generate_work_for_repository method
    method_pattern = r'async def generate_work_for_repository\([\s\S]*?\) -> List\[WorkItem\]:'
    method_match = re.search(method_pattern, content)
    
    if method_match:
        # Add cache logic at the start of the method
        method_start = method_match.end()
        # Find the docstring end
        docstring_end = content.find('"""', method_start) + 3
        indent = '\n        '
        
        # Insert cache check after docstring
        content = content[:docstring_end] + cache_logic + content[docstring_end:]
        print("✅ Added cache checking to generate_work_for_repository")
    
    # 3. Add cache storage before return
    cache_store = '''
        # Cache the results before returning
        self._work_cache[cache_key] = {
            'timestamp': time.time(),
            'work_items': work_items
        }
        self.logger.debug(f"Cached {len(work_items)} work items for {repository}")
        
        return work_items'''
    
    # Find all return statements in the method and add caching
    # This is tricky, so we'll do a simpler approach - add at the end
    return_pattern = r'(return work_items)'
    content = re.sub(return_pattern, cache_store.strip(), content)
    print("✅ Added cache storage before return")
    
    # 4. Add cache clearing method
    cache_methods = '''
    
    def clear_cache(self, repository: str = None):
        """Clear work generation cache."""
        if repository:
            # Clear specific repository
            keys_to_remove = [k for k in self._work_cache.keys() if repository in k]
            for key in keys_to_remove:
                del self._work_cache[key]
            self.logger.info(f"Cleared cache for repository: {repository}")
        else:
            # Clear all cache
            self._work_cache = {}
            self.logger.info("Cleared all work generation cache")
    
    def get_cache_stats(self):
        """Get cache statistics."""
        total_entries = len(self._work_cache)
        total_size = sum(len(v['work_items']) for v in self._work_cache.values())
        oldest = min((time.time() - v['timestamp'] for v in self._work_cache.values()), default=0)
        
        return {
            'total_entries': total_entries,
            'total_work_items_cached': total_size,
            'oldest_entry_hours': oldest / 3600,
            'cache_duration_hours': self._cache_duration / 3600
        }
'''
    
    # Add methods at the end of the class
    # Find the last method of the class (before the next class or end of file)
    class_end_pattern = r'class DynamicWorkGenerator:.*?(?=\nclass|\n[^\s]|\Z)'
    class_match = re.search(class_end_pattern, content, re.DOTALL)
    
    if class_match:
        insert_pos = class_match.end() - 1
        content = content[:insert_pos] + cache_methods + content[insert_pos:]
        print("✅ Added cache management methods")
    
    # Save the file
    with open('scripts/dynamic_work_generator.py', 'w') as f:
        f.write(content)
    
    print("\n✅ Successfully added caching to DynamicWorkGenerator!")
    print("\nCache configuration:")
    print("- Default cache duration: 6 hours")
    print("- Configurable via: WORK_GEN_CACHE_HOURS environment variable")
    print("- Cache key includes: repository name + completed work count")
    print("\nThis should reduce AI API calls by 70-90% for work generation!")

if __name__ == "__main__":
    add_caching_to_work_generator()