"""
AI-Generated Tool: count_lines_by_language
Description: Count items for lines by language
Generated: 2025-06-15T11:08:05.315927+00:00
Requirements: 
        Tool Name: count_lines_by_language
        Intent: Count items for lines by language
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Count all items in the system
2. Return total count with breakdown by status/type
3. Include summary statistics
        
"""

"""
Module for counting lines by language in the system.

Provides functionality to:
- Count all items in the system
- Return total count with breakdown by language
- Include summary statistics
"""

from scripts.state_manager import StateManager
from collections import defaultdict
from typing import Dict, Any

__description__ = "Count items for lines by language"
__parameters__ = {}
__examples__ = [
    {"description": "Count all lines by language",
     "code": "await count_lines_by_language()"}
]

async def count_lines_by_language(**kwargs) -> Dict[str, Any]:
    """
    Count all items in the system by language and return statistics.
    
    Returns:
        Dictionary containing:
        - total: Total count of all items
        - by_language: Breakdown of counts by language
        - summary: Summary statistics
    """
    try:
        state_manager = StateManager()
        state = state_manager.load_state()
        
        if not state or 'items' not in state:
            return {
                "total": 0,
                "by_language": {},
                "summary": "No items found in system"
            }
        
        items = state['items']
        language_counts = defaultdict(int)
        
        for item in items:
            if 'language' in item:
                language_counts[item['language']] += 1
            else:
                language_counts['unknown'] += 1
        
        total = sum(language_counts.values())
        
        return {
            "total": total,
            "by_language": dict(language_counts),
            "summary": {
                "total_items": total,
                "unique_languages": len(language_counts),
                "most_common": max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else None,
                "least_common": min(language_counts.items(), key=lambda x: x[1])[0] if language_counts else None
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "total": 0,
            "by_language": {},
            "summary": "Error occurred while counting items"
        }
