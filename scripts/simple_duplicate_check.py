"""
Simple duplicate check that works reliably with all AI providers
"""

import json
import logging
from typing import Dict, Any, Optional


async def simple_ai_duplicate_check(ai_brain, new_project: Dict[str, Any], 
                                  existing_projects_summary: str) -> Dict[str, Any]:
    """
    Perform a simple, reliable duplicate check using AI.
    
    Args:
        ai_brain: AI brain instance
        new_project: New project details
        existing_projects_summary: Summary of existing projects
        
    Returns:
        Duplicate check results
    """
    logger = logging.getLogger(__name__)
    
    # Simple prompt that asks for specific fields
    prompt = f"""
Analyze if this new project is a duplicate or too similar to existing projects.

NEW PROJECT:
- Name: {new_project.get('name', 'Unknown')}
- Description: {new_project.get('description', 'No description')}
- Problem Statement: {new_project.get('problem_statement', 'Not specified')}
- Target Market: {new_project.get('target_market', 'Not specified')}
- Key Features: {', '.join(new_project.get('initial_features', []))}

EXISTING PROJECTS:
{existing_projects_summary}

Please analyze and respond with a JSON object containing exactly these fields:
{{
    "is_duplicate": true/false,
    "confidence": 0.0 to 1.0,
    "similar_to": "name of most similar project or 'none'",
    "reason": "brief explanation",
    "recommendation": "create_new" or "enhance_existing" or "exact_duplicate"
}}

Important: 
- Set is_duplicate=true if the project serves the same purpose as an existing one
- confidence should be 0.9+ for very similar projects, 0.7-0.9 for somewhat similar
- recommendation should be "exact_duplicate" only for nearly identical projects
"""

    # Simple schema that all providers can handle
    schema = {
        "type": "object",
        "properties": {
            "is_duplicate": {"type": "boolean"},
            "confidence": {"type": "number"},
            "similar_to": {"type": "string"},
            "reason": {"type": "string"},
            "recommendation": {"type": "string"}
        }
    }
    
    try:
        response = await ai_brain.generate_enhanced_response(prompt, response_schema=schema)
        
        if response and response.get('structured_output'):
            # Successfully got structured output
            result = response.get('parsed_content', {})
            
            # Ensure all required fields are present
            return {
                'is_duplicate': result.get('is_duplicate', False),
                'duplicate_confidence': result.get('confidence', 0.0),
                'similar_to': result.get('similar_to', 'none'),
                'ai_analysis': result.get('reason', 'No analysis provided'),
                'recommendation': result.get('recommendation', 'create_new')
            }
            
        elif response and response.get('content'):
            # Try to parse from raw content
            content = response['content']
            
            # Remove markdown if present
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Try to parse JSON
            try:
                result = json.loads(content)
                return {
                    'is_duplicate': result.get('is_duplicate', False),
                    'duplicate_confidence': result.get('confidence', 0.0),
                    'similar_to': result.get('similar_to', 'none'),
                    'ai_analysis': result.get('reason', 'No analysis provided'),
                    'recommendation': result.get('recommendation', 'create_new')
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI response as JSON")
                
                # Try basic text analysis as fallback
                content_lower = content.lower()
                is_duplicate = 'duplicate' in content_lower and ('yes' in content_lower or 'true' in content_lower)
                
                return {
                    'is_duplicate': is_duplicate,
                    'duplicate_confidence': 0.5 if is_duplicate else 0.0,
                    'similar_to': 'unknown',
                    'ai_analysis': content[:200],
                    'recommendation': 'enhance_existing' if is_duplicate else 'create_new'
                }
                
    except Exception as e:
        logger.error(f"Simple duplicate check failed: {e}")
    
    # Fallback response
    return {
        'is_duplicate': False,
        'duplicate_confidence': 0.0,
        'similar_to': 'none',
        'ai_analysis': 'Unable to perform AI analysis',
        'recommendation': 'create_new'
    }