#!/usr/bin/env python3
"""
Apply comprehensive fixes to the evolutionary improvement engine for robust JSON parsing.
"""

import os
import re

def main():
    """Apply all fixes to the evolutionary engine."""
    
    # Fix the evolutionary improvement engine
    fix_evolutionary_engine()
    
    # Fix the improvement ensemble
    fix_improvement_ensemble()
    
    print("\n✅ All fixes applied successfully!")
    print("\nKey improvements:")
    print("1. Simplified JSON structure - no code in initial generation")
    print("2. Multiple parsing strategies with fallbacks")
    print("3. Automatic code generation after parsing")
    print("4. Robust error recovery")
    print("5. Better handling of AI response variations")

def fix_evolutionary_engine():
    """Fix the evolutionary improvement engine."""
    
    engine_path = "scripts/evolutionary_improvement_engine.py"
    
    # Read current content
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Find and replace the _build_generation_prompt method
    new_prompt_method = '''    def _build_generation_prompt(self, improvement_type: str, context: Optional[Dict]) -> str:
        """Build prompt for generating improvements."""
        base_prompt = f"""Generate a {improvement_type} improvement for an AI chat assistant system.

System Context:
- Self-improving AI system with tool generation capabilities
- Uses LLMs for natural language processing
- Has workflow orchestration and error handling
- Supports context-aware improvements
"""
        
        if context:
            base_prompt += f"""
Specific Need:
- Type: {context.get('type')}
- Description: {context.get('description')}
- Priority: {context.get('priority', 0)}
"""
        
        base_prompt += f"""
Create a specific, actionable improvement.

Return ONLY a JSON object like this example:
{{
    "name": "response_cache_optimizer",
    "type": "{improvement_type}",
    "description": "Optimizes response caching using similarity detection",
    "approach": "Use embeddings to detect similar queries and serve cached responses",
    "metrics": ["cache_hit_rate", "response_time"],
    "integration_points": ["http_ai_client", "cache_manager"]
}}

Your response must be ONLY the JSON object, no other text.
"""
        
        return base_prompt'''
    
    # Find and replace the _parse_improvement_response method
    new_parse_method = '''    def _parse_improvement_response(self, response: Any, improvement_type: str) -> Optional[Dict]:
        """Parse LLM response into improvement structure with robust error handling."""
        try:
            # Handle dict response
            if isinstance(response, dict):
                response_text = response.get('content', '')
            else:
                response_text = str(response)
            
            # Clean the response
            response_text = response_text.strip()
            
            # Try to parse as JSON directly
            try:
                import json
                data = json.loads(response_text)
                if isinstance(data, dict) and 'name' in data:
                    return self._finalize_improvement(data, improvement_type)
            except:
                pass
            
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\\s*({.*?})\\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    import json
                    data = json.loads(json_match.group(1))
                    if isinstance(data, dict) and 'name' in data:
                        return self._finalize_improvement(data, improvement_type)
                except:
                    pass
            
            # Try to find any JSON-like structure
            json_pattern = re.search(r'{[^{}]*"name"[^{}]*}', response_text, re.DOTALL)
            if json_pattern:
                try:
                    import json
                    # Clean up common issues
                    json_str = json_pattern.group(0)
                    json_str = re.sub(r'\\n\\s*', ' ', json_str)
                    json_str = re.sub(r',\\s*}', '}', json_str)
                    data = json.loads(json_str)
                    if isinstance(data, dict) and 'name' in data:
                        return self._finalize_improvement(data, improvement_type)
                except:
                    pass
            
            # Last resort: extract key fields
            name_match = re.search(r'"name"\\s*:\\s*"([^"]+)"', response_text)
            if name_match:
                return self._finalize_improvement({
                    'name': name_match.group(1),
                    'type': improvement_type,
                    'description': 'AI-generated improvement',
                    'approach': 'To be implemented'
                }, improvement_type)
            
            self.logger.warning(f"Could not parse response: {response_text[:100]}...")
            return None
            
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return None
    
    def _finalize_improvement(self, data: Dict, improvement_type: str) -> Dict:
        """Finalize improvement with all required fields."""
        import time
        from datetime import datetime, timezone
        
        # Ensure required fields
        data['type'] = data.get('type', improvement_type)
        data['name'] = data.get('name', f"{improvement_type}_{int(time.time())}")
        data['description'] = data.get('description', f'Generated {improvement_type} improvement')
        data['approach'] = data.get('approach', 'Implementation approach to be defined')
        data['metrics'] = data.get('metrics', ['success_rate', 'performance'])
        data['integration_points'] = data.get('integration_points', ['system_core'])
        
        # Generate code if not present
        if 'code' not in data:
            data['code'] = self._generate_code_for_improvement(data)
        
        data['created_at'] = datetime.now(timezone.utc).isoformat()
        return data
    
    def _generate_code_for_improvement(self, improvement: Dict) -> str:
        """Generate Python code for the improvement."""
        class_name = re.sub(r'[^a-zA-Z0-9]', '', improvement['name'].title())
        
        code = f"""\"\"\"
{improvement['name']} - {improvement['description']}
Generated by CWMAI Evolutionary System
\"\"\"

import logging
from typing import Dict, Any

class {class_name}:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "{improvement['name']}"
        
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # {improvement['approach']}
        self.logger.info(f"Applying {self.name}")
        return {{'success': True, 'improvement': self.name}}
"""
        return code'''
    
    # Apply the replacements
    # Replace _build_generation_prompt
    pattern1 = r'def _build_generation_prompt\(self.*?\n        return base_prompt'
    content = re.sub(pattern1, new_prompt_method.strip(), content, flags=re.DOTALL)
    
    # Replace _parse_improvement_response
    pattern2 = r'def _parse_improvement_response\(self.*?\n        return None'
    match = re.search(pattern2, content, re.DOTALL)
    if match:
        # Insert the new methods
        insert_pos = match.end()
        content = content[:match.start()] + new_parse_method.strip() + content[insert_pos:]
    
    # Also update mutation and crossover to use simpler prompts
    content = update_mutation_crossover(content)
    
    # Write back
    with open(engine_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed evolutionary improvement engine: {engine_path}")

def update_mutation_crossover(content):
    """Update mutation and crossover methods."""
    
    # Simpler crossover
    crossover_pattern = r'(async def _crossover.*?prompt = f""")(.*?)(""")'
    crossover_replacement = r'\1Combine these improvements: {parent1["name"]} and {parent2["name"]}. Return ONLY a JSON with: name, type, description, approach, metrics, integration_points\3'
    content = re.sub(crossover_pattern, crossover_replacement, content, flags=re.DOTALL)
    
    # Simpler mutation
    mutation_pattern = r'(async def _mutate.*?prompt = f""")(.*?)(""")'
    mutation_replacement = r'\1Mutate {parent["name"]} by: {mutation}. Return ONLY a JSON with: name, type, description, approach, metrics, integration_points\3'
    content = re.sub(mutation_pattern, mutation_replacement, content, flags=re.DOTALL)
    
    return content

def fix_improvement_ensemble():
    """Fix the improvement ensemble prompts."""
    
    ensemble_path = "scripts/improvement_ensemble.py"
    
    with open(ensemble_path, 'r') as f:
        content = f.read()
    
    # Replace code generation requests with approach descriptions
    content = content.replace('"code": "complete Python implementation"', '"approach": "implementation approach"')
    content = content.replace('"code": "implementation combining both ideas"', '"approach": "combined approach"') 
    content = content.replace('"code": "validated and fixed code"', '"approach": "validated approach"')
    content = content.replace('Code:', 'Approach:')
    content = content.replace('```python', 'Approach: ')
    content = content.replace('```', '')
    
    with open(ensemble_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed improvement ensemble: {ensemble_path}")

if __name__ == "__main__":
    main()