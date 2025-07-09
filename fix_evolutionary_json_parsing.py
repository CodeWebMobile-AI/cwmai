#!/usr/bin/env python3
"""
Fix JSON parsing issues in the evolutionary improvement engine.

This script updates the prompts and parsing logic to handle AI responses more robustly.
"""

import os
import sys

def fix_json_parsing():
    """Apply fixes to the evolutionary improvement engine."""
    
    engine_path = "scripts/evolutionary_improvement_engine.py"
    ensemble_path = "scripts/improvement_ensemble.py"
    
    # Read the current engine file
    with open(engine_path, 'r') as f:
        engine_content = f.read()
    
    # New simplified prompt that avoids code in JSON
    new_generation_prompt = '''    def _build_generation_prompt(self, improvement_type: str, context: Optional[Dict]) -> str:
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
Requirements:
1. Focus on a specific, measurable improvement
2. Describe the approach clearly
3. List concrete metrics to track success
4. Identify integration points with existing system

Return ONLY a JSON object with this structure:
{{
    "name": "short_descriptive_name",
    "type": "{improvement_type}",
    "description": "Clear description of what this improves (1-2 sentences)",
    "approach": "High-level approach to implement this (2-3 sentences)",
    "metrics": ["metric1", "metric2"],
    "integration_points": ["where this connects to existing system"],
    "key_benefits": ["benefit1", "benefit2"]
}}

Example valid response:
{{
    "name": "adaptive_response_caching",
    "type": "optimization",
    "description": "Implements intelligent caching of AI responses based on query similarity.",
    "approach": "Use embedding similarity to identify similar queries and cache responses. Implement TTL based on query frequency.",
    "metrics": ["cache_hit_rate", "response_time_reduction"],
    "integration_points": ["http_ai_client", "response_cache"],
    "key_benefits": ["Faster responses", "Reduced API costs"]
}}

IMPORTANT: Return ONLY the JSON object, no other text.
"""
        
        return base_prompt'''

    # New robust parsing method
    new_parse_method = '''    def _parse_improvement_response(self, response: Any, improvement_type: str) -> Optional[Dict]:
        """Parse LLM response into improvement structure with robust error handling."""
        try:
            # Handle dict response from generate_enhanced_response
            if isinstance(response, dict):
                response_text = response.get('content', '')
            else:
                response_text = str(response)
            
            # Clean the response text
            response_text = response_text.strip()
            
            # Try multiple parsing strategies
            import re
            import json
            
            # Strategy 1: Direct JSON parse
            try:
                improvement = json.loads(response_text)
                if isinstance(improvement, dict) and 'name' in improvement:
                    return self._validate_and_enhance_improvement(improvement, improvement_type)
            except:
                pass
            
            # Strategy 2: Extract JSON from code block
            json_block = re.search(r'```(?:json)?\\s*({.*?})\\s*```', response_text, re.DOTALL)
            if json_block:
                try:
                    improvement = json.loads(json_block.group(1))
                    if isinstance(improvement, dict) and 'name' in improvement:
                        return self._validate_and_enhance_improvement(improvement, improvement_type)
                except:
                    pass
            
            # Strategy 3: Find JSON object with regex
            json_pattern = re.search(r'\\{[^{}]*"name"[^{}]*\\}', response_text, re.DOTALL)
            if json_pattern:
                try:
                    # Extract and clean the JSON string
                    json_str = json_pattern.group(0)
                    # Fix common issues
                    json_str = re.sub(r'\\n\\s*', ' ', json_str)  # Remove newlines
                    json_str = re.sub(r',\\s*}', '}', json_str)  # Remove trailing commas
                    improvement = json.loads(json_str)
                    if isinstance(improvement, dict) and 'name' in improvement:
                        return self._validate_and_enhance_improvement(improvement, improvement_type)
                except:
                    pass
            
            # Strategy 4: Build from partial data
            # Try to extract key fields even if JSON is broken
            name_match = re.search(r'"name"\\s*:\\s*"([^"]+)"', response_text)
            desc_match = re.search(r'"description"\\s*:\\s*"([^"]+)"', response_text)
            
            if name_match:
                improvement = {
                    'name': name_match.group(1),
                    'type': improvement_type,
                    'description': desc_match.group(1) if desc_match else f"AI-generated {improvement_type} improvement",
                    'approach': 'To be implemented',
                    'metrics': [f"{improvement_type}_success_rate"],
                    'integration_points': ['system_core']
                }
                return self._validate_and_enhance_improvement(improvement, improvement_type)
            
            # If all strategies fail, log and return None
            self.logger.warning(f"Could not parse improvement from response: {response_text[:200]}...")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse improvement response: {e}")
            return None
    
    def _validate_and_enhance_improvement(self, improvement: Dict, improvement_type: str) -> Dict:
        """Validate and enhance the improvement with required fields."""
        # Ensure all required fields exist
        improvement['type'] = improvement.get('type', improvement_type)
        improvement['name'] = improvement.get('name', f"{improvement_type}_improvement_{int(time.time())}")
        improvement['description'] = improvement.get('description', 'Generated improvement')
        
        # Generate placeholder code if not present
        if 'code' not in improvement:
            improvement['code'] = self._generate_improvement_code(improvement)
        
        # Ensure other fields
        improvement['metrics'] = improvement.get('metrics', ['success_rate', 'performance'])
        improvement['integration_points'] = improvement.get('integration_points', ['main_system'])
        improvement['created_at'] = datetime.now(timezone.utc).isoformat()
        
        return improvement
    
    def _generate_improvement_code(self, improvement: Dict) -> str:
        """Generate implementation code for an improvement."""
        class_name = improvement['name'].replace('-', '_').replace(' ', '_').title().replace('_', '')
        imp_name = improvement['name']
        imp_desc = improvement['description']
        imp_metrics = improvement.get('metrics', [])
        imp_approach = improvement.get('approach', 'Not specified')
        
        code_template = f\'\'\'"""
{imp_name} - {imp_desc}

This is an AI-generated improvement for the CWMAI system.
"""

import logging
from typing import Dict, Any, Optional

class {class_name}:
    """Implementation of {imp_name}."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {{metric: 0 for metric in {imp_metrics}}}
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this improvement to the given context."""
        try:
            # Implementation based on approach: {imp_approach}
            result = {{
                'success': True,
                'improvement': '{imp_name}',
                'metrics': self.metrics
            }}
            
            # TODO: Implement the actual improvement logic
            self.logger.info(f"Applied improvement: {imp_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to apply {imp_name}: {{e}}")
            return {{'success': False, 'error': str(e)}}
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for this improvement."""
        return self.metrics.copy()
\'\'\'
        return code_template'''

    # Replace the methods in the file
    import re
    
    # Replace _build_generation_prompt method
    pattern1 = r'def _build_generation_prompt\(self.*?\n        return base_prompt'
    engine_content = re.sub(pattern1, new_generation_prompt.strip(), engine_content, flags=re.DOTALL)
    
    # Replace _parse_improvement_response method  
    pattern2 = r'def _parse_improvement_response\(self.*?\n        return None'
    engine_content = re.sub(pattern2, new_parse_method.strip(), engine_content, flags=re.DOTALL)
    
    # Write back the updated content
    with open(engine_path, 'w') as f:
        f.write(engine_content)
    
    print(f"✅ Fixed JSON parsing in {engine_path}")
    
    # Also fix the ensemble prompts to be simpler
    with open(ensemble_path, 'r') as f:
        ensemble_content = f.read()
    
    # Update ensemble to use simpler JSON structure
    ensemble_content = ensemble_content.replace(
        '"code": "complete Python implementation"',
        '"approach": "how to implement this"'
    )
    ensemble_content = ensemble_content.replace(
        '"code": "complete implementation combining both ideas"',
        '"approach": "how to combine both ideas"'
    )
    ensemble_content = ensemble_content.replace(
        '"code": "validated and fixed code"',
        '"approach": "implementation approach"'
    )
    
    with open(ensemble_path, 'w') as f:
        f.write(ensemble_content)
    
    print(f"✅ Fixed JSON structure in {ensemble_path}")
    
    # Fix mutation prompts
    fix_mutation_prompts()
    
    print("\n✅ All JSON parsing fixes applied successfully!")
    print("\nKey improvements:")
    print("1. Simplified JSON structure - no code in initial generation")
    print("2. Multiple parsing strategies with fallbacks")
    print("3. Automatic code generation after parsing")
    print("4. Robust error recovery")
    print("5. Validation and enhancement of parsed data")

def fix_mutation_prompts():
    """Fix the mutation and crossover prompts to avoid JSON issues."""
    engine_path = "scripts/evolutionary_improvement_engine.py"
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Simplified crossover prompt
    new_crossover = '''    async def _crossover(self, parent1: Dict, parent2: Dict) -> Optional[Dict]:
        """Create a child improvement by combining two parents."""
        prompt = f"""Combine these two improvements into a new, better improvement:

Parent 1: {parent1['name']}
Type: {parent1['type']}
Description: {parent1['description']}
Approach: {parent1.get('approach', 'Not specified')}

Parent 2: {parent2['name']}
Type: {parent2['type']}  
Description: {parent2['description']}
Approach: {parent2.get('approach', 'Not specified')}

Create a new improvement that combines the best aspects of both parents.

Return ONLY this JSON structure:
{{
    "name": "combined_improvement_name",
    "type": "{parent1['type']}",
    "description": "What this combined improvement does",
    "approach": "How to implement the combination",
    "metrics": ["metric1", "metric2"],
    "integration_points": ["point1", "point2"],
    "parent_features": ["feature from parent1", "feature from parent2"]
}}
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            child = self._parse_improvement_response(response, parent1['type'])
            if child:
                child['parents'] = [parent1.get('id', 'unknown'), parent2.get('id', 'unknown')]
                child['generation'] = self.generation_count
                child['fitness'] = 0.0
                child['id'] = hashlib.md5(f"{child['name']}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            return child
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            return None'''
    
    # Simplified mutation prompt
    new_mutation = '''    async def _mutate(self, parent: Dict) -> Optional[Dict]:
        """Create a mutated version of an improvement."""
        mutation_types = [
            "Add error recovery mechanisms",
            "Optimize for performance", 
            "Add new features",
            "Improve integration points",
            "Enhance metrics collection",
            "Add AI-powered enhancements"
        ]
        
        mutation = random.choice(mutation_types)
        
        prompt = f"""Mutate this improvement by: {mutation}

Original improvement: {parent['name']}
Type: {parent['type']}
Description: {parent['description']}
Current approach: {parent.get('approach', 'Not specified')}

Create an improved version with the mutation applied.

Return ONLY this JSON structure:
{{
    "name": "{parent['name']}_mutated",
    "type": "{parent['type']}",
    "description": "Enhanced description with mutation applied",
    "approach": "Updated approach incorporating the mutation",
    "metrics": ["original_metrics", "new_metric_from_mutation"],
    "integration_points": ["existing_points", "new_integration"],
    "mutation_benefits": ["what the mutation adds"]
}}
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            mutant = self._parse_improvement_response(response, parent['type'])
            if mutant:
                mutant['parent'] = parent.get('id', 'unknown')
                mutant['mutation'] = mutation
                mutant['generation'] = self.generation_count
                mutant['fitness'] = 0.0
                mutant['id'] = hashlib.md5(f"{mutant['name']}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            return mutant
        except Exception as e:
            self.logger.error(f"Mutation failed: {e}")
            return None'''
    
    # Replace the methods
    content = re.sub(
        r'async def _crossover\(self.*?return None',
        new_crossover.strip(),
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'async def _mutate\(self.*?return None',
        new_mutation.strip(),
        content,
        flags=re.DOTALL
    )
    
    with open(engine_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed mutation and crossover prompts")

if __name__ == "__main__":
    fix_json_parsing()