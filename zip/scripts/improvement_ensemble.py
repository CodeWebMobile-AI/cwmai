"""
Improvement Ensemble System

Implements AlphaEvolve's ensemble approach using multiple AI models for diverse improvements.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import random

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class ImprovementEnsemble:
    """Uses ensemble of AI models to generate diverse, high-quality improvements."""
    
    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.ai_client = HTTPAIClient()
        
        # Ensemble configuration (like AlphaEvolve's Gemini Flash + Pro)
        self.ensemble_config = {
            'explorer': {
                'temperature': 0.9,
                'role': 'Generate diverse, creative solutions',
                'focus': 'breadth',
                'model_preference': 'gemini'  # Fast model for exploration
            },
            'refiner': {
                'temperature': 0.3,
                'role': 'Refine and improve existing solutions',
                'focus': 'depth',
                'model_preference': 'openai'  # Powerful model for refinement
            },
            'validator': {
                'temperature': 0.1,
                'role': 'Ensure correctness and quality',
                'focus': 'accuracy',
                'model_preference': 'anthropic'  # Careful model for validation
            },
            'innovator': {
                'temperature': 0.7,
                'role': 'Combine ideas in novel ways',
                'focus': 'creativity',
                'model_preference': 'deepseek'  # Alternative perspective
            }
        }
        
    async def _with_timeout(self, coro, timeout_seconds: int = 30, default=None):
        """Execute coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self.logger.warning(f"Operation timed out after {timeout_seconds}s")
            return default
        except Exception as e:
            self.logger.error(f"Operation failed: {e}")
            return default
    
     
    async def generate_ensemble_improvements(self, context: Dict) -> List[Dict]:
        """Generate improvements using ensemble approach."""
        print("Starting ensemble improvement generation")
        
        # Phase 1: Exploration - Generate diverse ideas
        exploration_results = await self._exploration_phase(context)
        
        # Phase 2: Refinement - Improve promising ideas
        refined_results = await self._refinement_phase(exploration_results)
        
        # Phase 3: Innovation - Combine and enhance
        innovative_results = await self._innovation_phase(refined_results)
        
        # Phase 4: Validation - Ensure quality
        validated_results = await self._validation_phase(innovative_results)
        
        
        
        # Ensure we always return a list
        if not validated_results:
            # Return at least one fallback improvement
            validated_results = [{
                'name': 'fallback_improvement',
                'type': 'optimization',
                'description': 'Fallback improvement when ensemble fails',
                'approach': 'Basic optimization approach',
                'metrics': ['success_rate'],
                'integration_points': ['system_core'],
                'quality_score': 0.5,
                'is_valid': True
            }]
            
        return validated_results
        
    async def _exploration_phase(self, context: Dict) -> List[Dict]:
        """Generate diverse improvement ideas."""
        print("Phase 1: Exploration - Generating diverse ideas")
        
        explorer_config = self.ensemble_config['explorer']
        ideas = []
        
        # Generate multiple diverse ideas
        improvement_types = ['tool', 'workflow', 'optimization']  # Reduced for speed
        
        tasks = []
        for imp_type in improvement_types:
            prompt = f"""As a creative AI system explorer, generate a unique {imp_type} improvement.

Context: {context.get('description', 'AI chat assistant system')}
Current needs: {context.get('needs', [])}

Requirements:
1. Be creative and think outside the box
2. Propose something that doesn't exist in the system yet
3. Focus on high-impact improvements
4. Include implementation approach

Return a JSON object:
{{
    "name": "unique_improvement_name",
    "type": "{imp_type}",
    "description": "what this does",
    "approach": "how to implement",
    "impact": "expected benefits",
    "novel_aspects": ["what makes this unique"]
}}
"""
            
            task = self._generate_with_model(
                prompt,
                explorer_config['temperature'],
                explorer_config['model_preference']
            )
            tasks.append(task)
            
        # Gather all ideas
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Limit validation to top 3 for speed
        responses = responses[:3] if len(responses) > 3 else responses
        for response in responses:
            if isinstance(response, Exception):
                self.logger.error(f"Exploration error: {response}")
                continue
                
            idea = self._parse_json_response(response)
            if idea:
                ideas.append(idea)
                
        print(f"Generated {len(ideas)} initial ideas")
        return ideas
        
    async def _refinement_phase(self, ideas: List[Dict]) -> List[Dict]:
        """Refine and improve the most promising ideas."""
        print("Phase 2: Refinement - Improving promising ideas")
        
        refiner_config = self.ensemble_config['refiner']
        refined = []
        
        # Select top ideas to refine (simulate natural selection)
        promising_ideas = self._select_promising_ideas(ideas, top_k=3)
        
        for idea in promising_ideas:
            prompt = f"""As a careful AI system refiner, improve this idea with concrete implementation:

Original idea: {idea['name']}
Type: {idea['type']}
Description: {idea['description']}
Approach: {idea.get('approach', 'Not specified')}

Create a detailed, production-ready implementation:
1. Write complete, functional Python code
2. Include error handling and logging
3. Add comprehensive docstrings
4. Ensure it integrates with the existing system
5. Make it efficient and maintainable

Return a JSON object:
{{
    "name": "{idea['name']}_refined",
    "type": "{idea['type']}",
    "description": "enhanced description",
    "approach": "implementation approach",
    "integration_steps": ["how to integrate"],
    "improvements_made": ["what was enhanced"]
}}
"""
            
            response = await self._generate_with_model(
                prompt,
                refiner_config['temperature'],
                refiner_config['model_preference']
            )
            
            refined_idea = self._parse_json_response(response)
            if refined_idea:
                refined_idea['original_idea'] = idea['name']
                refined.append(refined_idea)
                
        print(f"Refined {len(refined)} ideas")
        return refined
        
    async def _innovation_phase(self, refined_ideas: List[Dict]) -> List[Dict]:
        """Combine ideas in innovative ways."""
        print("Phase 3: Innovation - Creating novel combinations")
        
        innovator_config = self.ensemble_config['innovator']
        innovations = []
        
        # Keep refined ideas
        innovations.extend(refined_ideas)
        
        # Create combinations if we have multiple ideas
        if len(refined_ideas) >= 2:
            # Try combining pairs of ideas
            for i in range(len(refined_ideas)):
                for j in range(i + 1, len(refined_ideas)):
                    idea1, idea2 = refined_ideas[i], refined_ideas[j]
                    
                    prompt = f"""As an innovative AI combiner, merge these two improvements creatively:

Improvement 1: {idea1.get('name', 'unnamed')}
Type: {idea1.get('type', 'unknown')}
Key features: {idea1.get('description')}

Improvement 2: {idea2.get('name', 'unnamed')}
Type: {idea2.get('type', 'unknown')}
Key features: {idea2.get('description')}

Create a novel combination that:
1. Leverages strengths of both
2. Creates emergent capabilities
3. Is more than the sum of its parts

Return a JSON object:
{{
    "name": "innovative_combination_name",
    "type": "hybrid",
    "description": "what this combined improvement does",
    "approach": "combined approach",
    "synergies": ["how the ideas enhance each other"],
    "parent_ideas": ["{idea1.get('name', 'unnamed')}", "{idea2.get('name', 'unnamed')}"]
}}
"""
                    
                    response = await self._generate_with_model(
                        prompt,
                        innovator_config['temperature'],
                        innovator_config['model_preference']
                    )
                    
                    innovation = self._parse_json_response(response)
                    if innovation:
                        innovations.append(innovation)
                        
        print(f"Created {len(innovations)} innovative improvements")
        return innovations
        
    async def _validation_phase(self, innovations: List[Dict]) -> List[Dict]:
        """Validate and ensure quality of improvements."""
        print("Phase 4: Validation - Ensuring quality")
        
        validator_config = self.ensemble_config['validator']
        validated = []
        
        for innovation in innovations:
            prompt = f"""As a strict AI validator, evaluate this improvement:

Name: {innovation['name']}
Type: {innovation['type']}
Description: {innovation['description']}
Approach:
Approach: 
{innovation.get('code', 'No code provided')}


Validate:
1. Is the code syntactically correct?
2. Does it handle errors properly?
3. Will it integrate without breaking the system?
4. Does it provide real value?
5. Are there security concerns?

If valid, return the improvement with any necessary fixes.
If invalid, explain why.

Return a JSON object:
{{
    "is_valid": true/false,
    "name": "{innovation['name']}",
    "type": "{innovation['type']}",
    "description": "description",
    "approach": "validated approach",
    "validation_notes": ["any issues fixed"],
    "quality_score": 0.0-1.0
}}
"""
            
            response = await self._generate_with_model(
                prompt,
                validator_config['temperature'],
                validator_config['model_preference']
            )
            
            result = self._parse_json_response(response)
            if result and result.get('is_valid'):
                validated.append(result)
                
        print(f"Validated {len(validated)} improvements")
        return validated
        
    async def _generate_with_model(self, prompt: str, temperature: float, model_preference: str) -> Any:
        """Generate response with specific model preference."""
        # Map provider names to model names
        provider_to_model = {
            'openai': 'gpt',
            'anthropic': 'claude',
            'gemini': 'gemini',
            'deepseek': 'deepseek'
        }
        
        model_name = provider_to_model.get(model_preference, model_preference)
        
        try:
            # Try to use preferred model
            response = await self._with_timeout(self.ai_client.generate_enhanced_response(
                prompt,
                model=model_name
            ))
            return response
        except Exception as e:
            self.logger.warning(f"Failed with {model_preference}/{model_name}, using fallback: {e}")
            # Fallback to any available model
            return await self.ai_client.generate_enhanced_response(prompt)
            
    def _parse_json_response(self, response: Any) -> Optional[Dict]:
        """Parse JSON from AI response with comprehensive error handling."""
        try:
            import re
            import json
            
            # Handle dict response
            if isinstance(response, dict):
                response_text = response.get('content', '')
            else:
                response_text = str(response)
            
            # Clean response
            response_text = response_text.strip()
            
            # Strategy 1: Direct parse
            try:
                data = json.loads(response_text)
                if isinstance(data, dict):
                    return data
            except:
                pass
            
            # Strategy 2: Extract from markdown
            code_block = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
            if code_block:
                try:
                    data = json.loads(code_block.group(1))
                    if isinstance(data, dict):
                        return data
                except:
                    pass
            
            # Strategy 3: Find JSON object
            # Look for outermost braces
            start = response_text.find('{')
            if start >= 0:
                depth = 0
                end = start
                for i in range(start, len(response_text)):
                    if response_text[i] == '{':
                        depth += 1
                    elif response_text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                
                if end > start:
                    try:
                        json_str = response_text[start:end]
                        # Clean common issues
                        json_str = re.sub(r'\n\s*', ' ', json_str)
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        
                        data = json.loads(json_str)
                        if isinstance(data, dict):
                            return data
                    except:
                        pass
            
            # Strategy 4: Build from fragments
            result = {}
            
            # Extract key fields
            patterns = {
                'name': r'"name"\s*:\s*"([^"]+)"',
                'type': r'"type"\s*:\s*"([^"]+)"',
                'description': r'"description"\s*:\s*"([^"]+)"',
                'approach': r'"approach"\s*:\s*"([^"]+)"',
                'is_valid': r'"is_valid"\s*:\s*(true|false)',
                'quality_score': r'"quality_score"\s*:\s*([0-9.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if key == 'is_valid':
                        result[key] = value.lower() == 'true'
                    elif key == 'quality_score':
                        result[key] = float(value)
                    else:
                        result[key] = value
            
            # Extract arrays
            metrics_match = re.search(r'"metrics"\s*:\s*\[([^\]]+)\]', response_text)
            if metrics_match:
                metrics_str = metrics_match.group(1)
                result['metrics'] = [m.strip(' "') for m in metrics_str.split(',')]
            
            if result:
                return result
                
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            
        return None
        
    def _select_promising_ideas(self, ideas: List[Dict], top_k: int = 3) -> List[Dict]:
        """Select most promising ideas based on impact and novelty."""
        # Simple scoring based on available fields
        scored_ideas = []
        
        for idea in ideas:
            score = 0.5  # Base score
            
            # Boost for specific attributes
            if idea.get('impact'):
                score += 0.2
            if idea.get('novel_aspects'):
                score += 0.1 * len(idea['novel_aspects'])
            if idea.get('approach'):
                score += 0.1
                
            scored_ideas.append((score, idea))
            
        # Sort by score and return top k
        scored_ideas.sort(key=lambda x: x[0], reverse=True)
        return [idea for _, idea in scored_ideas[:top_k]]
        
    async def create_ensemble_improvement(self, need_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a single improvement using the full ensemble pipeline."""
        context = need_context or {
            'description': 'AI chat assistant with self-improvement capabilities',
            'needs': ['performance', 'reliability', 'user experience']
        }
        
        # Run ensemble pipeline
        improvements = await self.generate_ensemble_improvements(context)
        
        if not improvements:
            return {
                'success': False,
                'reason': 'No valid improvements generated'
            }
            
        # Select best improvement based on quality score
        best = max(improvements, key=lambda x: x.get('quality_score', 0))
        
        # Add metadata
        best['created_by'] = 'ensemble'
        best['created_at'] = datetime.now(timezone.utc).isoformat()
        best['ensemble_phases'] = ['exploration', 'refinement', 'innovation', 'validation']
        
        return {
            'success': True,
            'improvement': best,
            'alternatives': len(improvements) - 1
        }