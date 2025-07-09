"""
Evolutionary Improvement Engine - Applies AlphaEvolve concepts to system improvements

Based on DeepMind's AlphaEvolve architecture for continuous system evolution.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import ast
import random
import hashlib
import re

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager
from scripts.real_improvement_validator import RealImprovementValidator
from scripts.improvement_needs_analyzer import ImprovementNeedsAnalyzer
from scripts.system_aligned_diversity_enhancer import SystemAlignedDiversityEnhancer


class EvolutionaryImprovementEngine:
    """Implements AlphaEvolve-inspired evolutionary improvement system."""
    
    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.ai_client = HTTPAIClient()
        self.validator = RealImprovementValidator()
        self.needs_analyzer = ImprovementNeedsAnalyzer(state_manager)
        self.diversity_enhancer = SystemAlignedDiversityEnhancer()
        
        # Evolutionary components
        self.population_size = 10
        self.elite_size = 3
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7  # Probability of crossover vs mutation
        self.tournament_size = 3   # For tournament selection
        
        # Track population of improvements
        self.improvement_population = []
        self.generation_count = 0
        self.fitness_history = []
        
        # Initialize type tracking
        self.type_history = []
        self.last_5_types = []
        
    async def initialize_population(self):
        """Initialize the first generation of improvements."""
        print("Initializing improvement population...")
        
        # Analyze system needs
        needs = await self.needs_analyzer.analyze_improvement_needs()
        
        # Generate initial population
        for i in range(self.population_size):
            improvement = await self._generate_improvement_candidate(needs)
            if improvement:
                self.improvement_population.append(improvement)
                
        print(f"Initialized population with {len(self.improvement_population)} improvements")
        
    async def _generate_improvement_candidate(self, needs: List[Dict]) -> Optional[Dict]:
        """Generate a single improvement candidate."""
        # Select improvement type based on needs or randomly
        if needs and random.random() < 0.7:  # 70% chance to address specific need
            need = random.choice(needs[:5])  # Pick from top 5 needs
            improvement_type = need['type']
            context = need
        else:
            # Force diversity - avoid repeating same type
            available_types = ['tool', 'workflow', 'optimization', 'error_fix', 'analysis', 
                              'integration', 'monitoring', 'documentation', 'testing', 'security']
            
            # Remove recently used types
            for recent_type in self.last_5_types[-3:]:
                if recent_type in available_types and len(available_types) > 3:
                    available_types.remove(recent_type)
            
            # Select type with bias toward unexplored
            if random.random() < 0.7 and available_types:  # 70% chance to force diversity
                improvement_type = random.choice(available_types)
            else:
                improvement_type = random.choice(['tool', 'workflow', 'optimization', 'error_fix', 'analysis'])
            context = None
            
        # Track type for diversity
        self.last_5_types.append(improvement_type)
        if len(self.last_5_types) > 5:
            self.last_5_types.pop(0)
        
        # Generate improvement using LLM
        prompt = self._build_generation_prompt(improvement_type, context)
        
        try:
            response = await self.ai_client.generate_enhanced_response(
                prompt,
                model=None  # Let it use the best available
            )
            
            # Parse and validate improvement
            improvement = self._parse_improvement_response(response, improvement_type)
            if improvement:
                improvement['generation'] = self.generation_count
                improvement['fitness'] = 0.0  # Will be evaluated later
                improvement['id'] = hashlib.md5(f"{improvement['name']}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
                
            return improvement
            
        except Exception as e:
            self.logger.error(f"Error generating improvement candidate: {e}")
            return None
            
    def _build_generation_prompt(self, improvement_type: str, context: Optional[Dict]) -> str:
        """Build prompt for generating improvements."""
        # Get diversity report to understand current patterns
        diversity_report = self.diversity_enhancer.get_diversity_report(self.improvement_population)
        
        # Extract overused patterns
        overused_patterns = []
        if 'most_common_patterns' in diversity_report and diversity_report['most_common_patterns']:
            # Get patterns that appear too frequently
            for pattern, count in diversity_report['most_common_patterns'][:3]:
                if count > len(self.improvement_population) * 0.3:
                    overused_patterns.append(pattern)
        
        # Get suggested patterns from system knowledge
        suggested_patterns = diversity_report.get('missing_system_patterns', [])[:5]
        
        base_prompt = f"""Generate a {improvement_type} improvement for an AI chat assistant system.

System Context:
- Self-improving AI system with tool generation capabilities
- Uses LLMs for natural language processing
- Has workflow orchestration and error handling
- Supports context-aware improvements

IMPORTANT DIVERSITY REQUIREMENT:
- Current diversity score: {diversity_report.get('diversity_score', 0):.2f}
- Avoid overused patterns: {', '.join(overused_patterns) if overused_patterns else 'none'}
- Consider incorporating: {', '.join(suggested_patterns) if suggested_patterns else 'explore new concepts'}
- Be creative and explore novel solution spaces
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
        
        return base_prompt
        
    def _parse_improvement_response(self, response: Any, improvement_type: str) -> Optional[Dict]:
        """Parse LLM response into improvement structure with robust error handling."""
        try:
            # Handle dict response
            if isinstance(response, dict):
                response_text = response.get('content', '')
            else:
                response_text = str(response)
            
            # Clean the response
            response_text = response_text.strip()
            
            # Strategy 1: Direct JSON parse
            try:
                data = json.loads(response_text)
                if isinstance(data, dict) and ('name' in data or 'Name' in data):
                    return self._normalize_and_finalize(data, improvement_type)
            except:
                pass
            
            # Strategy 2: Extract from markdown
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    if isinstance(data, dict):
                        return self._normalize_and_finalize(data, improvement_type)
                except:
                    pass
            
            # Strategy 3: Find JSON object with brace balancing
            start = response_text.find('{')
            if start >= 0:
                depth = 0
                end = start
                in_string = False
                escape = False
                
                for i in range(start, len(response_text)):
                    char = response_text[i]
                    
                    if escape:
                        escape = False
                        continue
                    
                    if char == '\\':
                        escape = True
                        continue
                    
                    if char == '"' and not escape:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                
                if end > start:
                    try:
                        json_str = response_text[start:end]
                        # Fix common JSON issues
                        json_str = self._fix_json_string(json_str)
                        data = json.loads(json_str)
                        if isinstance(data, dict):
                            return self._normalize_and_finalize(data, improvement_type)
                    except Exception as e:
                        self.logger.debug(f"JSON parse failed: {e}")
            
            # Strategy 4: Extract key fields and build JSON
            result = self._extract_fields_from_text(response_text, improvement_type)
            if result and 'name' in result:
                return self._finalize_improvement(result, improvement_type)
            
            # Strategy 5: Generate fallback improvement
            self.logger.warning("All parsing strategies failed, generating fallback")
            return self._generate_fallback_improvement(improvement_type)
            
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return self._generate_fallback_improvement(improvement_type)
    
    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix newlines in strings
        lines = json_str.split('\n')
        fixed_lines = []
        in_string = False
        
        for line in lines:
            # Simple string detection (not perfect but helps)
            quote_count = line.count('"') - line.count('\\"')
            if quote_count % 2 == 1:
                in_string = not in_string
            
            if in_string and not line.strip().endswith('"'):
                line = line.rstrip() + ' '  # Continue string on next line
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _normalize_and_finalize(self, data: Dict, improvement_type: str) -> Dict:
        """Normalize field names and finalize improvement."""
        # Normalize field names (handle Type vs type, etc.)
        normalized = {}
        for key, value in data.items():
            normalized_key = key.lower()
            normalized[normalized_key] = value
        
        return self._finalize_improvement(normalized, improvement_type)
    
    def _extract_fields_from_text(self, text: str, improvement_type: str) -> Dict:
        """Extract improvement fields from unstructured text."""
        result = {'type': improvement_type}
        
        # Patterns for common fields
        patterns = {
            'name': [
                r'"name"\s*:\s*"([^"]+)"',
                r'Name:\s*([^\n]+)',
                r'improvement\s+name:\s*([^\n]+)',
            ],
            'description': [
                r'"description"\s*:\s*"([^"]+)"',
                r'Description:\s*([^\n]+)',
                r'What\s+it\s+does:\s*([^\n]+)',
            ],
            'approach': [
                r'"approach"\s*:\s*"([^"]+)"',
                r'Approach:\s*([^\n]+)',
                r'How\s+to\s+implement:\s*([^\n]+)',
            ]
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result[field] = match.group(1).strip()
                    break
        
        return result if 'name' in result else {}
    
    def _generate_fallback_improvement(self, improvement_type: str) -> Dict:
        """Generate a valid fallback improvement when parsing fails."""
        import time
        timestamp = int(time.time())
        
        fallbacks = {
            'tool': {
                'name': f'error_recovery_tool_{timestamp}',
                'description': 'Automated error recovery and retry tool',
                'approach': 'Implements exponential backoff and smart error detection'
            },
            'workflow': {
                'name': f'resilient_workflow_{timestamp}',
                'description': 'Fault-tolerant workflow with automatic recovery',
                'approach': 'Uses checkpointing and state recovery mechanisms'
            },
            'optimization': {
                'name': f'performance_optimizer_{timestamp}',
                'description': 'Intelligent performance optimization system',
                'approach': 'Profiles code and applies targeted optimizations'
            },
            'error_fix': {
                'name': f'auto_error_fixer_{timestamp}',
                'description': 'Automatic error detection and correction',
                'approach': 'Analyzes error patterns and applies fixes'
            },
            'analysis': {
                'name': f'system_analyzer_{timestamp}',
                'description': 'Comprehensive system analysis tool',
                'approach': 'Collects metrics and generates insights'
            }
        }
        
        base = fallbacks.get(improvement_type, fallbacks['tool'])
        base['type'] = improvement_type
        return self._finalize_improvement(base, improvement_type)

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
        import re
        class_name = re.sub(r'[^a-zA-Z0-9]', '', improvement['name'].title())
        
        code = f'''"""
{improvement['name']} - {improvement['description']}
Generated by CWMAI Evolutionary System
"""

import logging
from typing import Dict, Any


class {class_name}:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "{improvement['name']}"
        
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # {improvement['approach']}
        self.logger.info(f"Applying {{self.name}}")
        return {{'success': True, 'improvement': self.name}}
'''
        return code
        
    async def evaluate_fitness(self, improvement: Dict) -> float:
        """Evaluate the fitness of an improvement (AlphaEvolve-style)."""
        fitness_score = 0.0
        
        # 1. Code quality evaluation
        if improvement.get('code'):
            try:
                # Check if code is parseable
                ast.parse(improvement['code'])
                fitness_score += 0.2
                
                # Check code length (complexity indicator)
                code_lines = len(improvement['code'].split('\n'))
                if code_lines > 50:  # Substantial implementation
                    fitness_score += 0.2
                    
                # Check for error handling
                if 'try:' in improvement['code'] and 'except' in improvement['code']:
                    fitness_score += 0.1
                    
            except SyntaxError:
                fitness_score -= 0.5  # Penalize invalid code
                
        # 2. Validation check
        is_valid, issues = await self.validator.validate_improvement(improvement)
        if is_valid:
            fitness_score += 0.3
        else:
            fitness_score -= 0.1 * len(issues)
            
        # 3. System need alignment
        needs = await self.needs_analyzer.analyze_improvement_needs()
        for need in needs[:5]:  # Check top 5 needs
            if need['type'] == improvement['type']:
                fitness_score += 0.2
                break
                
        # 4. Metrics clarity
        if improvement.get('metrics') and len(improvement['metrics']) > 0:
            fitness_score += 0.1
            
        # 5. Integration feasibility
        if improvement.get('integration_points'):
            fitness_score += 0.1
            
        # 6. Add novelty bonus for diverse solutions
        novelty_bonus = self.diversity_enhancer.calculate_novelty_bonus(improvement)
        fitness_score += novelty_bonus
        
        # Normalize to 0-1 range
        fitness_score = max(0.0, min(1.0, fitness_score))
        improvement['fitness'] = fitness_score
        
        return fitness_score
        
    async def evolve_generation(self):
        """Evolve the population to the next generation."""
        self.generation_count += 1
        print(f"\n=== Generation {self.generation_count} Evolution ===")
        
        # Evaluate fitness for all improvements
        for improvement in self.improvement_population:
            if improvement['fitness'] == 0.0:  # Not yet evaluated
                improvement['fitness'] = await self.evaluate_fitness(improvement)
                
        # Sort by fitness
        self.improvement_population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Log fitness statistics
        fitness_scores = [imp['fitness'] for imp in self.improvement_population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        best_fitness = max(fitness_scores) if fitness_scores else 0
        self.fitness_history.append({
            'generation': self.generation_count,
            'average_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'population_size': len(self.improvement_population)
        })
        
        if fitness_scores:
            print(f"Population fitness - Avg: {avg_fitness:.3f}, Best: {best_fitness:.3f}")
        else:
            self.logger.warning("No improvements in population to evaluate")
        
        # Handle empty population
        if not self.improvement_population:
            self.logger.warning("Population is empty, regenerating...")
            await self.initialize_population()
            return
            
        # Select elite improvements
        elite = self.improvement_population[:min(self.elite_size, len(self.improvement_population))]
        print(f"Elite improvements: {[e['name'] for e in elite]}")
        
        # Generate new population
        new_population = elite.copy()  # Keep elite
        
        # Fill rest of population with mutations and crossovers
        # Ensure we maintain population size
        attempts = 0
        while len(new_population) < self.population_size and attempts < 20:
            attempts += 1
            
            # Try different strategies
            if attempts % 4 == 0:
                # Generate new improvement
                improvement = await self._generate_improvement_candidate([])
                if improvement:
                    new_population.append(improvement)
            elif attempts % 4 == 1 and elite:
                # Mutate random elite
                parent = random.choice(elite)
                mutant = await self._mutate(parent)
                if mutant:
                    new_population.append(mutant)
            elif attempts % 4 == 2 and len(elite) >= 2:
                # Crossover
                p1, p2 = random.sample(elite, 2)
                child = await self._crossover(p1, p2)
                if child:
                    new_population.append(child)
            else:
                # Clone and slightly modify elite
                if elite:
                    clone = elite[attempts % len(elite)].copy()
                    clone['id'] = hashlib.md5(f"{clone['name']}_clone_{attempts}".encode()).hexdigest()[:8]
                    clone['name'] = f"{clone['name']}_v{attempts}"
                    clone['fitness'] = 0.0
                    new_population.append(clone)
                    
        # Enforce diversity in the new population
        diversity_score = self.diversity_enhancer.calculate_diversity_score(new_population)
        print(f"Population diversity before enforcement: {diversity_score:.3f}")
        
        if diversity_score < 0.5:  # Low diversity threshold
            new_population = self.diversity_enhancer.enforce_diversity(new_population, target_diversity=0.6)
            diversity_score = self.diversity_enhancer.calculate_diversity_score(new_population)
            print(f"Population diversity after enforcement: {diversity_score:.3f}")
            
        # Generate diversity report
        report = self.diversity_enhancer.get_diversity_report(new_population)
        print(f"Clusters found: {report.get('num_clusters', 0)}")
        print(f"Unique patterns: {report.get('unique_patterns', 0)}")
        if 'most_common_patterns' in report and report['most_common_patterns']:
            patterns = [f"{p[0]}({p[1]})" for p in report['most_common_patterns'][:5]]
            print(f"Common patterns: {', '.join(patterns)}")
        if report.get('recommendations'):
            print(f"Diversity recommendations: {', '.join(report['recommendations'])}")
                    
        self.improvement_population = new_population
        
    async def _crossover(self, parent1: Dict, parent2: Dict) -> Optional[Dict]:
        """Create a child improvement by combining two parents."""
        try:
            # Get diversity context
            diversity_report = self.diversity_enhancer.get_diversity_report(self.improvement_population)
            
            # Get patterns to avoid and explore
            overused = [p[0] for p in diversity_report.get('most_common_patterns', [])[:3]]
            suggested = diversity_report.get('missing_system_patterns', [])[:3]
            
            prompt = f"""Combine these improvements: {parent1.get("name", "unknown")} and {parent2.get("name", "unknown")}. 
            
DIVERSITY REQUIREMENT: 
- Create a hybrid that explores NEW patterns
- Avoid overused concepts: {', '.join(overused) if overused else 'none'}
- Consider incorporating: {', '.join(suggested) if suggested else 'novel approaches'}
- Focus on innovative combinations that bridge different solution spaces

Return ONLY a JSON with: name, type, description, approach, metrics, integration_points"""
            
            response = await self.ai_client.generate_enhanced_response(prompt)
            child = self._parse_improvement_response(response, parent1.get('type', 'optimization'))
            if child:
                child['parents'] = [parent1.get('id', 'p1'), parent2.get('id', 'p2')]
                child['generation'] = self.generation_count
                child['fitness'] = 0.0
                child['id'] = hashlib.md5(f"{child['name']}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            return child
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            # Return a simple combination as fallback
            return {
                'name': f"{parent1.get('name', 'p1')}_{parent2.get('name', 'p2')}_hybrid",
                'type': parent1.get('type', 'optimization'),
                'description': f"Combination of {parent1.get('name', 'p1')} and {parent2.get('name', 'p2')}",
                'approach': 'Combined approach',
                'metrics': ['combined_metric'],
                'integration_points': ['system'],
                'parents': [parent1.get('id', 'p1'), parent2.get('id', 'p2')],
                'generation': self.generation_count,
                'fitness': 0.0,
                'id': hashlib.md5(f"hybrid_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
                'code': self._generate_code_for_improvement({
                    'name': f"{parent1.get('name', 'p1')}_{parent2.get('name', 'p2')}_hybrid",
                    'description': f"Combination of {parent1.get('name', 'p1')} and {parent2.get('name', 'p2')}",
                    'approach': 'Combined approach'
                })
            }
            
    async def _mutate(self, parent: Dict) -> Optional[Dict]:
        """Create a mutated version of an improvement."""
        mutation_types = [
            "Add error recovery mechanisms",
            "Optimize for performance",
            "Add new features",
            "Improve integration points",
            "Enhance metrics collection",
            "Add AI-powered enhancements",
            "Improve scalability",
            "Add real-time capabilities",
            "Enhance security features",
            "Add automation capabilities"
        ]
        
        mutation = random.choice(mutation_types)
        
        # Get diversity context
        diversity_report = self.diversity_enhancer.get_diversity_report(self.improvement_population)
        
        # Get patterns to avoid and explore
        overused = [p[0] for p in diversity_report.get('most_common_patterns', [])[:3]]
        suggested = diversity_report.get('missing_system_patterns', [])[:5]
        
        prompt = f"""Mutate {parent.get("name", "unnamed")} by: {mutation}. 

DIVERSITY REQUIREMENT: 
- Transform this into a solution that explores DIFFERENT patterns
- Avoid overused concepts: {', '.join(overused) if overused else 'none'}
- Consider incorporating: {', '.join(suggested) if suggested else 'unexplored patterns'}
- Current clusters: {diversity_report.get('num_clusters', 0)}

Return ONLY a JSON with: name, type, description, approach, metrics, integration_points"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            mutant = self._parse_improvement_response(response, parent.get('type', 'optimization'))
            if mutant:
                mutant['parent'] = parent.get('id', 'unknown')
                mutant['mutation'] = mutation
                mutant['generation'] = self.generation_count
                mutant['fitness'] = 0.0
                # Ensure mutant has an id
                if 'id' not in mutant:
                    mutant['id'] = hashlib.md5(f"{mutant.get('name', 'mutant')}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            return mutant
        except Exception as e:
            self.logger.error(f"Mutation failed: {e}")
            return None
            
    async def apply_best_improvement(self) -> Dict[str, Any]:
        """Apply the best improvement from current generation."""
        if not self.improvement_population:
            return {'success': False, 'reason': 'No improvements in population'}
            
        # Get best improvement
        best_improvement = max(self.improvement_population, key=lambda x: x['fitness'])
        
        print(f"Applying best improvement: {best_improvement['name']} (fitness: {best_improvement['fitness']:.3f})")
        
        # Create the actual improvement file
        if best_improvement['type'] == 'tool':
            file_path = Path(f"scripts/custom_tools/{best_improvement['name']}.py")
        elif best_improvement['type'] == 'workflow':
            file_path = Path(f"scripts/workflows/{best_improvement['name']}.py")
        else:
            file_path = Path(f"scripts/improvements/{best_improvement['name']}.py")
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the improvement
        with open(file_path, 'w') as f:
            f.write(best_improvement['code'])
            
        # Update system state
        state = self.state_manager.get_state()
        if 'evolutionary_improvements' not in state:
            state['evolutionary_improvements'] = []
            
        state['evolutionary_improvements'].append({
            'id': best_improvement.get('id', hashlib.md5(f"{best_improvement['name']}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]),
            'name': best_improvement['name'],
            'type': best_improvement['type'],
            'generation': best_improvement.get('generation', self.generation_count),
            'fitness': best_improvement.get('fitness', 0.0),
            'applied_at': datetime.now(timezone.utc).isoformat(),
            'file_path': str(file_path)
        })
        
        self.state_manager.update_state(state)
        
        return {
            'success': True,
            'improvement': best_improvement,
            'file_path': str(file_path)
        }
    
    def learn_from_failure(self, improvement: Dict, error: str):
        """Learn from verification failures to improve future generations."""
        if not hasattr(self, 'failure_patterns'):
            self.failure_patterns = []
        
        # Analyze the failure
        failure_info = {
            'error': error,
            'type': improvement.get('type'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Common syntax error patterns
        if 'expected an indented block' in error:
            failure_info['pattern'] = 'empty_block'
            failure_info['fix'] = 'Add pass or implementation to empty blocks'
        elif 'SyntaxError' in error:
            failure_info['pattern'] = 'syntax'
            failure_info['fix'] = 'Improve code structure'
        elif 'NameError' in error:
            failure_info['pattern'] = 'undefined'
            failure_info['fix'] = 'Add missing imports or definitions'
        
        self.failure_patterns.append(failure_info)
        
        # Update generation prompt based on failures
        if len(self.failure_patterns) > 3:
            recent_patterns = [f['pattern'] for f in self.failure_patterns[-5:]]
            
            # Add failure-specific instructions to next generation
            self.generation_hints = []
            if 'empty_block' in recent_patterns:
                self.generation_hints.append("NEVER leave empty function bodies - always add implementation")
            if 'syntax' in recent_patterns:
                self.generation_hints.append("ENSURE proper Python syntax with correct indentation")
            if 'undefined' in recent_patterns:
                self.generation_hints.append("INCLUDE all necessary imports at the top")
        
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolutionary progress."""
        return {
            'current_generation': self.generation_count,
            'population_size': len(self.improvement_population),
            'fitness_history': self.fitness_history,
            'best_improvements': sorted(
                self.improvement_population,
                key=lambda x: x['fitness'],
                reverse=True
            )[:3]
        }