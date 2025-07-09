"""
Knowledge-Driven Evolutionary Engine

Integrates evolutionary algorithms with knowledge management for
intelligent, learning-based improvement generation.
"""

import asyncio
import random
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import Counter
import numpy as np

from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine
from scripts.evolutionary_knowledge_store import EvolutionaryKnowledgeStore
from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.simple_knowledge_synthesis_engine import SimpleKnowledgeSynthesisEngine
from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient
from scripts.evolutionary_code_generator_fix import RealCodeEvolutionFix, apply_real_code_generation_fixes


class KnowledgeDrivenEvolutionaryEngine(EvolutionaryImprovementEngine):
    """Enhanced evolutionary engine that uses accumulated knowledge."""
    
    def __init__(self, state_manager: StateManager):
        super().__init__(state_manager)
        
        # Knowledge components
        self.improvement_store = EvolutionaryKnowledgeStore()
        self.research_store = ResearchKnowledgeStore()
        self.synthesis_engine = SimpleKnowledgeSynthesisEngine()
        
        # Knowledge-driven parameters
        self.min_novelty_threshold = 0.3  # Minimum novelty to accept
        self.knowledge_weight = 0.4  # Weight of knowledge in fitness
        self.exploration_bonus = 0.2  # Bonus for exploring gaps
        
        # Tracking
        self.generation_insights = []
        self.avoided_duplicates = 0
        self.knowledge_guided_mutations = 0
        
        # Initialize attributes from base class
        self.type_history = []
        self.last_5_types = []
        
        # Initialize real code generation validator
        self.code_validator = apply_real_code_generation_fixes()
        self.logger.info("Real code generation validator initialized")
    
    def _generate_code_for_improvement(self, improvement: Dict) -> str:
        """Generate REAL implementation code, not placeholders."""
        # First try the base implementation
        placeholder_code = super()._generate_code_for_improvement(improvement)
        
        # Validate if it's real code
        is_valid, error_msg, test_results = self.code_validator.validate_generated_code(
            placeholder_code, improvement
        )
        
        if is_valid:
            return placeholder_code
            
        # Not valid - generate real implementation
        self.logger.warning(f"Generated placeholder code: {error_msg}")
        
        # For now, mark the improvement as needing real code generation
        # This will be handled in the async context
        improvement['needs_real_code'] = True
        improvement['code_generation_error'] = error_msg
        
        # Return placeholder for now - will be regenerated later
        return placeholder_code
    
    def _extract_code_from_response(self, response: Any) -> str:
        """Extract code from AI response."""
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        elif isinstance(response, str):
            content = response
        else:
            return ""
            
        # Look for code blocks
        import re
        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1)
            
        # Look for class definition
        class_match = re.search(r'(class\s+\w+.*?(?=\n\n|\Z))', content, re.DOTALL)
        if class_match:
            return class_match.group(1)
            
        return content
    
    async def _regenerate_real_code(self, improvement: Dict) -> bool:
        """Regenerate code to be real implementation instead of placeholder."""
        if not improvement.get('needs_real_code', False):
            return True
            
        self.logger.info(f"Regenerating real code for {improvement['name']}")
        
        # Create explicit prompt for real implementation
        prompt = self.code_validator.create_implementation_prompt(improvement)
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            if response and 'code' in response:
                real_code = response['code']
            else:
                # Extract code from response
                real_code = self._extract_code_from_response(response)
                
            # Validate the new code
            is_valid, error_msg, test_results = self.code_validator.validate_generated_code(
                real_code, improvement
            )
            
            if is_valid:
                self.logger.info(f"Generated valid implementation for {improvement['name']}")
                improvement['code'] = real_code
                improvement['needs_real_code'] = False
                improvement['validated'] = True
                improvement['test_results'] = test_results
                return True
            else:
                self.logger.warning(f"Generated code still invalid: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error generating real code: {e}")
            return False
        
    async def initialize_population(self):
        """Initialize population with knowledge-guided diversity."""
        print("🧠 Initializing knowledge-driven population...")
        
        # Get knowledge gaps and successful patterns
        knowledge_gaps = self.improvement_store.find_knowledge_gaps()
        successful_patterns = self.improvement_store.get_successful_patterns()
        
        # Analyze system needs
        needs = await self.needs_analyzer.analyze_improvement_needs()
        
        # Generate diverse initial population
        self.improvement_population = []
        
        # 1. Address top knowledge gaps (30%)
        gap_count = int(self.population_size * 0.3)
        for i in range(min(gap_count, len(knowledge_gaps))):
            gap = knowledge_gaps[i]
            improvement = await self._generate_gap_targeted_improvement(gap)
            if improvement:
                self.improvement_population.append(improvement)
                
        # 2. Build on successful patterns (30%)
        pattern_count = int(self.population_size * 0.3)
        pattern_keys = list(successful_patterns.keys())
        for i in range(min(pattern_count, len(pattern_keys))):
            pattern = random.choice(pattern_keys)
            improvement = await self._generate_pattern_based_improvement(pattern, successful_patterns[pattern])
            if improvement:
                self.improvement_population.append(improvement)
                
        # 3. Address system needs (20%)
        need_count = int(self.population_size * 0.2)
        for i in range(min(need_count, len(needs))):
            improvement = await self._generate_improvement_candidate(needs)
            if improvement:
                self.improvement_population.append(improvement)
                
        # 4. Random exploration (20%)
        while len(self.improvement_population) < self.population_size:
            improvement = await self._generate_improvement_candidate([])
            if improvement:
                self.improvement_population.append(improvement)
                
        print(f"✅ Initialized population with {len(self.improvement_population)} improvements")
        print(f"   - Gap-targeted: {gap_count}")
        print(f"   - Pattern-based: {pattern_count}")
        print(f"   - Need-based: {need_count}")
        print(f"   - Exploratory: {self.population_size - gap_count - pattern_count - need_count}")
        
        # Regenerate real code for improvements that need it
        regeneration_needed = sum(1 for imp in self.improvement_population if imp.get('needs_real_code', False))
        if regeneration_needed > 0:
            print(f"\n🔧 Regenerating real code for {regeneration_needed} improvements...")
            regenerated = 0
            for improvement in self.improvement_population:
                if improvement.get('needs_real_code', False):
                    if await self._regenerate_real_code(improvement):
                        regenerated += 1
            print(f"✅ Regenerated {regenerated}/{regeneration_needed} improvements with real code")
    
    async def evaluate_fitness(self, improvement: Dict) -> float:
        """Evaluate fitness with heavy penalty for placeholder code."""
        # Get base fitness
        base_fitness = await super().evaluate_fitness(improvement)
        
        # Check if code is validated as real implementation
        if improvement.get('validated', False):
            # Boost fitness for validated real code
            fitness = base_fitness * 1.5
        elif improvement.get('needs_real_code', False):
            # Heavy penalty for placeholder code
            fitness = base_fitness * 0.1
        else:
            # Check if code looks like placeholder
            code = improvement.get('code', '')
            if 'return {\'success\': True' in code and '# TODO' not in code:
                # Likely placeholder - validate to be sure
                is_valid, _, _ = self.code_validator.validate_generated_code(code, improvement)
                if not is_valid:
                    fitness = base_fitness * 0.1
                else:
                    fitness = base_fitness
            else:
                fitness = base_fitness
                
        # Apply knowledge-based adjustments
        if hasattr(self, 'improvement_store'):
            # Predict success based on patterns
            prediction = self.improvement_store.predict_improvement_success(improvement)
            fitness *= (0.5 + prediction['overall'] * 0.5)
            
        return fitness
        
    async def _generate_improvement_candidate(self, needs: List[Dict]) -> Optional[Dict]:
        """Generate improvement with duplicate checking."""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Generate candidate
            candidate = await super()._generate_improvement_candidate(needs)
            
            if not candidate:
                continue
                
            # Check for duplicates
            similar = self.improvement_store.find_similar_improvements(candidate, threshold=0.9)
            
            if not similar:
                # Truly novel improvement
                return candidate
            elif similar[0]["similarity"] < 0.95:
                # Different enough to keep
                candidate["inspired_by"] = similar[0]["id"]
                return candidate
            else:
                # Too similar - retry with different approach
                self.avoided_duplicates += 1
                self.logger.info(f"Avoided duplicate (similarity: {similar[0]['similarity']:.2f})")
                
        return None
        
    async def _generate_gap_targeted_improvement(self, gap: Dict) -> Optional[Dict]:
        """Generate improvement targeting specific knowledge gap."""
        prompt = f"""Generate an improvement that explores: {gap}

Focus on creating something novel that hasn't been tried before.
This is a knowledge gap in our system.

Return ONLY a JSON with: name, type, description, approach, metrics, integration_points"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            improvement = self._parse_improvement_response(response, gap.get("type", "exploration"))
            
            if improvement:
                improvement["targets_gap"] = gap
                improvement["exploration_bonus"] = self.exploration_bonus
                
            return improvement
            
        except Exception as e:
            self.logger.error(f"Gap-targeted generation failed: {e}")
            return None
            
    async def _generate_pattern_based_improvement(self, pattern: str, pattern_data: Dict) -> Optional[Dict]:
        """Generate improvement based on successful pattern."""
        examples = pattern_data.get("examples", [])
        success_rate = pattern_data.get("success_rate", 0)
        
        prompt = f"""Generate an improvement that builds on the successful pattern: {pattern}

This pattern has {success_rate:.0%} success rate.
Create a novel variation that extends this pattern in new ways.

Return ONLY a JSON with: name, type, description, approach, metrics, integration_points"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            improvement = self._parse_improvement_response(response, "pattern_based")
            
            if improvement:
                improvement["based_on_pattern"] = pattern
                improvement["pattern_success_rate"] = success_rate
                
            return improvement
            
        except Exception as e:
            self.logger.error(f"Pattern-based generation failed: {e}")
            return None
            
    async def evaluate_fitness(self, improvement: Dict) -> float:
        """Evaluate fitness with knowledge-based components."""
        # Get base fitness
        base_fitness = await super().evaluate_fitness(improvement)
        
        # Predict success based on knowledge
        prediction = self.improvement_store.predict_improvement_success(improvement)
        
        # Calculate novelty
        similar = self.improvement_store.find_similar_improvements(improvement)
        if similar:
            novelty = 1.0 - similar[0]["similarity"]
        else:
            novelty = 1.0
            
        # Check if targets knowledge gap
        gap_bonus = improvement.get("exploration_bonus", 0)
        
        # Knowledge-enhanced fitness
        knowledge_fitness = (
            base_fitness * (1 - self.knowledge_weight) +
            prediction["overall"] * self.knowledge_weight * 0.5 +
            novelty * self.knowledge_weight * 0.3 +
            gap_bonus * self.knowledge_weight * 0.2
        )
        
        # Store fitness components for analysis
        improvement["fitness_components"] = {
            "base": base_fitness,
            "predicted_success": prediction["overall"],
            "novelty": novelty,
            "gap_bonus": gap_bonus,
            "final": knowledge_fitness
        }
        
        improvement["fitness"] = knowledge_fitness
        return knowledge_fitness
        
    async def _mutate(self, parent: Dict) -> Optional[Dict]:
        """Knowledge-guided mutation."""
        # Analyze parent's patterns
        parent_patterns = self.improvement_store._extract_patterns(parent)
        
        # Get failed patterns to avoid
        failed_patterns = self.improvement_store.get_failed_patterns()
        patterns_to_avoid = set(failed_patterns.keys())
        
        # Get successful patterns to incorporate
        successful_patterns = self.improvement_store.get_successful_patterns()
        
        # Build mutation prompt
        avoid_list = [p for p in parent_patterns if p in patterns_to_avoid]
        incorporate_list = random.sample(
            list(successful_patterns.keys()), 
            min(3, len(successful_patterns))
        )
        
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
        
        prompt = f"""Mutate {parent.get("name", "unnamed")} by: {mutation}.

IMPORTANT:
- Avoid these failing patterns: {', '.join(avoid_list[:3]) if avoid_list else 'none'}
- Consider incorporating: {', '.join(incorporate_list) if incorporate_list else 'explore new approaches'}
- Current novelty in population: {len(self.improvement_population)} unique solutions

Create something genuinely different.

Return ONLY a JSON with: name, type, description, approach, metrics, integration_points"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            mutant = self._parse_improvement_response(response, parent.get('type', 'optimization'))
            
            if mutant:
                mutant['parent'] = parent.get('id', 'unknown')
                mutant['mutation'] = mutation
                mutant['generation'] = self.generation_count
                mutant['fitness'] = 0.0
                mutant['knowledge_guided'] = True
                self.knowledge_guided_mutations += 1
                
                # Ensure mutant has an id
                if 'id' not in mutant:
                    mutant['id'] = hashlib.md5(
                        f"{mutant.get('name', 'mutant')}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()[:8]
                    
            return mutant
            
        except Exception as e:
            self.logger.error(f"Knowledge-guided mutation failed: {e}")
            return None
            
    async def _crossover(self, parent1: Dict, parent2: Dict) -> Optional[Dict]:
        """Knowledge-enhanced crossover."""
        # Analyze parent patterns
        p1_patterns = set(self.improvement_store._extract_patterns(parent1))
        p2_patterns = set(self.improvement_store._extract_patterns(parent2))
        
        # Find complementary patterns
        unique_p1 = p1_patterns - p2_patterns
        unique_p2 = p2_patterns - p1_patterns
        shared = p1_patterns.intersection(p2_patterns)
        
        # Check if combination has been tried
        combined_patterns = sorted(list(p1_patterns.union(p2_patterns)))
        similar_combinations = []
        
        for imp_id, imp_data in self.improvement_store._iterate_all_improvements():
            if "patterns" in imp_data:
                imp_patterns = set(imp_data["patterns"])
                similarity = len(imp_patterns.intersection(set(combined_patterns))) / len(combined_patterns)
                if similarity > 0.7:
                    similar_combinations.append({
                        "id": imp_id,
                        "similarity": similarity,
                        "success": imp_data.get("verification", {}).get("is_deployable", False)
                    })
                    
        prompt = f"""Combine improvements: {parent1.get("name")} and {parent2.get("name")}.

Parent 1 unique strengths: {', '.join(list(unique_p1)[:3])}
Parent 2 unique strengths: {', '.join(list(unique_p2)[:3])}
Shared patterns: {', '.join(list(shared)[:3])}

Previous similar combinations: {len(similar_combinations)} found
Success rate of similar: {f"{sum(1 for s in similar_combinations if s['success'])/len(similar_combinations):.0%}" if similar_combinations else "N/A"}

Create an innovative hybrid that combines the best of both in unexpected ways.

Return ONLY a JSON with: name, type, description, approach, metrics, integration_points"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            child = self._parse_improvement_response(response, parent1.get('type', 'optimization'))
            
            if child:
                child['parents'] = [parent1.get('id', 'p1'), parent2.get('id', 'p2')]
                child['generation'] = self.generation_count
                child['fitness'] = 0.0
                child['crossover_type'] = 'knowledge_enhanced'
                
                # Store pattern combination
                child['combined_patterns'] = combined_patterns
                
                if 'id' not in child:
                    child['id'] = hashlib.md5(
                        f"{child['name']}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()[:8]
                    
            return child
            
        except Exception as e:
            self.logger.error(f"Knowledge-enhanced crossover failed: {e}")
            return super()._crossover(parent1, parent2)  # Fallback
            
    async def evolve_generation(self):
        """Evolve with knowledge tracking."""
        generation_start = datetime.now()
        
        # Store pre-evolution state
        pre_evolution_patterns = Counter()
        for imp in self.improvement_population:
            patterns = self.improvement_store._extract_patterns(imp)
            pre_evolution_patterns.update(patterns)
            
        # Perform evolution
        await super().evolve_generation()
        
        # Store improvements in knowledge base
        for imp in self.improvement_population:
            if imp.get('fitness', 0) > 0:
                verification = {
                    "is_deployable": imp.get('fitness', 0) > 0.5,
                    "fitness_score": imp.get('fitness', 0)
                }
                
                context = {
                    "generation": self.generation_count,
                    "population_size": len(self.improvement_population),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self.improvement_store.store_improvement(imp, verification, context)
                
        # Analyze generation results
        post_evolution_patterns = Counter()
        for imp in self.improvement_population:
            patterns = self.improvement_store._extract_patterns(imp)
            post_evolution_patterns.update(patterns)
            
        # Generate insights
        insight = {
            "generation": self.generation_count,
            "duration": (datetime.now() - generation_start).total_seconds(),
            "population_size": len(self.improvement_population),
            "avoided_duplicates": self.avoided_duplicates,
            "knowledge_guided_mutations": self.knowledge_guided_mutations,
            "pattern_changes": {
                "new": list(set(post_evolution_patterns) - set(pre_evolution_patterns)),
                "lost": list(set(pre_evolution_patterns) - set(post_evolution_patterns)),
                "persistent": list(set(pre_evolution_patterns).intersection(set(post_evolution_patterns)))
            },
            "fitness_stats": {
                "min": min(imp.get('fitness', 0) for imp in self.improvement_population),
                "max": max(imp.get('fitness', 0) for imp in self.improvement_population),
                "avg": np.mean([imp.get('fitness', 0) for imp in self.improvement_population])
            }
        }
        
        self.generation_insights.append(insight)
        
        # Synthesize knowledge periodically
        if self.generation_count % 5 == 0:
            await self._synthesize_and_apply_knowledge()
            
    async def _synthesize_and_apply_knowledge(self):
        """Synthesize insights and apply to evolution strategy."""
        print(f"\n🧪 Synthesizing knowledge at generation {self.generation_count}...")
        
        # Get meta insights
        meta_insights = self.improvement_store.synthesize_meta_insights()
        
        # Adjust parameters based on insights
        if meta_insights["pattern_insights"]:
            # Find most successful patterns
            best_patterns = sorted(
                meta_insights["pattern_insights"],
                key=lambda x: x["success_rate"] * x["usage"],
                reverse=True
            )[:3]
            
            print(f"Top patterns: {[p['pattern'] for p in best_patterns]}")
            
        # Identify stagnation
        if len(self.generation_insights) > 3:
            recent_fitness = [g["fitness_stats"]["max"] for g in self.generation_insights[-3:]]
            if max(recent_fitness) - min(recent_fitness) < 0.05:
                print("⚠️ Fitness stagnation detected - increasing exploration")
                self.mutation_rate = min(0.9, self.mutation_rate * 1.5)
                self.exploration_bonus *= 1.2
                
        # Store synthesis results
        synthesis_result = {
            "generation": self.generation_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "meta_insights": meta_insights,
            "parameter_adjustments": {
                "mutation_rate": self.mutation_rate,
                "exploration_bonus": self.exploration_bonus
            }
        }
        
        # Save to research knowledge
        synthesis_result["topic"] = "evolutionary_synthesis"
        synthesis_result["metadata"] = {
            "quality_score": 0.8,
            "confidence": 0.9,
            "source": "evolutionary_engine"
        }
        
        self.research_store.store_research(
            research_type="meta_analysis",
            content=synthesis_result,
            quality_score=0.8
        )
        
    async def apply_best_improvement(self) -> Dict[str, Any]:
        """Apply best improvement with deployment tracking."""
        result = await super().apply_best_improvement()
        
        if result["success"]:
            improvement = result["improvement"]
            
            # Simulate deployment metrics (in real system, get actual metrics)
            metrics = {
                "deployment_time": datetime.now(timezone.utc).isoformat(),
                "initial_performance": improvement.get("fitness", 0),
                "estimated_impact": "high" if improvement.get("fitness", 0) > 0.7 else "medium"
            }
            
            # Record deployment
            self.improvement_store.record_deployment_result(
                improvement.get("id", "unknown"),
                metrics,
                success=True
            )
            
        return result
        
    def get_evolution_report(self) -> Dict[str, Any]:
        """Get comprehensive evolution report."""
        base_report = self.get_evolution_summary()
        
        # Enhance with knowledge insights
        knowledge_report = {
            "knowledge_stats": {
                "total_improvements_stored": len(self.improvement_store.improvement_vectors),
                "avoided_duplicates": self.avoided_duplicates,
                "knowledge_guided_mutations": self.knowledge_guided_mutations,
                "successful_patterns": len(self.improvement_store.get_successful_patterns()),
                "failed_patterns": len(self.improvement_store.get_failed_patterns()),
                "knowledge_gaps": len(self.improvement_store.find_knowledge_gaps())
            },
            "generation_insights": self.generation_insights[-5:],  # Last 5 generations
            "top_patterns": list(self.improvement_store.get_successful_patterns().keys())[:10],
            "recommendations": self._generate_recommendations()
        }
        
        return {**base_report, **knowledge_report}
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on accumulated knowledge."""
        recommendations = []
        
        # Check for stagnation
        if len(self.fitness_history) > 5:
            recent = self.fitness_history[-5:]
            if all(abs(recent[i]["best_fitness"] - recent[i-1]["best_fitness"]) < 0.01 
                   for i in range(1, len(recent))):
                recommendations.append("Consider increasing mutation rate or population size due to stagnation")
                
        # Check pattern diversity
        patterns = set()
        for imp in self.improvement_population:
            patterns.update(self.improvement_store._extract_patterns(imp))
        if len(patterns) < len(self.improvement_population):
            recommendations.append("Low pattern diversity - encourage more exploration")
            
        # Knowledge gaps
        gaps = self.improvement_store.find_knowledge_gaps()
        if gaps:
            recommendations.append(f"Explore {len(gaps)} identified knowledge gaps")
            
        return recommendations