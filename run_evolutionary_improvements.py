#!/usr/bin/env python3
"""
Run Evolutionary Improvements System

This script runs the AlphaEvolve-inspired evolutionary improvement system for a specified duration,
creating and evolving improvements through multiple generations.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime, timezone, timedelta
import hashlib
from typing import Dict, List, Any, Optional
import json
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evolutionary_improvements.log')
    ]
)
logger = logging.getLogger(__name__)

from scripts.state_manager import StateManager
from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine
from scripts.multi_objective_optimizer import MultiObjectiveOptimizer
from scripts.improvement_ensemble import ImprovementEnsemble
from scripts.automated_code_verifier import AutomatedCodeVerifier


class EvolutionaryImprovementRunner:
    """Runs the evolutionary improvement system with comprehensive monitoring."""
    
    def __init__(self, runtime_minutes: Optional[int] = None):
        self.runtime_minutes = runtime_minutes  # None means run indefinitely
        self.logger = logger  # Use module-level logger
        self.state_manager = StateManager()
        self.evolutionary_engine = EvolutionaryImprovementEngine(self.state_manager)
        self.optimizer = MultiObjectiveOptimizer(self.state_manager)
        self.ensemble = ImprovementEnsemble(self.state_manager)
        self.verifier = AutomatedCodeVerifier(self.state_manager)
        
        # Tracking
        self.start_time = None
        self.errors_encountered = []
        self.improvements_created = []
        self.generations_completed = 0
        self.verification_results = []
        
    async def run(self):
        """Run the evolutionary improvement system for the specified duration."""
        self.start_time = datetime.now(timezone.utc)
        
        if self.runtime_minutes:
            end_time = self.start_time + timedelta(minutes=self.runtime_minutes)
            mode_text = f"{self.runtime_minutes} MINUTE TEST"
        else:
            end_time = None
            mode_text = "CONTINUOUS MODE (Ctrl+C to stop)"
        
        print("="*80)
        print(f"🧬 EVOLUTIONARY IMPROVEMENTS SYSTEM - {mode_text}")
        print("="*80)
        print(f"Start time: {self.start_time.isoformat()}")
        if end_time:
            print(f"End time: {end_time.isoformat()}")
        else:
            print("End time: Manual stop required")
        
        print("\n📚 EVOLUTIONARY ALGORITHM PARAMETERS:")
        print(f"Population size: {self.evolutionary_engine.population_size}")
        print("  → Number of improvement candidates maintained in each generation")
        print("  → Larger populations explore more diverse solutions")
        
        print(f"\nElite size: {self.evolutionary_engine.elite_size}")
        print("  → Number of best-performing improvements preserved across generations")
        print("  → Ensures the best solutions are never lost")
        
        print(f"\nMutation rate: {self.evolutionary_engine.mutation_rate}")
        print("  → Probability of randomly modifying improvements (30% chance)")
        print("  → Introduces variation and prevents getting stuck in local optima")
        print("="*80)
        
        try:
            # Initialize population
            print("\n📊 Phase 1: Initializing Population")
            await self._initialize_population()
            
            # Evolution loop
            generation = 0
            while True:
                # Check if we should stop
                if end_time and datetime.now(timezone.utc) >= end_time:
                    break
                    
                generation += 1
                
                if end_time:
                    print(f"\n🔄 Generation {generation} - {self._time_remaining(end_time)} remaining")
                else:
                    runtime = datetime.now(timezone.utc) - self.start_time
                    print(f"\n🔄 Generation {generation} - Runtime: {self._format_duration(runtime)}")
                
                try:
                    # Evolve population
                    await self._evolve_generation()
                    
                    # Every 3 generations, apply best improvement
                    if generation % 3 == 0:
                        await self._apply_best_improvement()
                        
                except Exception as e:
                    error_info = {
                        'generation': generation,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    self.errors_encountered.append(error_info)
                    logger.error(f"Generation {generation} error: {e}")
                    
                # Brief pause between generations
                await asyncio.sleep(2)
                
            # Final summary
            await self._final_summary()
            
        except Exception as e:
            logger.error(f"Fatal error in evolutionary system: {e}")
            logger.error(traceback.format_exc())
            self.errors_encountered.append({
                'phase': 'fatal',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
    async def _initialize_population(self):
        """Initialize the first generation of improvements."""
        try:
            # Use ensemble to generate diverse initial population
            context = {
                'system_needs': await self._analyze_system_needs(),
                'existing_tools': self._get_existing_tools(),
                'performance_metrics': self._get_performance_metrics()
            }
            
            print("🧪 Generating initial population using ensemble approach...")
            try:
                initial_improvements = await self.ensemble.generate_ensemble_improvements(context)
            except Exception as e:
                self.logger.warning(f"Ensemble generation failed: {e}, using direct generation")
                initial_improvements = []
            
            # If ensemble fails or returns too few, generate directly
            if len(initial_improvements) < self.evolutionary_engine.population_size:
                print(f"📊 Ensemble provided {len(initial_improvements)} improvements, generating more...")
                await self.evolutionary_engine.initialize_population()
                # Add ensemble improvements if any
                for imp in initial_improvements[:self.evolutionary_engine.population_size]:
                    # Check if improvement already exists by name
                    existing_names = {existing.get('name') for existing in self.evolutionary_engine.improvement_population}
                    if imp.get('name') not in existing_names:
                        self.evolutionary_engine.improvement_population.append(imp)
            else:
                # Add to evolutionary engine
                self.evolutionary_engine.improvement_population = initial_improvements[:self.evolutionary_engine.population_size]
            
            # Ensure we have the right population size
            while len(self.evolutionary_engine.improvement_population) < self.evolutionary_engine.population_size:
                self.evolutionary_engine.improvement_population.append({
                    'name': f'placeholder_improvement_{len(self.evolutionary_engine.improvement_population)}',
                    'type': 'optimization',
                    'description': 'Placeholder improvement',
                    'approach': 'To be implemented',
                    'metrics': ['placeholder_metric'],
                    'integration_points': ['system'],
                    'fitness': 0.0,
                    'generation': 0,
                    'id': hashlib.md5(f"placeholder_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
                    'code': '# Placeholder code'
                })
            
            print(f"✅ Initialized population with {len(self.evolutionary_engine.improvement_population)} improvements")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize population: {e}")
            # Fallback to standard initialization
            await self.evolutionary_engine.initialize_population()
            
    async def _evolve_generation(self):
        """Evolve the current generation."""
        try:
            # Evaluate all improvements
            print("📊 Evaluating population fitness...")
            for improvement in self.evolutionary_engine.improvement_population:
                if improvement.get('fitness', 0) == 0:
                    # Multi-objective evaluation
                    scores = await self.optimizer.evaluate_improvement(improvement)
                    improvement['fitness'] = scores.get('overall', 0.5)
                    improvement['objective_scores'] = scores
                    
            # Evolve to next generation
            await self.evolutionary_engine.evolve_generation()
            self.generations_completed += 1
            
            # Log top performers
            top_3 = sorted(
                self.evolutionary_engine.improvement_population,
                key=lambda x: x.get('fitness', 0),
                reverse=True
            )[:3]
            
            print("\n🏆 Top 3 Improvements:")
            for i, imp in enumerate(top_3, 1):
                print(f"{i}. {imp['name']} - Fitness: {imp.get('fitness', 0):.3f}")
                if 'objective_scores' in imp:
                    scores = imp['objective_scores']
                    print(f"   Performance: {scores.get('performance', 0):.2f}, "
                          f"Quality: {scores.get('quality', 0):.2f}, "
                          f"Maintainability: {scores.get('maintainability', 0):.2f}, "
                          f"Innovation: {scores.get('innovation', 0):.2f}")
                          
        except Exception as e:
            logger.error(f"Evolution error: {e}")
            raise
            
    async def _apply_best_improvement(self):
        """Apply and verify the best improvement."""
        try:
            if not self.evolutionary_engine.improvement_population:
                return
                
            # Get best improvement
            best = max(
                self.evolutionary_engine.improvement_population,
                key=lambda x: x.get('fitness', 0)
            )
            
            print(f"\n🚀 Applying best improvement: {best['name']}")
            
            # Verify before applying
            verification = await self.verifier.verify_improvement(best)
            self.verification_results.append(verification)
            
            if verification['is_deployable']:
                result = await self.evolutionary_engine.apply_best_improvement()
                if result['success']:
                    self.improvements_created.append({
                        'improvement': best,
                        'verification': verification,
                        'file_path': result['file_path'],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    print(f"✅ Successfully applied: {result['file_path']}")
                else:
                    print(f"❌ Failed to apply improvement: {result.get('reason', 'Unknown')}")
            else:
                print(f"❌ Verification failed: Not deployable (score: {verification.get('overall_score', 0):.2f})")
                print(f"   Issues: {', '.join(verification.get('issues', []))}")
                
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            self.errors_encountered.append({
                'phase': 'apply_improvement',
                'error': str(e),
                'improvement': best.get('name', 'Unknown')
            })
            
    async def _analyze_system_needs(self) -> List[Dict]:
        """Analyze what the system needs improved."""
        needs = []
        
        # Check for common issues
        state = self.state_manager.get_state()
        
        # Performance needs
        if state.get('performance_metrics', {}).get('average_response_time', 0) > 2.0:
            needs.append({
                'type': 'optimization',
                'priority': 'high',
                'description': 'Response time optimization needed'
            })
            
        # Tool health
        if len(state.get('failed_tools', [])) > 0:
            needs.append({
                'type': 'error_fix',
                'priority': 'critical',
                'description': 'Fix failing tools'
            })
            
        # Feature gaps
        needs.append({
            'type': 'feature',
            'priority': 'medium',
            'description': 'Add new capabilities'
        })
        
        return needs
        
    def _get_existing_tools(self) -> List[str]:
        """Get list of existing tools."""
        tools_dir = os.path.join(os.path.dirname(__file__), 'scripts', 'custom_tools')
        if os.path.exists(tools_dir):
            return [f for f in os.listdir(tools_dir) if f.endswith('.py')]
        return []
        
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        state = self.state_manager.get_state()
        return state.get('performance_metrics', {
            'average_response_time': 1.5,
            'success_rate': 0.95,
            'memory_usage': 0.3
        })
        
    def _time_remaining(self, end_time: datetime) -> str:
        """Get formatted time remaining."""
        remaining = end_time - datetime.now(timezone.utc)
        minutes = int(remaining.total_seconds() / 60)
        seconds = int(remaining.total_seconds() % 60)
        return f"{minutes}m {seconds}s"
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format a duration nicely."""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
        
    async def _final_summary(self):
        """Generate final summary of the run."""
        print("\n" + "="*80)
        print("📊 EVOLUTIONARY IMPROVEMENTS FINAL SUMMARY")
        print("="*80)
        
        runtime = datetime.now(timezone.utc) - self.start_time
        
        summary = {
            'runtime': str(runtime),
            'generations_completed': self.generations_completed,
            'improvements_created': len(self.improvements_created),
            'errors_encountered': len(self.errors_encountered),
            'final_population_size': len(self.evolutionary_engine.improvement_population),
            'fitness_history': self.evolutionary_engine.fitness_history,
            'verification_pass_rate': sum(1 for v in self.verification_results if v['is_deployable']) / max(1, len(self.verification_results))
        }
        
        print(f"Runtime: {summary['runtime']}")
        print(f"Generations completed: {summary['generations_completed']}")
        print(f"Improvements created: {summary['improvements_created']}")
        print(f"Errors encountered: {summary['errors_encountered']}")
        print(f"Verification pass rate: {summary['verification_pass_rate']:.1%}")
        
        # Best improvement details
        if self.evolutionary_engine.improvement_population:
            best = max(
                self.evolutionary_engine.improvement_population,
                key=lambda x: x.get('fitness', 0)
            )
            print(f"\n🏆 Best Improvement:")
            print(f"Name: {best['name']}")
            print(f"Type: {best['type']}")
            print(f"Fitness: {best.get('fitness', 0):.3f}")
            if 'objective_scores' in best:
                scores = best['objective_scores']
                print(f"Objective Scores:")
                for obj, score in scores.items():
                    if obj != 'overall':
                        print(f"  - {obj}: {score:.3f}")
                        
        # Error summary
        if self.errors_encountered:
            print(f"\n❌ Errors Encountered ({len(self.errors_encountered)}):")
            error_types = {}
            for error in self.errors_encountered:
                error_type = error.get('phase', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
            for error_type, count in error_types.items():
                print(f"  - {error_type}: {count}")
                
        # Save detailed results
        results_file = f"evolutionary_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'improvements_created': self.improvements_created,
                'errors': self.errors_encountered,
                'verification_results': self.verification_results,
                'final_population': [
                    {
                        'name': imp['name'],
                        'type': imp['type'],
                        'fitness': imp.get('fitness', 0),
                        'generation': imp.get('generation', 0)
                    }
                    for imp in self.evolutionary_engine.improvement_population
                ]
            }, f, indent=2, default=str)
            
        print(f"\n📄 Detailed results saved to: {results_file}")
        print("="*80)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AlphaEvolve-inspired evolutionary improvements")
    parser.add_argument(
        '--minutes', 
        type=int, 
        default=None,
        help='Runtime in minutes (omit for continuous mode)'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=10,
        help='Number of improvements per generation (default: 10)'
    )
    parser.add_argument(
        '--elite-size',
        type=int,
        default=3,
        help='Number of top improvements to preserve (default: 3)'
    )
    parser.add_argument(
        '--mutation-rate',
        type=float,
        default=0.3,
        help='Probability of mutation (0.0-1.0, default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Create runner with specified or default settings
    runner = EvolutionaryImprovementRunner(runtime_minutes=args.minutes)
    
    # Apply custom evolutionary parameters if provided
    if args.population_size != 10:
        runner.evolutionary_engine.population_size = args.population_size
    if args.elite_size != 3:
        runner.evolutionary_engine.elite_size = args.elite_size
    if args.mutation_rate != 0.3:
        runner.evolutionary_engine.mutation_rate = args.mutation_rate
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evolutionary system stopped by user")
        await runner._final_summary()


if __name__ == "__main__":
    asyncio.run(main())