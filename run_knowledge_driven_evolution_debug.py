#!/usr/bin/env python3
"""
Knowledge-Driven Evolutionary System

Runs the evolutionary improvement system with full knowledge integration,
learning from past improvements and research to generate breakthroughs.
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.state_manager import StateManager
from scripts.knowledge_driven_evolutionary_engine import KnowledgeDrivenEvolutionaryEngine
from scripts.evolutionary_knowledge_synthesis import EvolutionaryKnowledgeSynthesis
from scripts.evolutionary_knowledge_store import EvolutionaryKnowledgeStore
from scripts.evolutionary_config import EvolutionaryConfig
from scripts.adaptive_parameters import AdaptiveParameterEngine
from scripts.improvement_discovery import ImprovementDiscoveryEngine
from scripts.evolutionary_memory import EvolutionaryMemory
from scripts.evolutionary_plugins import EvolutionaryPluginManager
from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.ai_brain_factory import AIBrainFactory
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.safe_self_improver import SafeSelfImprover
from scripts.outcome_learning import OutcomeLearningSystem
from auto_recommendation_handler import AutoRecommendationHandler


class KnowledgeDrivenEvolutionRunner:
    """Orchestrates knowledge-driven evolutionary improvement generation."""
    
    def __init__(self, runtime_minutes: Optional[int] = None, enable_research: bool = True, enable_auto_tuning: bool = True):
        self.runtime_minutes = runtime_minutes
        self.enable_research = enable_research
        self.enable_auto_tuning = enable_auto_tuning
        self.state_manager = StateManager()
        
        # Core components
        self.evolutionary_engine = KnowledgeDrivenEvolutionaryEngine(self.state_manager)
        self.synthesis_engine = EvolutionaryKnowledgeSynthesis()
        self.knowledge_store = EvolutionaryKnowledgeStore()
        
        # Smart components
        self.config = EvolutionaryConfig()
        self.memory = EvolutionaryMemory()
        self.plugin_manager = EvolutionaryPluginManager()
        self.adaptive_engine = AdaptiveParameterEngine(self.config)
        self.discovery_engine = ImprovementDiscoveryEngine(self.config)
        
        # Research components (initialized if enabled)
        self.research_engine = None
        self.ai_brain = None
        self.task_generator = None
        self.self_improver = None
        self.outcome_learning = None
        
        if enable_research:
            self._initialize_research_components()
        
        # Auto-tuning
        self.auto_tuning_state = {
            'mutation_rate': 0.1,
            'population_size': 10,
            'synthesis_frequency': 10,
            'applied_adjustments': [],
            'last_fitness_improvement': 0,
            'stagnation_counter': 0,
            'generation_fitness_history': [],
            'auto_tuning_enabled': enable_auto_tuning
        }
        
        # Initialize auto-tuning handler if enabled
        self.auto_handler = AutoRecommendationHandler(self) if enable_auto_tuning else None
        
        # Tracking
        self.improvements_created = []
        self.breakthroughs_found = []
        self.research_insights = []
        self.generations_completed = 0
        self.research_cycles_completed = 0
        self.synthesis_counter = 0
        self.start_time = None
        
    async def run(self):
        """Run the knowledge-driven evolutionary system."""
        self.start_time = datetime.now()
        
        # Calculate end time or set to None for unlimited
        if self.runtime_minutes is not None:
            end_time = self.start_time.timestamp() + (self.runtime_minutes * 60)
            runtime_display = f"{self.runtime_minutes} MINUTE RUN"
        else:
            end_time = None
            runtime_display = "UNLIMITED RUN"
        
        print(f"\n{'='*80}")
        print(f"🧠 KNOWLEDGE-DRIVEN EVOLUTIONARY SYSTEM - {runtime_display}")
        print(f"{'='*80}")
        print(f"Start time: {self.start_time}")
        print(f"\n📊 KNOWLEDGE STATUS:")
        
        # Show current knowledge state
        meta_insights = self.knowledge_store.synthesize_meta_insights()
        print(f"  • Stored improvements: {meta_insights['total_improvements']}")
        print(f"  • Successful patterns: {meta_insights['success_patterns']}")
        print(f"  • Known failures: {meta_insights['failure_patterns']}")
        print(f"  • Knowledge gaps: {meta_insights['knowledge_gaps']}")
        
        # Show synthesis state
        synthesis_analysis = self.synthesis_engine.analyze_knowledge_flows()
        print(f"\n📈 SYNTHESIS STATUS:")
        print(f"  • Knowledge nodes: {synthesis_analysis['total_nodes']}")
        print(f"  • Cross-connections: {synthesis_analysis['cross_connections']}")
        print(f"  • Insights generated: {len(self.synthesis_engine.cross_domain_insights)}")
        
        # Show research state if enabled
        if self.enable_research and self.research_engine:
            print(f"\n🔬 RESEARCH STATUS:")
            research_state = self.research_engine.get_research_state()
            print(f"  • Research cycles completed: {research_state.get('cycles_completed', 0)}")
            print(f"  • Research topics explored: {len(research_state.get('topics_explored', []))}")
            print(f"  • Research insights available: {len(research_state.get('insights', []))}")
        
        print(f"\n{'='*80}\n")
        
        # Initialize population with knowledge
        await self.evolutionary_engine.initialize_population()
        
        generation = 0
        research_counter = 0
        
        last_generation_time = datetime.now()
        while end_time is None or datetime.now().timestamp() < end_time:
            # Check if generation is taking too long
            current_time = datetime.now()
            if (current_time - last_generation_time).seconds > 300:  # 5 minutes
                print(f"WARNING: Generation {generation} has been running for over 5 minutes")
                print(f"DEBUG: Last checkpoint was at {last_generation_time}")
            last_generation_time = current_time
            generation += 1
            
            # Calculate time remaining or show unlimited
            if end_time is not None:
                time_remaining = int(end_time - datetime.now().timestamp())
                time_display = f"{time_remaining}s remaining"
            else:
                elapsed = int(datetime.now().timestamp() - self.start_time.timestamp())
                time_display = f"{elapsed}s elapsed (unlimited)"
            
            print(f"\n🔄 Generation {generation} - {time_display}")
            print(f"DEBUG: Main loop - Generation {generation} starting")
            print(f"DEBUG: Population size: {len(self.evolutionary_engine.improvement_population) if hasattr(self.evolutionary_engine, \'improvement_population\') else \'Unknown\'}")
            
            # Run research cycle periodically
            if self.enable_research and self.research_engine:
                research_counter += 1
                if research_counter >= 3:  # Every 3 generations
                    print("\n🔬 Running research cycle...")
                    research_results = await self._run_research_cycle()
                    if research_results.get('insights'):
                        self.research_insights.extend(research_results['insights'])
                        print(f"  📚 Generated {len(research_results['insights'])} research insights")
                        
                        # Feed research insights into evolution
                        print(f"DEBUG: Integrating {len(research_results[\'insights\'])} research insights")
                        await self._integrate_research_insights(research_results['insights'])
                        print("DEBUG: Research insights integrated")
                    research_counter = 0
            
            # Evolve generation
            print(f"DEBUG: Starting evolve_generation for generation {generation}")
            await self.evolutionary_engine.evolve_generation()
            print(f"DEBUG: Completed evolve_generation for generation {generation}")
            
            # Apply best improvement
            if generation % 3 == 0:  # Every 3rd generation
                result = await self.evolutionary_engine.apply_best_improvement()
                if result['success']:
                    self.improvements_created.append(result)
                    print(f"✅ Applied: {result['improvement']['name']}")
                    
                    # Save improvement code to file
                    await self._save_improvement_to_file(result)
                    
            # Generate breakthroughs periodically
            if generation % 5 == 0:  # Every 5th generation
                print("\n🚀 Generating breakthrough improvements...")
                breakthroughs = await self.synthesis_engine.generate_breakthrough_improvements(3)
                
                for breakthrough in breakthroughs:
                    # Add to population for evolution
                    self.evolutionary_engine.improvement_population.append(breakthrough)
                    self.breakthroughs_found.append(breakthrough)
                    print(f"  💡 Breakthrough: {breakthrough['name']}")
                    
            # Run synthesis periodically (using dynamic frequency)
            self.synthesis_counter += 1
            synthesis_freq = self.auto_tuning_state.get('synthesis_frequency', 10)
            if self.synthesis_counter >= synthesis_freq:
                print(f"\n🧪 Running knowledge synthesis (frequency: every {synthesis_freq} generations)...")
                insights = await self.synthesis_engine.synthesize_cross_domain_insights()
                print(f"  Generated {len(insights)} new cross-domain insights")
                self.synthesis_counter = 0
                
            # Check early termination (only for time-limited runs)
            if end_time is not None:
                time_remaining = int(end_time - datetime.now().timestamp())
                if time_remaining < 30:
                    print("\n⏰ Time limit approaching, finalizing...")
                    break
                
            self.generations_completed = generation
            
            # Save state after each generation
            await self._save_all_state()
            
            # Apply auto-tuning if enabled
            if self.enable_auto_tuning and generation % 5 == 0:
                await self._apply_auto_tuning()
            
        # Final summary
        await self._final_summary()
        
        # Final state save
        await self._save_all_state()
        
    async def _final_summary(self):
        """Generate final summary of the run."""
        runtime = datetime.now() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"📊 FINAL KNOWLEDGE-DRIVEN EVOLUTION SUMMARY")
        print(f"{'='*80}")
        
        # Basic stats
        print(f"\n⏱️ Runtime: {runtime}")
        print(f"🔄 Generations completed: {self.generations_completed}")
        print(f"✅ Improvements created: {len(self.improvements_created)}")
        print(f"💡 Breakthroughs found: {len(self.breakthroughs_found)}")
        if self.enable_research:
            print(f"🔬 Research insights generated: {len(self.research_insights)}")
            print(f"🔄 Research cycles completed: {self.research_cycles_completed}")
        
        # Knowledge stats
        report = self.evolutionary_engine.get_evolution_report()
        knowledge_stats = report.get('knowledge_stats', {})
        
        print(f"\n🧠 KNOWLEDGE STATISTICS:")
        print(f"  • Total improvements stored: {knowledge_stats.get('total_improvements_stored', 0)}")
        print(f"  • Duplicates avoided: {knowledge_stats.get('avoided_duplicates', 0)}")
        print(f"  • Knowledge-guided mutations: {knowledge_stats.get('knowledge_guided_mutations', 0)}")
        print(f"  • Successful patterns identified: {knowledge_stats.get('successful_patterns', 0)}")
        print(f"  • Failed patterns identified: {knowledge_stats.get('failed_patterns', 0)}")
        print(f"  • Knowledge gaps remaining: {knowledge_stats.get('knowledge_gaps', 0)}")
        
        # Top patterns
        if 'top_patterns' in report:
            print(f"\n🏆 TOP SUCCESSFUL PATTERNS:")
            for i, pattern in enumerate(report['top_patterns'][:5], 1):
                print(f"  {i}. {pattern}")
                
        # Synthesis insights
        synthesis_analysis = self.synthesis_engine.analyze_knowledge_flows()
        print(f"\n🔗 KNOWLEDGE SYNTHESIS:")
        print(f"  • Cross-domain connections: {synthesis_analysis['cross_connections']}")
        print(f"  • Knowledge bridges found: {len(synthesis_analysis['knowledge_bridges'])}")
        print(f"  • Total insights: {len(self.synthesis_engine.cross_domain_insights)}")
        
        # Recommendations
        evolution_recommendations = report.get('recommendations', [])
        synthesis_recommendations = self.synthesis_engine.get_synthesis_recommendations()
        
        if evolution_recommendations or synthesis_recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in evolution_recommendations:
                print(f"  • [Evolution] {rec}")
            for rec in synthesis_recommendations:
                print(f"  • [Synthesis] {rec}")
                
        # Auto-tuning summary
        if self.enable_auto_tuning:
            auto_report = self.get_auto_tuning_report()
            print(f"\n🔧 AUTO-TUNING SUMMARY:")
            print(f"  • Enabled: {auto_report['enabled']}")
            print(f"  • Current mutation rate: {auto_report['current_parameters']['mutation_rate']:.2f}")
            print(f"  • Current population size: {auto_report['current_parameters']['population_size']}")
            print(f"  • Adjustments applied: {len(auto_report['applied_adjustments'])}")
            if auto_report['applied_adjustments']:
                print(f"\n  Applied adjustments:")
                for adj in auto_report['applied_adjustments']:
                    print(f"    - {adj}")
        
        # Save detailed report
        self._save_detailed_report(report, synthesis_analysis)
        
    def _save_detailed_report(self, evolution_report: Dict, synthesis_analysis: Dict):
        """Save detailed report to file."""
        report = {
            "run_info": {
                "start_time": self.start_time.isoformat(),
                "runtime_minutes": self.runtime_minutes if self.runtime_minutes else "unlimited",
                "generations": self.generations_completed
            },
            "results": {
                "improvements_created": len(self.improvements_created),
                "breakthroughs_found": len(self.breakthroughs_found),
                "improvement_details": [
                    {
                        "name": imp['improvement']['name'],
                        "type": imp['improvement'].get('type', 'unknown'),
                        "fitness": imp['improvement'].get('fitness', 0),
                        "file_path": imp.get('file_path', '')
                    }
                    for imp in self.improvements_created
                ],
                "breakthrough_details": [
                    {
                        "name": b['name'],
                        "breakthrough_score": b.get('breakthrough_score', 0),
                        "breakthrough_nature": b.get('breakthrough_nature', '')
                    }
                    for b in self.breakthroughs_found
                ]
            },
            "evolution_report": evolution_report,
            "synthesis_analysis": synthesis_analysis,
            "research_summary": {
                "enabled": self.enable_research,
                "cycles_completed": self.research_cycles_completed,
                "insights_generated": len(self.research_insights),
                "research_state": self.research_engine.get_research_state() if self.research_engine else {}
            } if self.enable_research else None,
            "auto_tuning_report": self.get_auto_tuning_report() if self.enable_auto_tuning else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_file = f"knowledge_evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    async def _save_improvement_to_file(self, result: Dict[str, Any]):
        """Save improvement code to a file."""
        try:
            improvement = result.get('improvement', {})
            if not improvement.get('code'):
                return
                
            # Create improvements directory
            improvement_dir = Path("scripts/improvements")
            improvement_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the improvement code
            file_name = f"{improvement['name']}.py"
            file_path = improvement_dir / file_name
            
            with open(file_path, 'w') as f:
                f.write(improvement['code'])
                
            print(f"💾 Saved improvement to: {file_path}")
            
            # Also save metadata
            metadata_path = improvement_dir / f"{improvement['name']}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'name': improvement.get('name'),
                    'type': improvement.get('type'),
                    'description': improvement.get('description'),
                    'fitness': improvement.get('fitness'),
                    'created_at': improvement.get('created_at'),
                    'generation': improvement.get('generation'),
                    'file_path': str(file_path)
                }, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving improvement: {e}")
    
    async def _save_all_state(self):
        """Save all critical state data."""
        try:
            # Save evolutionary knowledge store
            if hasattr(self.evolutionary_engine, 'knowledge_store'):
                self.evolutionary_engine.knowledge_store.save_state()
                print("💾 Saved evolutionary knowledge store")
            
            # Save research knowledge store
            if self.research_engine and hasattr(self.research_engine, 'knowledge_store'):
                await self._save_research_knowledge()
                
            # Save population snapshot
            population_dir = Path("population_snapshots")
            population_dir.mkdir(exist_ok=True)
            
            population_file = population_dir / f"population_gen_{self.generations_completed}.json"
            with open(population_file, 'w') as f:
                json.dump({
                    'generation': self.generations_completed,
                    'timestamp': datetime.now().isoformat(),
                    'population': self.evolutionary_engine.improvement_population,
                    'fitness_stats': {
                        'avg': sum(imp.get('fitness', 0) for imp in self.evolutionary_engine.improvement_population) / len(self.evolutionary_engine.improvement_population) if self.evolutionary_engine.improvement_population else 0,
                        'best': max((imp.get('fitness', 0) for imp in self.evolutionary_engine.improvement_population), default=0)
                    }
                }, f, indent=2)
            
            # Save synthesis state
            if hasattr(self.synthesis_engine, 'save_state'):
                self.synthesis_engine.save_state()
            
            # Save memory state
            if hasattr(self.memory, 'save_to_disk'):
                self.memory.save_to_disk("evolutionary_memory.json")
                
        except Exception as e:
            print(f"❌ Error saving state: {e}")
    
    async def _save_research_knowledge(self):
        """Save research knowledge store."""
        try:
            research_dir = Path("research_knowledge")
            research_dir.mkdir(exist_ok=True)
            
            # Save research results
            with open(research_dir / "research_store.json", 'w') as f:
                json.dump({
                    'cycles_completed': self.research_cycles_completed,
                    'insights': self.research_insights,
                    'timestamp': datetime.now().isoformat(),
                    'research_state': self.research_engine.get_research_state()
                }, f, indent=2)
                
            print("💾 Saved research knowledge store")
            
        except Exception as e:
            print(f"❌ Error saving research knowledge: {e}")
    
    def _initialize_research_components(self):
        """Initialize research-related components."""
        print("🔬 Initializing research components...")
        
        # Create AI brain for research
        self.ai_brain = AIBrainFactory.create_for_research()
        
        # Initialize task generator
        from scripts.dynamic_charter import DynamicCharter
        charter_system = DynamicCharter(self.ai_brain)
        self.task_generator = IntelligentTaskGenerator(
            self.ai_brain,
            charter_system,
            None  # outcome_learning will be set later
        )
        
        # Initialize self improver
        self.self_improver = SafeSelfImprover(".", 24)  # Use current directory and default max changes
        
        # Initialize outcome learning
        self.outcome_learning = OutcomeLearningSystem(self.ai_brain)
        
        # Now set outcome learning in task generator
        self.task_generator.learning_system = self.outcome_learning
        
        # Initialize research engine with all components
        self.research_engine = ResearchEvolutionEngine(
            state_manager=self.state_manager,
            ai_brain=self.ai_brain,
            task_generator=self.task_generator,
            self_improver=self.self_improver,
            outcome_learning=self.outcome_learning
        )
        
        # Configure for aggressive research in evolution context
        self.research_engine.config.update({
            "cycle_interval_seconds": 60,  # 1 minute between cycles
            "max_research_per_cycle": 5,
            "min_insight_confidence": 0.5,
            "auto_implement_threshold": 0.7,
            "enable_dynamic_triggering": True,
            "enable_external_agent_research": True,
            "external_research_frequency": 3  # Every 3rd cycle
        })
        
        print("✅ Research components initialized")
    
    async def _run_research_cycle(self):
        """Run a single research cycle."""
        if not self.research_engine:
            return {}
        
        try:
            # Set context for the research engine before execution
            if hasattr(self.research_engine, 'last_trigger_context'):
                self.research_engine.last_trigger_context = {
                    'reason': "Knowledge-driven evolution cycle",
                    'generation': self.generations_completed,
                    'improvements_created': len(self.improvements_created),
                    'breakthroughs_found': len(self.breakthroughs_found),
                    'evolution_stage': 'active'
                }
            
            # Execute research cycle (no parameters)
            results = await self.research_engine.execute_research_cycle()
            print(f"DEBUG: Research cycle returned: {type(results)}, keys: {results.keys() if isinstance(results, dict) else \'Not a dict\'}")
            print(f"DEBUG: Research insights count: {len(results.get(\'insights\', []))}")
            
            self.research_cycles_completed += 1
            return results
            
        except Exception as e:
            print(f"❌ Research cycle error: {e}")
            import traceback
            print(f"DEBUG: Full research error traceback:")
            traceback.print_exc()
            return {}
    
    async def _integrate_research_insights(self, insights: List[Dict]):
        """Integrate research insights into the evolutionary process."""
        print(f"DEBUG: _integrate_research_insights called with {len(insights)} insights")
        """Integrate research insights into the evolutionary process."""
        for insight in insights:
            # Convert research insight to improvement candidate
            improvement = {
                'name': f"Research-driven: {insight.get('title', 'Improvement')}",
                'type': insight.get('improvement_type', 'enhancement'),
                'description': insight.get('description', ''),
                'code': insight.get('implementation', ''),
                'fitness': insight.get('confidence', 0.5),
                'source': 'research',
                'research_insight': insight
            }
            
            # Add to evolutionary population
            self.evolutionary_engine.improvement_population.append(improvement)
            
            # Store in knowledge store
            self.knowledge_store.store_improvement(improvement, {
                'source': 'research_integration',
                'insight_id': insight.get('id', ''),
                'integration_time': datetime.now().isoformat()
            })
    
    async def _apply_auto_tuning(self):
        """Apply auto-tuning based on evolution performance."""
        if not self.auto_handler:
            return
        
        print("\n🔧 Running auto-tuning analysis...")
        
        # Calculate fitness trends
        current_best_fitness = 0
        if self.evolutionary_engine.improvement_population:
            current_best_fitness = max(
                imp.get('fitness', 0) 
                for imp in self.evolutionary_engine.improvement_population
            )
        
        # Track fitness history
        self.auto_tuning_state['generation_fitness_history'].append({
            'generation': self.generations_completed,
            'best_fitness': current_best_fitness,
            'avg_fitness': sum(imp.get('fitness', 0) for imp in self.evolutionary_engine.improvement_population) / len(self.evolutionary_engine.improvement_population) if self.evolutionary_engine.improvement_population else 0
        })
        
        # Check for stagnation (no improvement in last 10 generations)
        if len(self.auto_tuning_state['generation_fitness_history']) >= 10:
            recent_best = max(
                h['best_fitness'] 
                for h in self.auto_tuning_state['generation_fitness_history'][-10:]
            )
            if recent_best <= self.auto_tuning_state['last_fitness_improvement']:
                self.auto_tuning_state['stagnation_counter'] += 1
                print(f"  ⚠️ Stagnation detected: {self.auto_tuning_state['stagnation_counter']} cycles without improvement")
            else:
                self.auto_tuning_state['stagnation_counter'] = 0
                self.auto_tuning_state['last_fitness_improvement'] = recent_best
        
        # Generate recommendations based on performance
        recommendations = []
        
        if self.auto_tuning_state['stagnation_counter'] >= 2:
            # Population stagnating - need more diversity
            recommendations.append("Consider increasing mutation rate to add diversity")
            recommendations.append("Consider increasing population size for broader exploration")
        
        if len(self.breakthroughs_found) < self.generations_completed / 10:
            # Not enough breakthroughs
            recommendations.append("Consider increasing synthesis frequency for more breakthrough attempts")
        
        # Apply recommendations
        if recommendations:
            print(f"\n💡 AUTO-TUNING RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  • {rec}")
            
            # Apply the recommendations
            self.auto_handler.apply_recommendations(recommendations)
            
            # Reset stagnation counter after intervention
            if self.auto_tuning_state['stagnation_counter'] >= 2:
                self.auto_tuning_state['stagnation_counter'] = 0
                print("  ✅ Auto-tuning applied, resetting stagnation counter")
        else:
            print("  ✅ Performance is good, no tuning needed")
    
    def get_auto_tuning_report(self) -> Dict[str, Any]:
        """Get a report of auto-tuning activities."""
        return {
            'enabled': self.enable_auto_tuning,
            'current_parameters': {
                'mutation_rate': getattr(self.evolutionary_engine, 'mutation_rate', self.auto_tuning_state['mutation_rate']),
                'population_size': getattr(self.evolutionary_engine, 'population_size', self.auto_tuning_state['population_size']),
                'synthesis_frequency': self.auto_tuning_state['synthesis_frequency']
            },
            'fitness_history': self.auto_tuning_state['generation_fitness_history'][-20:],  # Last 20 generations
            'stagnation_cycles': self.auto_tuning_state['stagnation_counter'],
            'applied_adjustments': self.auto_handler.applied_recommendations if self.auto_handler else [],
            'performance_metrics': {
                'improvements_per_generation': len(self.improvements_created) / max(1, self.generations_completed),
                'breakthroughs_per_generation': len(self.breakthroughs_found) / max(1, self.generations_completed),
                'research_insights_per_cycle': len(self.research_insights) / max(1, self.research_cycles_completed) if self.enable_research else 0
            }
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run knowledge-driven evolutionary improvements"
    )
    parser.add_argument(
        '--minutes', 
        type=int, 
        default=None,
        help='Runtime in minutes (default: unlimited)'
    )
    parser.add_argument(
        '--breakthrough-focus',
        action='store_true',
        help='Focus on generating breakthrough improvements'
    )
    parser.add_argument(
        '--learn-from-history',
        action='store_true',
        help='Initialize from historical knowledge'
    )
    parser.add_argument(
        '--no-research',
        action='store_true',
        help='Disable research integration'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create runner
    runner = KnowledgeDrivenEvolutionRunner(
        runtime_minutes=args.minutes,
        enable_research=not args.no_research,
        enable_auto_tuning=True  # Enable auto-tuning by default
    )
    
    # Configure based on args
    if args.breakthrough_focus:
        runner.evolutionary_engine.exploration_bonus = 0.3
        runner.evolutionary_engine.min_novelty_threshold = 0.5
        print("🚀 Breakthrough focus mode enabled")
        
    if args.learn_from_history:
        # This is already done by default in knowledge-driven engine
        print("📚 Learning from historical knowledge")
        
    # Run
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        await runner._final_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())