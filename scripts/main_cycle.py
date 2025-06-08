"""
Main Cycle Module

Orchestrates the complete execution cycle of the autonomous AI system.
Coordinates state management, context gathering, and intelligent decision-making.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state_manager import StateManager
from context_gatherer import ContextGatherer
from ai_brain import IntelligentAIBrain


def load_context() -> Dict[str, Any]:
    """Load external context from context.json file.
    
    Returns:
        Context dictionary or empty dict if file doesn't exist
    """
    context_path = "context.json"
    
    try:
        if os.path.exists(context_path):
            with open(context_path, 'r') as f:
                context = json.load(f)
                print(f"Loaded context with {len(context.get('market_trends', []))} trends, "
                      f"{len(context.get('security_alerts', []))} security alerts")
                return context
        else:
            print("No context.json file found, using empty context")
            return {}
    except Exception as e:
        print(f"Error loading context: {e}")
        return {}


def validate_environment() -> bool:
    """Validate that required environment variables are available.
    
    Returns:
        True if environment is properly configured
    """
    required_vars = ['CLAUDE_PAT']
    optional_vars = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', 'DEEPSEEK_API_KEY']
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        print(f"ERROR: Missing required environment variables: {missing_required}")
        return False
    
    if missing_optional:
        print(f"WARNING: Missing optional environment variables: {missing_optional}")
        print("Some AI features may not be available")
    
    return True




def handle_forced_action() -> str:
    """Check for forced action from workflow input.
    
    Returns:
        Forced action type or 'auto' for normal operation
    """
    forced_action = os.getenv('FORCE_ACTION', 'auto')
    
    if forced_action != 'auto':
        print(f"Forced action specified: {forced_action}")
        
        # Validate forced action
        valid_actions = [
            'GENERATE_TASKS',
            'REVIEW_TASKS', 
            'PRIORITIZE_TASKS',
            'ANALYZE_PERFORMANCE',
            'UPDATE_DASHBOARD'
        ]
        
        if forced_action in valid_actions:
            return forced_action
        else:
            print(f"Invalid forced action '{forced_action}', falling back to auto")
    
    return 'auto'


def update_system_metrics(state: Dict[str, Any], report_data: Dict[str, Any]) -> None:
    """Update system-level performance metrics.
    
    Args:
        state: System state dictionary
        report_data: Report data from the cycle
    """
    performance = state.get("system_performance", {})
    
    # Update learning metrics based on recent performance
    total_cycles = performance.get("total_cycles", 0)
    successful_actions = performance.get("successful_actions", 0)
    failed_actions = performance.get("failed_actions", 0)
    
    if total_cycles > 0:
        # Calculate decision accuracy (success rate)
        total_actions = successful_actions + failed_actions
        decision_accuracy = successful_actions / total_actions if total_actions > 0 else 0.0
        
        # Calculate resource efficiency (successful actions per cycle as proxy)
        resource_efficiency = successful_actions / max(total_cycles, 1.0)
        
        # Calculate goal achievement (simplified metric based on project health)
        avg_health = sum(p.get("health_score", 50) for p in state.get("projects", {}).values())
        avg_health = avg_health / len(state.get("projects", {})) if state.get("projects") else 50
        goal_achievement = min(avg_health / 100.0, 1.0)
        
        # Update learning metrics
        learning_metrics = performance.get("learning_metrics", {})
        learning_metrics["decision_accuracy"] = decision_accuracy
        learning_metrics["resource_efficiency"] = resource_efficiency
        learning_metrics["goal_achievement"] = goal_achievement
        
        performance["learning_metrics"] = learning_metrics
        
        print(f"Updated metrics - Accuracy: {decision_accuracy:.3f}, "
              f"Efficiency: {resource_efficiency:.3f}, Achievement: {goal_achievement:.3f}")


def create_cycle_summary(state: Dict[str, Any], report_data: Dict[str, Any]) -> None:
    """Create and display cycle summary.
    
    Args:
        state: System state dictionary
        report_data: Report data from the cycle
    """
    print("\n" + "="*60)
    print("CYCLE SUMMARY")
    print("="*60)
    
    print(f"Cycle Number: {report_data.get('cycle_number', 'Unknown')}")
    print(f"Action Taken: {report_data.get('action_taken', 'Unknown')}")
    print(f"Outcome: {report_data.get('outcome', 'Unknown')}")
    print(f"Duration: {report_data.get('duration_seconds', 0):.1f} seconds")
    
    
    # Portfolio status
    projects = state.get("projects", {})
    if projects:
        avg_health = sum(p.get("health_score", 50) for p in projects.values()) / len(projects)
        print(f"Portfolio: {len(projects)} projects, avg health {avg_health:.1f}")
    else:
        print("Portfolio: No projects")
    
    # Performance metrics
    performance = state.get("system_performance", {})
    total_cycles = performance.get("total_cycles", 0)
    successful = performance.get("successful_actions", 0)
    failed = performance.get("failed_actions", 0)
    success_rate = successful / (successful + failed) if (successful + failed) > 0 else 0
    
    print(f"Performance: {successful}/{successful + failed} success rate ({success_rate:.1%}) over {total_cycles} cycles")
    
    # Learning metrics
    learning = performance.get("learning_metrics", {})
    if learning:
        print(f"Learning: Accuracy {learning.get('decision_accuracy', 0):.3f}, "
              f"Efficiency {learning.get('resource_efficiency', 0):.3f}, "
              f"Achievement {learning.get('goal_achievement', 0):.3f}")
    
    print("="*60)


def main():
    """Main execution function."""
    print("Starting Autonomous AI Software Development System Cycle")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed - exiting")
        sys.exit(1)
    
    # Initialize state manager
    print("\nInitializing state manager...")
    state_manager = StateManager()
    
    # Load system state
    print("Loading system state...")
    state = state_manager.load_state()
    print(f"Loaded state with {len(state.get('projects', {}))} projects")
    
    
    # Initialize AI Brain first
    print("Initializing AI Brain...")
    # Create temporary context for initial setup
    temp_context = load_context()
    ai_brain = IntelligentAIBrain(state, temp_context)
    
    # Log AI provider status
    ai_status = ai_brain.get_research_ai_status()
    print(f"AI Providers: Anthropic(primary)={ai_status['anthropic_primary']}, "
          f"OpenAI(secondary)={ai_status['openai_secondary']}, "
          f"Gemini={ai_status['gemini_available']}, "
          f"DeepSeek={ai_status['deepseek_available']}")
    
    # Load external context with AI enhancement
    print("Loading and enhancing external context...")
    context_gatherer = ContextGatherer(ai_brain=ai_brain)
    enhanced_context = context_gatherer.gather_context(state.get("charter", {}))
    
    # Update AI brain with enhanced context
    ai_brain.context = enhanced_context
    
    # Handle forced actions
    forced_action = handle_forced_action()
    
    # Override decision if action is forced
    if forced_action != 'auto':
        print(f"Overriding AI decision with forced action: {forced_action}")
        # Manually execute the forced action
        prompt = ai_brain.generate_dynamic_prompt(forced_action)
        success = ai_brain._execute_action_placeholder(forced_action, prompt)
        
        # Record the action
        outcome = "success_forced" if success else "failure_forced"
        ai_brain._record_action_outcome(forced_action, outcome, prompt)
        
        # Update performance
        state["system_performance"]["total_cycles"] += 1
        if success:
            state["system_performance"]["successful_actions"] += 1
        else:
            state["system_performance"]["failed_actions"] += 1
        
        # Create report
        report_data = {
            "cycle_number": state["system_performance"]["total_cycles"],
            "action_taken": forced_action,
            "outcome": outcome,
            "duration_seconds": 1.0,
            "portfolio_health": ai_brain._get_average_project_health(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": "task_management"
        }
        
        updated_state = ai_brain.state
    else:
        # Run intelligent cycle
        print("Running intelligent cycle...")
        updated_state, report_data = ai_brain.run_intelligent_cycle()
    
    # Update system metrics
    update_system_metrics(updated_state, report_data)
    
    # Save updated state
    print("Saving updated state...")
    state_manager.save_state_locally(updated_state)
    
    # Create cycle summary
    create_cycle_summary(updated_state, report_data)
    
    # Save report data for create_report.py
    with open("cycle_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print("\nCycle completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCycle interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Cycle failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)