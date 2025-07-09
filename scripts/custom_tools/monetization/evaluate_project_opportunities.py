"""
AI-Generated Tool: evaluate_project_opportunities
Description: Comprehensive evaluation of all project opportunities (new and existing)
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from scripts.ai_brain import IntelligentAIBrain
from scripts.integrated_project_evaluation_system import IntegratedProjectEvaluationSystem
from scripts.state_manager import StateManager

__description__ = "Evaluate all project opportunities with unified scoring and actionable recommendations"
__parameters__ = {
    "include_existing": {
        "type": "boolean",
        "required": False,
        "default": True,
        "description": "Include analysis of existing projects for monetization"
    },
    "max_new_opportunities": {
        "type": "integer",
        "required": False,
        "default": 5,
        "description": "Maximum new opportunities to evaluate"
    },
    "output_report": {
        "type": "boolean",
        "required": False,
        "default": False,
        "description": "Generate detailed evaluation report file"
    }
}
__examples__ = [
    "evaluate_project_opportunities()",
    "evaluate_project_opportunities(include_existing=False, max_new_opportunities=3)",
    "evaluate_project_opportunities(output_report=True)"
]

async def evaluate_project_opportunities(**kwargs) -> Dict[str, Any]:
    """
    Comprehensive evaluation of all project opportunities using unified scoring.
    
    This tool combines analysis of new business opportunities and existing projects,
    providing data-driven recommendations on whether to create new projects or
    monetize existing ones.
    """
    try:
        # Initialize components
        ai_brain = IntelligentAIBrain()
        evaluator = IntegratedProjectEvaluationSystem(ai_brain)
        state_manager = StateManager()
        
        # Get parameters
        include_existing = kwargs.get('include_existing', True)
        max_new = kwargs.get('max_new_opportunities', 5)
        output_report = kwargs.get('output_report', False)
        
        # Get existing portfolio
        try:
            state = state_manager.load_state()
            existing_portfolio = state.get('repositories', {})
        except:
            existing_portfolio = {}
        
        print("Running comprehensive project evaluation...")
        
        # Run evaluation
        evaluation_report = await evaluator.evaluate_all_opportunities(
            existing_portfolio=existing_portfolio,
            max_new_opportunities=max_new,
            include_existing=include_existing
        )
        
        # Generate decision summary
        decision = await evaluator.create_decision_report(evaluation_report)
        
        # Categorize opportunities
        categorized = await evaluator.score_new_vs_existing(evaluation_report)
        
        # Generate recommendations
        recommendations = await evaluator.generate_recommendations(evaluation_report)
        
        # Format results for clarity
        formatted_result = {
            "status": "success",
            "evaluation_summary": {
                "total_opportunities": len(evaluation_report['all_opportunities']),
                "new_projects": len(evaluation_report.get('new_opportunities', [])),
                "existing_projects": len(evaluation_report.get('existing_projects', [])),
                "strategic_alignment": evaluation_report['charter'].get('primary_goal', 'Not specified')
            },
            "decision": {
                "recommendation": decision['decision'],
                "primary_opportunity": decision['primary_opportunity'],
                "expected_roi": decision['financial_summary']['expected_roi'],
                "investment_required": decision['financial_summary']['investment_required']
            },
            "top_opportunities": _extract_top_opportunities(
                evaluation_report['all_opportunities'], 5
            ),
            "by_priority": {
                "critical": _format_opportunities(categorized['critical']),
                "high": _format_opportunities(categorized['high']),
                "medium": _format_opportunities(categorized['medium'])
            },
            "recommendations": recommendations,
            "action_plan": evaluation_report.get('action_plan', {})
        }
        
        # Add portfolio insights if available
        if 'portfolio_insights' in evaluation_report:
            formatted_result['portfolio_insights'] = {
                "gaps": evaluation_report['portfolio_insights'].get('gaps', []),
                "strengths": evaluation_report['portfolio_insights'].get('strengths', []),
                "market_coverage": evaluation_report['portfolio_insights'].get('market_coverage', {})
            }
        
        # Generate report file if requested
        if output_report:
            report_path = Path("evaluation_report.md")
            report_content = await evaluator.generate_decision_report(
                evaluation_report,
                output_path=report_path
            )
            formatted_result['report_generated'] = str(report_path)
        
        # Add quick wins section
        quick_wins = _identify_quick_wins(evaluation_report['all_opportunities'])
        if quick_wins:
            formatted_result['quick_wins'] = quick_wins
        
        return formatted_result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error evaluating opportunities: {str(e)}"
        }

def _extract_top_opportunities(opportunities: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Extract and format top opportunities."""
    # Sort by total score
    sorted_opps = sorted(opportunities, key=lambda x: x['scores']['total'], reverse=True)
    
    top_opps = []
    for opp in sorted_opps[:limit]:
        formatted = {
            "title": opp['title'],
            "type": opp['type'],
            "score": round(opp['scores']['total'], 2),
            "time_to_revenue": opp.get('time_to_revenue', 'N/A'),
            "revenue_potential": opp.get('revenue_potential', 'N/A'),
            "complexity": opp.get('complexity', 'medium'),
            "key_benefit": _get_key_benefit(opp)
        }
        top_opps.append(formatted)
    
    return top_opps

def _format_opportunities(opportunities: List[Dict[str, Any]]) -> List[str]:
    """Format opportunities for display."""
    formatted = []
    for opp in opportunities[:3]:  # Limit to top 3 per category
        opp_str = f"{opp['title']} ({opp['type']}) - Score: {round(opp['scores']['total'], 2)}"
        if opp.get('revenue_potential'):
            opp_str += f" - Revenue: {opp['revenue_potential']}"
        formatted.append(opp_str)
    return formatted

def _identify_quick_wins(opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify quick win opportunities."""
    quick_wins = []
    
    for opp in opportunities:
        # Quick win criteria:
        # 1. Existing project needing monetization (faster than new)
        # 2. Low complexity
        # 3. Good score
        # 4. Fast time to revenue
        
        is_quick_win = False
        win_reason = ""
        
        if opp['type'] == 'existing_monetization':
            if opp['scores']['total'] > 0.6:
                is_quick_win = True
                win_reason = "Existing project ready for monetization"
        elif opp.get('complexity') == 'low' and opp['scores']['total'] > 0.7:
            time_to_revenue = opp.get('time_to_revenue_days', 999)
            if isinstance(time_to_revenue, int) and time_to_revenue < 30:
                is_quick_win = True
                win_reason = f"Low complexity, revenue in {time_to_revenue} days"
        
        if is_quick_win:
            quick_wins.append({
                "opportunity": opp['title'],
                "reason": win_reason,
                "estimated_effort": opp.get('complexity', 'low'),
                "revenue_timeline": opp.get('time_to_revenue', 'Fast')
            })
    
    return quick_wins[:3]  # Top 3 quick wins

def _get_key_benefit(opportunity: Dict[str, Any]) -> str:
    """Extract the key benefit of an opportunity."""
    if opportunity['type'] == 'existing_monetization':
        return "Leverage existing codebase for immediate revenue"
    elif opportunity['scores']['market_validation'] > 0.8:
        return "Strong market validation and demand"
    elif opportunity['scores']['monetization_clarity'] > 0.8:
        return "Clear path to revenue with proven model"
    elif opportunity['scores']['time_to_revenue'] > 0.8:
        return "Fast time to revenue generation"
    else:
        return "Balanced opportunity with good potential"

# Test function for standalone execution
async def main():
    # Test comprehensive evaluation
    result = await evaluate_project_opportunities()
    print(json.dumps(result, indent=2))
    
    # Test with report generation
    if result['status'] == 'success':
        report_result = await evaluate_project_opportunities(
            include_existing=True,
            max_new_opportunities=3,
            output_report=True
        )
        print("\nEvaluation with report:")
        print(f"Report saved to: {report_result.get('report_generated', 'Not generated')}")

if __name__ == "__main__":
    asyncio.run(main())