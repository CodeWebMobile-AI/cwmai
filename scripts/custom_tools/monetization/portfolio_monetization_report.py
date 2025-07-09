"""
AI-Generated Tool: portfolio_monetization_report
Description: Generates comprehensive monetization report for entire portfolio
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

from scripts.ai_brain import IntelligentAIBrain
from scripts.existing_project_monetization_analyzer import ExistingProjectMonetizationAnalyzer
from scripts.integrated_project_evaluation_system import IntegratedProjectEvaluationSystem
from scripts.state_manager import StateManager

__description__ = "Generate comprehensive monetization report showing revenue potential across entire portfolio"
__parameters__ = {
    "output_format": {
        "type": "string",
        "required": False,
        "default": "detailed",
        "description": "Report format (summary, detailed, executive)"
    },
    "include_projections": {
        "type": "boolean",
        "required": False,
        "default": True,
        "description": "Include revenue projections and forecasts"
    },
    "save_report": {
        "type": "boolean",
        "required": False,
        "default": True,
        "description": "Save report to file"
    }
}
__examples__ = [
    "portfolio_monetization_report()",
    "portfolio_monetization_report(output_format='summary')",
    "portfolio_monetization_report(output_format='executive', include_projections=False)"
]

async def portfolio_monetization_report(**kwargs) -> Dict[str, Any]:
    """
    Generate comprehensive monetization report for the entire portfolio.
    
    Analyzes all projects, calculates total revenue potential, identifies
    opportunities, and provides strategic recommendations.
    """
    try:
        # Initialize components
        ai_brain = IntelligentAIBrain()
        monetization_analyzer = ExistingProjectMonetizationAnalyzer(ai_brain)
        evaluator = IntegratedProjectEvaluationSystem(ai_brain)
        state_manager = StateManager()
        
        # Get parameters
        output_format = kwargs.get('output_format', 'detailed')
        include_projections = kwargs.get('include_projections', True)
        save_report = kwargs.get('save_report', True)
        
        # Get portfolio
        try:
            state = state_manager.load_state()
            portfolio = state.get('repositories', {})
        except:
            portfolio = {}
        
        print("Generating portfolio monetization report...")
        
        # Analyze all projects
        project_analyses = await monetization_analyzer.analyze_all_projects()
        
        # Get new opportunities
        evaluation = await evaluator.evaluate_all_opportunities(
            existing_portfolio=portfolio,
            max_new_opportunities=3,
            include_existing=True
        )
        
        # Generate report based on format
        if output_format == 'summary':
            report_data = _generate_summary_report(project_analyses, evaluation)
        elif output_format == 'executive':
            report_data = _generate_executive_report(project_analyses, evaluation)
        else:  # detailed
            report_data = _generate_detailed_report(
                project_analyses, evaluation, include_projections
            )
        
        # Save report if requested
        output_path = None
        if save_report:
            report_content = _format_report_markdown(report_data, output_format)
            
            reports_dir = Path("portfolio_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_monetization_{output_format}_{timestamp}.md"
            output_path = reports_dir / filename
            
            output_path.write_text(report_content)
        
        # Prepare response
        result = {
            "status": "success",
            "report_type": output_format,
            "portfolio_summary": report_data['summary'],
            "key_insights": report_data['insights'],
            "recommendations": report_data['recommendations'],
            "total_revenue_potential": report_data['total_revenue_potential']
        }
        
        if output_path:
            result['report_saved'] = str(output_path)
        
        # Add format-specific data
        if output_format == 'detailed':
            result['project_details'] = report_data.get('projects', [])
            result['opportunity_pipeline'] = report_data.get('opportunities', [])
        
        if include_projections:
            result['revenue_forecast'] = report_data.get('forecast', {})
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating portfolio report: {str(e)}"
        }

def _generate_summary_report(
    project_analyses: Dict[str, Any],
    evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate summary format report."""
    
    # Calculate totals
    total_projects = len(project_analyses)
    monetized_projects = sum(1 for p in project_analyses.values() if p.get('exists'))
    new_strategies = sum(1 for p in project_analyses.values() 
                        if not p.get('exists') and p.get('executive_summary'))
    
    # Calculate revenue potential
    total_existing_revenue = 0
    total_potential_revenue = 0
    
    for project, analysis in project_analyses.items():
        if analysis.get('executive_summary'):
            target_mrr = analysis['executive_summary'].get('target_mrr', 0)
            if isinstance(target_mrr, (int, float)):
                total_potential_revenue += target_mrr
    
    # Add new opportunities
    for opp in evaluation.get('new_opportunities', []):
        if 'revenue_projections' in opp:
            # Get month 6 projection
            total_potential_revenue += opp['revenue_projections'].get(6, 0)
    
    return {
        'summary': {
            'total_projects': total_projects,
            'monetized_projects': monetized_projects,
            'unmonetized_projects': total_projects - monetized_projects,
            'new_strategies_generated': new_strategies,
            'new_opportunities_identified': len(evaluation.get('new_opportunities', []))
        },
        'total_revenue_potential': f"${total_potential_revenue:,.0f}/month",
        'insights': [
            f"{((total_projects - monetized_projects) / total_projects * 100):.0f}% of portfolio lacks monetization",
            f"Total addressable revenue: ${total_potential_revenue * 12:,.0f}/year",
            f"Average revenue per project: ${total_potential_revenue / max(total_projects, 1):,.0f}/month"
        ],
        'recommendations': [
            f"Prioritize monetizing {total_projects - monetized_projects} existing projects",
            "Focus on quick wins with existing user bases",
            "Implement consistent pricing strategy across portfolio"
        ]
    }

def _generate_executive_report(
    project_analyses: Dict[str, Any],
    evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate executive format report."""
    
    base_report = _generate_summary_report(project_analyses, evaluation)
    
    # Add strategic insights
    top_opportunities = []
    
    # Extract top existing projects
    for project, analysis in project_analyses.items():
        if analysis.get('executive_summary') and not analysis.get('exists'):
            summary = analysis['executive_summary']
            top_opportunities.append({
                'name': project,
                'type': 'existing_project',
                'revenue_potential': summary.get('target_mrr', 0),
                'time_to_revenue': '30 days',
                'investment': 'Low - existing codebase'
            })
    
    # Add top new opportunities
    for opp in evaluation.get('new_opportunities', [])[:3]:
        top_opportunities.append({
            'name': opp.get('name', 'New Opportunity'),
            'type': 'new_project',
            'revenue_potential': opp.get('revenue_projections', {}).get(6, 0),
            'time_to_revenue': f"{opp.get('time_to_revenue', 45)} days",
            'investment': f"{opp.get('implementation_complexity', 'Medium')} complexity"
        })
    
    # Sort by revenue potential
    top_opportunities.sort(key=lambda x: x['revenue_potential'], reverse=True)
    
    base_report['strategic_opportunities'] = top_opportunities[:5]
    base_report['portfolio_health'] = _assess_portfolio_health(project_analyses)
    
    # Executive recommendations
    base_report['recommendations'] = [
        f"Immediate action: Monetize top {min(3, len(top_opportunities))} opportunities",
        "Establish revenue targets: $1M ARR within 12 months",
        "Allocate resources: 70% existing projects, 30% new ventures",
        "Implement portfolio-wide analytics and billing infrastructure"
    ]
    
    return base_report

def _generate_detailed_report(
    project_analyses: Dict[str, Any],
    evaluation: Dict[str, Any],
    include_projections: bool
) -> Dict[str, Any]:
    """Generate detailed format report with all data."""
    
    # Start with executive report
    report = _generate_executive_report(project_analyses, evaluation)
    
    # Add detailed project information
    projects_detail = []
    
    for project, analysis in project_analyses.items():
        project_info = {
            'name': project,
            'has_monetization': analysis.get('exists', False),
            'status': 'Monetized' if analysis.get('exists') else 'Unmonetized'
        }
        
        if analysis.get('executive_summary'):
            summary = analysis['executive_summary']
            project_info.update({
                'revenue_model': summary.get('revenue_model', 'N/A'),
                'target_mrr': summary.get('target_mrr', 0),
                'target_audience': summary.get('target_audience', 'N/A'),
                'pricing_tiers': analysis.get('pricing_tiers', [])
            })
            
            # Add projections if available
            if include_projections and 'financial_projections' in analysis:
                proj = analysis['financial_projections']
                project_info['projections'] = {
                    'month_1': proj.get('months', [{}])[0].get('mrr', 0),
                    'month_6': _find_month_projection(proj.get('months', []), 6),
                    'month_12': _find_month_projection(proj.get('months', []), 12)
                }
        
        projects_detail.append(project_info)
    
    report['projects'] = projects_detail
    
    # Add opportunity pipeline
    opportunities = []
    for opp in evaluation.get('all_opportunities', []):
        opportunities.append({
            'title': opp['title'],
            'type': opp['type'],
            'score': opp['scores']['total'],
            'revenue_potential': opp.get('revenue_potential', 'N/A'),
            'complexity': opp.get('complexity', 'medium')
        })
    
    report['opportunities'] = sorted(opportunities, key=lambda x: x['score'], reverse=True)
    
    # Add revenue forecast
    if include_projections:
        report['forecast'] = _generate_revenue_forecast(
            project_analyses, evaluation
        )
    
    # Add portfolio gaps
    if 'portfolio_insights' in evaluation:
        report['portfolio_gaps'] = evaluation['portfolio_insights'].get('gaps', [])
        report['portfolio_strengths'] = evaluation['portfolio_insights'].get('strengths', [])
    
    return report

def _assess_portfolio_health(project_analyses: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall portfolio health."""
    
    total = len(project_analyses)
    monetized = sum(1 for p in project_analyses.values() if p.get('exists'))
    
    monetization_rate = (monetized / max(total, 1)) * 100
    
    if monetization_rate > 80:
        health_status = "Excellent"
        health_score = "A"
    elif monetization_rate > 60:
        health_status = "Good"
        health_score = "B"
    elif monetization_rate > 40:
        health_status = "Fair"
        health_score = "C"
    else:
        health_status = "Needs Improvement"
        health_score = "D"
    
    return {
        'status': health_status,
        'score': health_score,
        'monetization_rate': f"{monetization_rate:.0f}%",
        'improvement_areas': [
            'Monetization strategy implementation',
            'Revenue diversification',
            'Customer acquisition optimization'
        ]
    }

def _find_month_projection(months: List[Dict], target_month: int) -> float:
    """Find projection for specific month."""
    for month in months:
        if month.get('month') == target_month:
            return month.get('mrr', 0)
    return 0

def _generate_revenue_forecast(
    project_analyses: Dict[str, Any],
    evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate revenue forecast for next 12 months."""
    
    forecast = {}
    
    for month in range(1, 13):
        total_mrr = 0
        
        # Add existing project projections
        for project, analysis in project_analyses.items():
            if 'financial_projections' in analysis:
                month_mrr = _find_month_projection(
                    analysis['financial_projections'].get('months', []), 
                    month
                )
                total_mrr += month_mrr
        
        # Add new opportunity projections
        for opp in evaluation.get('new_opportunities', [])[:3]:
            if 'revenue_projections' in opp:
                total_mrr += opp['revenue_projections'].get(month, 0)
        
        forecast[f'month_{month}'] = {
            'mrr': total_mrr,
            'arr': total_mrr * 12
        }
    
    # Add summary metrics
    forecast['summary'] = {
        'month_1_mrr': forecast['month_1']['mrr'],
        'month_6_mrr': forecast['month_6']['mrr'],
        'month_12_mrr': forecast['month_12']['mrr'],
        'year_1_total': sum(f['mrr'] for f in forecast.values() if isinstance(f, dict))
    }
    
    return forecast

def _format_report_markdown(report_data: Dict[str, Any], format_type: str) -> str:
    """Format report data as markdown."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# Portfolio Monetization Report

Generated: {timestamp}
Report Type: {format_type.title()}

## Executive Summary

Total Revenue Potential: **{report_data['total_revenue_potential']}**

### Portfolio Overview
"""
    
    summary = report_data['summary']
    for key, value in summary.items():
        content += f"- {key.replace('_', ' ').title()}: {value}\n"
    
    content += "\n## Key Insights\n\n"
    for insight in report_data['insights']:
        content += f"- {insight}\n"
    
    content += "\n## Strategic Recommendations\n\n"
    for i, rec in enumerate(report_data['recommendations'], 1):
        content += f"{i}. {rec}\n"
    
    # Add format-specific sections
    if format_type in ['executive', 'detailed'] and 'strategic_opportunities' in report_data:
        content += "\n## Top Revenue Opportunities\n\n"
        content += "| Opportunity | Type | Revenue Potential | Time to Revenue | Investment |\n"
        content += "|------------|------|------------------|-----------------|------------|\n"
        
        for opp in report_data['strategic_opportunities']:
            content += f"| {opp['name']} | {opp['type']} | "
            content += f"${opp['revenue_potential']:,}/mo | "
            content += f"{opp['time_to_revenue']} | {opp['investment']} |\n"
    
    if format_type == 'detailed' and 'projects' in report_data:
        content += "\n## Project Details\n\n"
        for project in report_data['projects'][:10]:  # Limit to top 10
            content += f"### {project['name']}\n"
            content += f"- Status: {project['status']}\n"
            if 'revenue_model' in project:
                content += f"- Revenue Model: {project['revenue_model']}\n"
                content += f"- Target MRR: ${project.get('target_mrr', 0):,}\n"
    
    if 'portfolio_health' in report_data:
        health = report_data['portfolio_health']
        content += f"\n## Portfolio Health\n\n"
        content += f"- Overall Status: **{health['status']}** ({health['score']})\n"
        content += f"- Monetization Rate: {health['monetization_rate']}\n"
    
    return content

# Test function
async def main():
    # Test different report formats
    print("Testing summary report...")
    summary = await portfolio_monetization_report(output_format='summary')
    print(json.dumps(summary, indent=2))
    
    print("\nTesting executive report...")
    executive = await portfolio_monetization_report(
        output_format='executive',
        include_projections=True
    )
    print(f"Report saved to: {executive.get('report_saved', 'Not saved')}")

if __name__ == "__main__":
    asyncio.run(main())