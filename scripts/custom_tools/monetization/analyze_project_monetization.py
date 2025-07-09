"""
AI-Generated Tool: analyze_project_monetization
Description: Analyzes existing projects and generates comprehensive monetization strategies
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import json

from scripts.ai_brain import IntelligentAIBrain
from scripts.existing_project_monetization_analyzer import ExistingProjectMonetizationAnalyzer

__description__ = "Analyze existing projects for monetization potential and generate revenue strategies"
__parameters__ = {
    "project_name": {
        "type": "string",
        "required": False,
        "description": "Specific project to analyze (analyzes all if not specified)"
    },
    "generate_docs": {
        "type": "boolean", 
        "required": False,
        "default": True,
        "description": "Whether to generate monetization strategy documents"
    }
}
__examples__ = [
    "analyze_project_monetization()",
    "analyze_project_monetization(project_name='ai-code-assistant')",
    "analyze_project_monetization(project_name='smart-task-manager', generate_docs=False)"
]

async def analyze_project_monetization(**kwargs) -> Dict[str, Any]:
    """
    Analyze existing projects and generate monetization strategies.
    
    This tool examines projects in local_repos/repositories/CodeWebMobile-AI
    and creates comprehensive monetization strategies including pricing tiers,
    revenue projections, and market analysis.
    """
    try:
        # Initialize components
        ai_brain = IntelligentAIBrain()
        analyzer = ExistingProjectMonetizationAnalyzer(ai_brain)
        
        project_name = kwargs.get('project_name')
        generate_docs = kwargs.get('generate_docs', True)
        
        if project_name:
            # Analyze specific project
            project_path = analyzer.local_repos_path / project_name
            if not project_path.exists():
                return {
                    "status": "error",
                    "message": f"Project '{project_name}' not found in local repositories"
                }
            
            print(f"Analyzing monetization potential for {project_name}...")
            analysis = await analyzer.analyze_project(project_path)
            
            if analysis:
                result = {
                    "status": "success",
                    "project": project_name,
                    "analysis": analysis,
                    "monetization_exists": analysis.get('exists', False)
                }
                
                if not analysis.get('exists') and analysis.get('executive_summary'):
                    # Extract key metrics
                    summary = analysis['executive_summary']
                    result['key_metrics'] = {
                        "target_mrr": summary.get('target_mrr', 'N/A'),
                        "revenue_model": summary.get('revenue_model', 'N/A'),
                        "target_audience": summary.get('target_audience', 'N/A'),
                        "timeframe": summary.get('timeframe', '6 months')
                    }
                    
                    if generate_docs:
                        result['documents_generated'] = [
                            f"{project_name}/monetization_strategy.md",
                            f"{project_name}/market_analysis.md"
                        ]
                
                return result
            else:
                return {
                    "status": "error",
                    "message": f"Could not analyze project '{project_name}'"
                }
                
        else:
            # Analyze all projects
            print("Analyzing monetization potential for all projects...")
            all_results = await analyzer.analyze_all_projects()
            
            # Generate summary report
            report = await analyzer.generate_monetization_report()
            
            # Summarize results
            summary = {
                "total_projects": len(all_results),
                "with_monetization": sum(1 for r in all_results.values() if r.get('exists')),
                "new_strategies_generated": sum(1 for r in all_results.values() if not r.get('exists') and r.get('executive_summary')),
                "total_revenue_potential": 0
            }
            
            # Calculate total revenue potential
            for project, analysis in all_results.items():
                if analysis.get('executive_summary'):
                    target_mrr = analysis['executive_summary'].get('target_mrr', 0)
                    if isinstance(target_mrr, (int, float)):
                        summary['total_revenue_potential'] += target_mrr
            
            # Build detailed results
            projects_analyzed = []
            for project, analysis in all_results.items():
                project_info = {
                    "name": project,
                    "has_monetization": analysis.get('exists', False)
                }
                
                if analysis.get('executive_summary'):
                    exec_summary = analysis['executive_summary']
                    project_info.update({
                        "target_mrr": exec_summary.get('target_mrr'),
                        "revenue_model": exec_summary.get('revenue_model'),
                        "target_audience": exec_summary.get('target_audience')
                    })
                    
                projects_analyzed.append(project_info)
            
            return {
                "status": "success",
                "summary": summary,
                "projects": projects_analyzed,
                "report": report,
                "message": f"Analyzed {summary['total_projects']} projects. Generated {summary['new_strategies_generated']} new monetization strategies."
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing monetization: {str(e)}"
        }

# Test function for standalone execution
async def main():
    # Test analyzing all projects
    result = await analyze_project_monetization()
    print(json.dumps(result, indent=2))
    
    # Test analyzing specific project
    if result['status'] == 'success' and result.get('projects'):
        first_project = result['projects'][0]['name']
        specific_result = await analyze_project_monetization(project_name=first_project)
        print(f"\nSpecific analysis for {first_project}:")
        print(json.dumps(specific_result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())