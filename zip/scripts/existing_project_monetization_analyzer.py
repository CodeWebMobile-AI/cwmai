"""
Existing Project Monetization Analyzer

Analyzes existing projects in local repositories and generates comprehensive
monetization strategies for projects that lack them.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from scripts.ai_brain import IntelligentAIBrain
from scripts.business_market_research_engine import (
    BusinessMarketResearchEngine, 
    MonetizationModel,
    MarketValidation,
    BusinessOpportunity
)


class ExistingProjectMonetizationAnalyzer:
    """Analyzes existing projects and creates monetization strategies."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize the analyzer."""
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        self.market_research = BusinessMarketResearchEngine(ai_brain)
        self.local_repos_path = Path("local_repos/repositories/CodeWebMobile-AI")
        
    async def analyze_all_projects(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all projects and generate monetization strategies.
        
        Returns:
            Dictionary mapping project names to their monetization analysis
        """
        results = {}
        
        if not self.local_repos_path.exists():
            self.logger.error(f"Local repos path not found: {self.local_repos_path}")
            return results
        
        # Get all project directories
        projects = [d for d in self.local_repos_path.iterdir() if d.is_dir()]
        
        self.logger.info(f"Found {len(projects)} projects to analyze")
        
        for project_dir in projects:
            try:
                analysis = await self.analyze_project(project_dir)
                if analysis:
                    results[project_dir.name] = analysis
            except Exception as e:
                self.logger.error(f"Error analyzing {project_dir.name}: {e}")
        
        return results
    
    async def analyze_project(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single project and generate monetization strategy.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Complete monetization analysis or None
        """
        project_name = project_path.name
        self.logger.info(f"Analyzing project: {project_name}")
        
        # Check if monetization strategy already exists
        monetization_file = project_path / "monetization_strategy.md"
        if monetization_file.exists():
            self.logger.info(f"Monetization strategy already exists for {project_name}")
            return self._load_existing_strategy(monetization_file)
        
        # Gather project information
        project_info = self._gather_project_info(project_path)
        
        if not project_info:
            self.logger.warning(f"Could not gather info for {project_name}")
            return None
        
        # Generate monetization strategy
        strategy = await self._generate_monetization_strategy(project_info)
        
        if strategy:
            # Save the strategy
            await self._save_monetization_strategy(project_path, strategy)
            
            # Also save market analysis
            market_analysis = await self._generate_market_analysis(project_info, strategy)
            if market_analysis:
                await self._save_market_analysis(project_path, market_analysis)
        
        return strategy
    
    def _gather_project_info(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Gather information about the project from various sources."""
        info = {
            'name': project_path.name,
            'path': str(project_path),
            'files': []
        }
        
        # Read README if exists
        readme_path = project_path / "README.md"
        if readme_path.exists():
            try:
                info['readme'] = readme_path.read_text()
            except Exception as e:
                self.logger.error(f"Error reading README: {e}")
        
        # Read ARCHITECTURE.md if exists - NEW
        architecture_path = project_path / "ARCHITECTURE.md"
        if architecture_path.exists():
            try:
                info['architecture'] = architecture_path.read_text()
                self.logger.info(f"Found ARCHITECTURE.md for {project_path.name}")
            except Exception as e:
                self.logger.error(f"Error reading ARCHITECTURE.md: {e}")
        
        # Read metadata.json if exists
        metadata_path = project_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    info.update(metadata)
            except Exception as e:
                self.logger.error(f"Error reading metadata: {e}")
        
        # Read package.json for tech stack
        package_path = project_path / "package.json"
        if package_path.exists():
            try:
                with open(package_path, 'r') as f:
                    package = json.load(f)
                    info['tech_stack'] = list(package.get('dependencies', {}).keys())
                    info['description'] = package.get('description', '')
            except Exception as e:
                self.logger.error(f"Error reading package.json: {e}")
        
        # Read composer.json for PHP projects
        composer_path = project_path / "composer.json"
        if composer_path.exists():
            try:
                with open(composer_path, 'r') as f:
                    composer = json.load(f)
                    info['tech_stack'] = info.get('tech_stack', []) + ['php', 'laravel']
                    if not info.get('description'):
                        info['description'] = composer.get('description', '')
            except Exception as e:
                self.logger.error(f"Error reading composer.json: {e}")
        
        # List key directories
        for subdir in ['app', 'src', 'components', 'services']:
            subdir_path = project_path / subdir
            if subdir_path.exists():
                info['files'].append(subdir)
        
        return info if info.get('description') else self._infer_project_info(info)
    
    def _infer_project_info(self, basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Infer project information from name and structure."""
        name = basic_info['name']
        
        # Common patterns in project names
        patterns = {
            'dashboard': 'Analytics and visualization platform',
            'ai': 'AI-powered solution',
            'marketplace': 'Connect buyers and sellers',
            'hub': 'Central platform for coordination',
            'tracker': 'Monitor and track metrics',
            'manager': 'Manage resources efficiently',
            'sync': 'Synchronize data across platforms'
        }
        
        description = basic_info.get('description', '')
        if not description:
            for pattern, desc in patterns.items():
                if pattern in name.lower():
                    description = desc
                    break
        
        basic_info['description'] = description or f"Project {name}"
        basic_info['inferred'] = True
        
        return basic_info
    
    async def _generate_monetization_strategy(self, project_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a complete monetization strategy for the project."""
        # Include README and ARCHITECTURE context if available
        readme_context = ""
        if 'readme' in project_info:
            readme_context = f"\n\nREADME Content (for understanding features):\n{project_info['readme'][:2000]}"
            
        architecture_context = ""
        if 'architecture' in project_info:
            architecture_context = f"\n\nARCHITECTURE Content (for technical capabilities):\n{project_info['architecture'][:2000]}"
        
        prompt = f"""
        Analyze this existing project and create a comprehensive monetization strategy:
        
        Project: {project_info['name']}
        Description: {project_info.get('description', 'N/A')}
        Tech Stack: {', '.join(project_info.get('tech_stack', []))}
        {readme_context}
        {architecture_context}
        
        Based on the project documentation above, create a monetization strategy that:
        1. Aligns pricing tiers with ACTUAL features from ARCHITECTURE.md (NOT README)
        2. Leverages the technical architecture described in ARCHITECTURE.md
        3. Prices services/APIs that actually exist in the system
        4. Targets the audience mentioned in the documentation
        5. IMPORTANT: Extract features from ARCHITECTURE.md sections, not README
        
        Create a complete monetization strategy including:
        - Executive summary with specific MRR target
        - Detailed value proposition based on documented features
        - Pricing tiers (Free, Pro, Enterprise) aligned with real features
        - Multiple revenue streams based on actual capabilities
        - Customer acquisition strategy for target audience
        - Financial projections for 12 months
        
        Return as structured data matching our monetization template.
        Format as JSON with all necessary fields.
        """
        
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        if not response or response.get('status') != 'success':
            return None
        
        try:
            # Parse the response and structure it
            result_text = response.get('result', '')
            
            # Create structured monetization data
            strategy = await self._structure_monetization_data(project_info, result_text)
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error generating monetization strategy: {e}")
            return None
    
    async def _structure_monetization_data(self, project_info: Dict[str, Any], ai_response: str) -> Dict[str, Any]:
        """Structure the AI response into our monetization template format."""
        # Use AI to structure the response properly
        structure_prompt = f"""
        Convert this monetization analysis into structured JSON data:
        
        {ai_response}
        
        Create a complete JSON object with these exact fields:
        {{
            "project_name": "{project_info['name']}",
            "executive_summary": {{
                "target_mrr": number,
                "timeframe": "6 months",
                "revenue_model": "subscription/marketplace/usage-based",
                "target_audience": "specific segment",
                "problem_statement": "what problem they have"
            }},
            "value_proposition": {{
                "problem": {{}},
                "solution": {{}},
                "value_calculation": {{}}
            }},
            "pricing_tiers": [
                {{
                    "name": "Free",
                    "price": 0,
                    "features": [],
                    "limits": {{}},
                    "purpose": ""
                }},
                {{
                    "name": "Pro",
                    "price": number,
                    "features": [],
                    "limits": {{}},
                    "value_prop": ""
                }},
                {{
                    "name": "Enterprise",
                    "price": "custom",
                    "features": [],
                    "value_prop": ""
                }}
            ],
            "revenue_streams": {{
                "primary": {{}},
                "secondary": {{}},
                "tertiary": {{}}
            }},
            "growth_strategy": {{}},
            "financial_projections": {{
                "months": [
                    {{"month": 1, "users_free": 100, "users_paid": 2, "mrr": 58}},
                    {{"month": 3, "users_free": 500, "users_paid": 25, "mrr": 725}},
                    {{"month": 6, "users_free": 2000, "users_paid": 140, "mrr": 4060}},
                    {{"month": 12, "users_free": 10000, "users_paid": 700, "mrr": 20300}}
                ],
                "unit_economics": {{
                    "cac": 50,
                    "ltv": 870,
                    "ltv_cac_ratio": 17.4,
                    "payback_months": 2
                }}
            }}
        }}
        """
        
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': structure_prompt})
        
        if response and response.get('status') == 'success':
            try:
                result_text = response.get('result', '')
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                self.logger.error(f"Error structuring monetization data: {e}")
        
        # Return a basic structure if parsing fails
        return {
            "project_name": project_info['name'],
            "executive_summary": {
                "target_mrr": 10000,
                "timeframe": "6 months",
                "revenue_model": "subscription",
                "target_audience": "small businesses",
                "problem_statement": "inefficient processes"
            }
        }
    
    async def _save_monetization_strategy(self, project_path: Path, strategy: Dict[str, Any]) -> None:
        """Save the monetization strategy to a markdown file."""
        # Generate monetization content dynamically
        filled_content = await self._generate_monetization_content(strategy)
        
        # Save to project directory
        output_path = project_path / "monetization_strategy.md"
        output_path.write_text(filled_content)
        
        self.logger.info(f"Saved monetization strategy to {output_path}")
    
    async def _generate_monetization_content(self, data: Dict[str, Any]) -> str:
        """Generate monetization strategy content dynamically."""
        # Extract key data
        project_name = data.get('project_name', 'Project')
        exec_summary = data.get('executive_summary', {})
        pricing_tiers = data.get('pricing_tiers', [])
        projections = data.get('financial_projections', {}).get('months', [])
        unit_econ = data.get('financial_projections', {}).get('unit_economics', {})
        
        prompt = f"""
        Create a comprehensive MONETIZATION_STRATEGY.md document for {project_name}.
        
        Project Context:
        - Target MRR: ${exec_summary.get('target_mrr', 10000)}
        - Revenue Model: {exec_summary.get('revenue_model', 'subscription')}
        - Target Audience: {exec_summary.get('target_audience', 'businesses')}
        - Problem Statement: {exec_summary.get('problem_statement', 'addresses market challenges')}
        - Timeframe: {exec_summary.get('timeframe', '6 months')}
        
        Pricing Tiers:
        {json.dumps(pricing_tiers, indent=2)}
        
        Financial Projections:
        {json.dumps(projections, indent=2)}
        
        Unit Economics:
        - CAC: ${unit_econ.get('cac', 50)}
        - LTV: ${unit_econ.get('ltv', 1000)}
        - LTV:CAC Ratio: {unit_econ.get('ltv_cac_ratio', 20)}:1
        
        Create a professional monetization strategy document with:
        1. Executive Summary
        2. Value Proposition Analysis
        3. Revenue Model and Pricing Strategy
        4. 24/7 Revenue Generation Mechanisms
        5. Customer Acquisition Strategy
        6. Financial Projections (12-month forecast)
        7. Unit Economics Analysis
        8. Growth Strategy
        9. Risk Analysis
        10. Success Metrics and Milestones
        
        Format as a complete Markdown document.
        """
        
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        if response and response.get('status') == 'success' and response.get('result'):
            return response['result']
        
        # Fallback to structured format if AI generation fails
        return self._create_fallback_content(data)
    
    def _create_fallback_content(self, data: Dict[str, Any]) -> str:
        """Create fallback monetization content if AI fails."""
        project_name = data.get('project_name', 'Project')
        exec_summary = data.get('executive_summary', {})
        pricing_tiers = data.get('pricing_tiers', [])
        projections = data.get('financial_projections', {}).get('months', [])
        unit_econ = data.get('financial_projections', {}).get('unit_economics', {})
        
        # Build pricing tiers section
        pricing_text = ""
        for tier in pricing_tiers:
            pricing_text += f"\n### {tier.get('name', 'Tier')} - ${tier.get('price', 0)}/month\n"
            pricing_text += f"Target: {tier.get('target_users', 'General users')}\n\n"
            pricing_text += "Features:\n"
            for feature in tier.get('features', []):
                pricing_text += f"- {feature}\n"
        
        # Build projections table
        proj_table = "| Month | MRR | Paid Users | Free Users |\n|-------|-----|------------|------------|\n"
        for proj in projections:
            proj_table += f"| {proj.get('month', 0)} | ${proj.get('mrr', 0):,} | {proj.get('users_paid', 0)} | {proj.get('users_free', 0)} |\n"
        
        return f"""# {project_name} Monetization Strategy

## Executive Summary

{project_name} targets ${exec_summary.get('target_mrr', 10000):,} MRR within {exec_summary.get('timeframe', '6 months')} through a {exec_summary.get('revenue_model', 'subscription')} model serving {exec_summary.get('target_audience', 'businesses')}.

**Problem Statement**: {exec_summary.get('problem_statement', 'Addresses critical market challenges')}

## Value Proposition

{data.get('value_proposition', {}).get('description', 'Delivers significant value through innovative solutions.')}

## Revenue Model

### Pricing Strategy
{pricing_text}

## 24/7 Revenue Generation

- Self-service signup and onboarding
- Automated billing and subscription management
- Global availability across all timezones
- Usage-based upgrade triggers

## Financial Projections

### Monthly Forecast
{proj_table}

### Unit Economics
- **Customer Acquisition Cost (CAC)**: ${unit_econ.get('cac', 50)}
- **Lifetime Value (LTV)**: ${unit_econ.get('ltv', 1000)}
- **LTV:CAC Ratio**: {unit_econ.get('ltv_cac_ratio', 20)}:1
- **Payback Period**: {unit_econ.get('payback_months', 2)} months
- **Gross Margin**: {unit_econ.get('gross_margin', 80)}%

## Growth Strategy

{data.get('growth_strategy', {}).get('description', 'Multi-channel growth approach focusing on product-led growth, content marketing, and strategic partnerships.')}

## Risk Analysis

{data.get('risk_analysis', {}).get('description', 'Key risks include competition, market size limitations, and technical scalability challenges.')}

## Success Metrics

- MRR Growth Rate: {data.get('success_metrics', {}).get('mrr_growth_target', 50)}% month-over-month
- Customer Churn: < {data.get('success_metrics', {}).get('churn_target', 5)}% monthly
- CAC Payback: < {data.get('success_metrics', {}).get('payback_target', 3)} months

---
*Generated on {datetime.now(timezone.utc).isoformat()}*
"""
    
    async def _generate_market_analysis(self, project_info: Dict[str, Any], strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate market analysis document."""
        prompt = f"""
        Create a market analysis for {project_info['name']}:
        
        Target audience: {strategy.get('executive_summary', {}).get('target_audience', 'N/A')}
        Revenue model: {strategy.get('executive_summary', {}).get('revenue_model', 'N/A')}
        
        Analyze:
        1. Total addressable market (TAM)
        2. Serviceable addressable market (SAM)
        3. Serviceable obtainable market (SOM)
        4. Main competitors and their weaknesses
        5. Market trends favoring this solution
        6. Barriers to entry
        7. Growth potential over 3 years
        
        Format as structured market analysis document.
        """
        
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        if response and response.get('status') == 'success':
            return {
                'analysis': response.get('result', ''),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
        
        return None
    
    async def _save_market_analysis(self, project_path: Path, analysis: Dict[str, Any]) -> None:
        """Save market analysis to project directory."""
        content = f"""# Market Analysis for {project_path.name}

Generated: {analysis['generated_at']}

{analysis['analysis']}
"""
        
        output_path = project_path / "market_analysis.md"
        output_path.write_text(content)
        
        self.logger.info(f"Saved market analysis to {output_path}")
    
    def _load_existing_strategy(self, strategy_file: Path) -> Dict[str, Any]:
        """Load and parse existing monetization strategy."""
        try:
            content = strategy_file.read_text()
            # Extract key metrics from the markdown
            return {
                'exists': True,
                'path': str(strategy_file),
                'content_preview': content[:500]
            }
        except Exception as e:
            self.logger.error(f"Error loading existing strategy: {e}")
            return {'exists': True, 'error': str(e)}
    
    async def generate_monetization_report(self) -> str:
        """Generate a report of all projects and their monetization status."""
        all_projects = await self.analyze_all_projects()
        
        report = f"""# Monetization Analysis Report
Generated: {datetime.now(timezone.utc).isoformat()}

## Summary
Total Projects Analyzed: {len(all_projects)}

## Projects

"""
        
        for project_name, analysis in all_projects.items():
            if analysis.get('exists'):
                report += f"### ✅ {project_name}\n"
                report += f"- Status: Monetization strategy exists\n"
                report += f"- Path: {analysis.get('path', 'N/A')}\n\n"
            else:
                report += f"### 📝 {project_name}\n"
                report += f"- Status: Generated new monetization strategy\n"
                if 'executive_summary' in analysis:
                    summary = analysis['executive_summary']
                    report += f"- Target MRR: ${summary.get('target_mrr', 'N/A')}\n"
                    report += f"- Model: {summary.get('revenue_model', 'N/A')}\n"
                    report += f"- Audience: {summary.get('target_audience', 'N/A')}\n"
                report += "\n"
        
        return report