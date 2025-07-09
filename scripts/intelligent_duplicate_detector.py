"""
Intelligent Duplicate Detection System using AI

This module provides AI-powered duplicate detection that doesn't rely on
hardcoded patterns or rules. It uses the AI brain to understand semantic
similarities and make intelligent decisions about duplicates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

from scripts.duplicate_prevention_system import ProjectMetadata


class IntelligentDuplicateDetector:
    """AI-powered duplicate detection system."""
    
    def __init__(self, local_repos_path: Path, ai_brain):
        """Initialize the intelligent duplicate detector.
        
        Args:
            local_repos_path: Path to local repositories directory
            ai_brain: AI brain for intelligent analysis
        """
        self.local_repos_path = local_repos_path
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Cache for metadata
        self._metadata_cache: Dict[str, ProjectMetadata] = {}
        
    def load_all_metadata(self) -> Dict[str, ProjectMetadata]:
        """Load metadata from all local repositories."""
        self._metadata_cache.clear()
        
        if not self.local_repos_path.exists():
            self.logger.warning(f"Local repos path does not exist: {self.local_repos_path}")
            return self._metadata_cache
        
        # Check if we're looking at a repositories directory
        repositories_path = self.local_repos_path / 'repositories'
        if repositories_path.exists():
            # Look for organization folders (e.g., CodeWebMobile-AI)
            for org_dir in repositories_path.iterdir():
                if not org_dir.is_dir():
                    continue
                
                # Look for project folders within organization
                for project_dir in org_dir.iterdir():
                    if not project_dir.is_dir():
                        continue
                    
                    metadata_path = project_dir / 'metadata.json'
                    if metadata_path.exists():
                        metadata = ProjectMetadata.from_file(metadata_path)
                        if metadata:
                            self._metadata_cache[metadata.repository_name] = metadata
                            self.logger.debug(f"Loaded metadata for {metadata.repository_name}")
        
        # Also check direct subdirectories for backward compatibility
        for repo_dir in self.local_repos_path.iterdir():
            if not repo_dir.is_dir() or repo_dir.name == 'repositories':
                continue
                
            metadata_path = repo_dir / 'metadata.json'
            if metadata_path.exists():
                metadata = ProjectMetadata.from_file(metadata_path)
                if metadata:
                    self._metadata_cache[metadata.repository_name] = metadata
                    
        self.logger.info(f"Loaded metadata for {len(self._metadata_cache)} projects")
        return self._metadata_cache
        
    async def check_duplicate_with_ai(self, new_project: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a new project would be a duplicate using AI analysis.
        
        Args:
            new_project: Dictionary with project details
                
        Returns:
            Dictionary with duplicate detection results
        """
        # Load existing metadata
        existing_metadata = self.load_all_metadata()
        
        if not existing_metadata:
            return {
                'is_duplicate': False,
                'similar_projects': [],
                'ai_analysis': 'No existing projects to compare against',
                'recommendation': 'create_new'
            }
        
        # Prepare existing projects summary for AI
        existing_projects = []
        for repo_name, metadata in existing_metadata.items():
            existing_projects.append({
                'name': metadata.display_name,
                'repository': repo_name,
                'description': metadata.description,
                'features': metadata.key_features[:5]  # First 5 features
            })
        
        # Use AI to analyze similarities
        prompt = f"""
        Analyze this new project idea for potential duplicates among existing projects.
        
        NEW PROJECT:
        - Name: {new_project.get('name', 'Unnamed')}
        - Description: {new_project.get('description', '')}
        - Problem Statement: {new_project.get('problem_statement', '')}
        - Target Market: {new_project.get('target_market', '')}
        - Key Features: {json.dumps(new_project.get('initial_features', [])[:5])}
        - Monetization: {new_project.get('monetization_model', '')}
        
        EXISTING PROJECTS:
        {json.dumps(existing_projects, indent=2)}
        
        Perform a comprehensive analysis:
        
        1. Identify any existing projects that solve similar problems
        2. Check for feature overlap (not just exact matches, but functional equivalents)
        3. Consider if the target audiences overlap significantly
        4. Determine if the new project offers unique value or is redundant
        5. Consider if combining/enhancing an existing project would be better
        
        Return a JSON response with:
        {{
            "duplicate_analysis": {{
                "is_duplicate": true/false,
                "confidence": 0.0-1.0,
                "primary_reason": "Main reason for decision"
            }},
            "similar_projects": [
                {{
                    "repository_name": "repo-name",
                    "display_name": "Project Name",
                    "similarity_score": 0.0-1.0,
                    "similarity_type": "exact|high|moderate|low",
                    "overlapping_aspects": ["aspect1", "aspect2"],
                    "key_differences": ["diff1", "diff2"],
                    "could_merge": true/false
                }}
            ],
            "recommendation": {{
                "action": "create_new|enhance_existing|merge_projects|pivot_idea",
                "explanation": "Detailed explanation",
                "suggested_approach": "Specific suggestion"
            }},
            "unique_value_analysis": {{
                "unique_aspects": ["aspect1", "aspect2"],
                "market_gaps_filled": ["gap1", "gap2"],
                "innovation_score": 0.0-1.0
            }},
            "differentiation_suggestions": [
                "suggestion1",
                "suggestion2"
            ]
        }}
        
        Be thorough but avoid false positives. Projects can target similar markets but solve different problems.
        Consider functional similarity, not just keyword matching.
        """
        
        # Define schema for structured output - fixed for all providers
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_duplicate": {"type": "boolean"},
                "duplicate_analysis": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "highest_similarity_score": {"type": "number"},
                        "primary_duplicate": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["highest_similarity_score", "primary_duplicate", "reasoning"]
                },
                "similar_projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "repository_name": {"type": "string"},
                            "display_name": {"type": "string"},
                            "similarity_score": {"type": "number"},
                            "similarity_type": {
                                "type": "string",
                                "enum": ["exact", "high", "moderate", "low"]
                            },
                            "overlapping_aspects": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "key_differences": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "could_merge": {"type": "boolean"}
                        }
                    }
                },
                "recommendation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create_new", "enhance_existing", "merge_projects", "pivot_idea"]
                        },
                        "explanation": {"type": "string"},
                        "suggested_approach": {"type": "string"}
                    }
                },
                "unique_value_analysis": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unique_aspects": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "market_gaps_filled": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "innovation_score": {"type": "number"}
                    }
                },
                "differentiation_suggestions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["is_duplicate", "duplicate_analysis", "similar_projects", "recommendation", "unique_value_analysis", "differentiation_suggestions"]
        }
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt, response_schema=schema)
            
            if response and response.get('structured_output'):
                analysis = response.get('parsed_content')
            elif response:
                # Fallback to parsing for non-structured output
                result_text = response.get('content', '')
                
                # Parse JSON from response with better error handling
                import re
                
                # Try to extract JSON - use the improved parsing from gemini_deep_research
                # Remove markdown code blocks if present
                content = result_text.replace('```json', '').replace('```', '').strip()
                
                # Try to parse entire content first
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError:
                    # Try to find JSON object with balanced braces
                    start_idx = content.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        end_idx = -1
                        
                        for i in range(start_idx, len(content)):
                            char = content[i]
                            
                            if escape_next:
                                escape_next = False
                                continue
                            
                            if char == '\\' and in_string:
                                escape_next = True
                                continue
                            
                            if char == '"':
                                in_string = not in_string
                                continue
                            
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                        
                        if end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            # Clean up common JSON issues
                            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                            json_str = re.sub(r',\s*]', ']', json_str)
                            
                            analysis = json.loads(json_str)
                        else:
                            raise ValueError("Could not find complete JSON object")
                    else:
                        raise ValueError("No JSON object found in response")
                
                # Format the response
                return self._format_ai_response(analysis, new_project)
        except Exception as e:
            self.logger.error(f"AI duplicate detection failed: {e}")
            
        # Fallback response
        return {
            'is_duplicate': False,
            'similar_projects': [],
            'ai_analysis': 'AI analysis failed, proceeding with caution',
            'recommendation': 'create_new'
        }
        
    def _format_ai_response(self, analysis: Dict[str, Any], 
                          new_project: Dict[str, Any]) -> Dict[str, Any]:
        """Format AI analysis into standardized response."""
        
        # Extract duplicate status - it's at the root level now
        is_duplicate = analysis.get('is_duplicate', False)
        dup_analysis = analysis.get('duplicate_analysis', {})
        confidence = dup_analysis.get('highest_similarity_score', 0.0)
        
        # Extract similar projects
        similar_projects = []
        for proj in analysis.get('similar_projects', []):
            similar_projects.append({
                'repository_name': proj.get('repository_name'),
                'display_name': proj.get('display_name'),
                'similarity_score': proj.get('similarity_score', 0.0),
                'similarity_type': proj.get('similarity_type', 'low'),
                'reasons': proj.get('overlapping_aspects', []),
                'differences': proj.get('key_differences', []),
                'could_merge': proj.get('could_merge', False)
            })
        
        # Sort by similarity score
        similar_projects.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Extract recommendation
        recommendation = analysis.get('recommendation', {})
        action = recommendation.get('action', 'create_new')
        
        # Build summary
        if is_duplicate and confidence > 0.8:
            summary = f"High confidence duplicate detected ({confidence:.0%}). {dup_analysis.get('primary_reason', '')}"
        elif is_duplicate:
            summary = f"Potential duplicate detected ({confidence:.0%}). {dup_analysis.get('primary_reason', '')}"
        elif similar_projects and similar_projects[0]['similarity_score'] > 0.6:
            summary = f"Similar projects exist but unique value identified. Consider: {recommendation.get('suggested_approach', '')}"
        else:
            summary = "No significant duplicates found. Project appears unique."
        
        # Get unique value analysis
        unique_analysis = analysis.get('unique_value_analysis', {})
        
        return {
            'is_duplicate': is_duplicate,
            'duplicate_confidence': confidence,
            'similar_projects': similar_projects[:5],  # Top 5
            'ai_analysis': summary,
            'recommendation': action,
            'recommendation_details': recommendation,
            'unique_aspects': unique_analysis.get('unique_aspects', []),
            'innovation_score': unique_analysis.get('innovation_score', 0.0),
            'suggested_differentiators': analysis.get('differentiation_suggestions', []),
            'market_gaps': unique_analysis.get('market_gaps_filled', [])
        }
        
    async def analyze_portfolio_overlap(self, new_project: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how a new project would fit into the existing portfolio.
        
        Args:
            new_project: New project details
            
        Returns:
            Portfolio overlap analysis
        """
        existing_metadata = self.load_all_metadata()
        
        if not existing_metadata:
            return {
                'portfolio_fit': 'first_project',
                'synergies': [],
                'conflicts': [],
                'recommendation': 'Great first project for the portfolio'
            }
        
        # Prepare portfolio summary
        portfolio_summary = {
            'total_projects': len(existing_metadata),
            'project_types': {},
            'problem_domains': [],
            'target_markets': []
        }
        
        for metadata in existing_metadata.values():
            # Count project types
            proj_type = metadata.project_type or 'Unknown'
            portfolio_summary['project_types'][proj_type] = portfolio_summary['project_types'].get(proj_type, 0) + 1
            
            # Collect domains and markets (from descriptions)
            portfolio_summary['problem_domains'].append(metadata.description[:100])
            
        prompt = f"""
        Analyze how this new project would fit into the existing portfolio.
        
        NEW PROJECT:
        {json.dumps(new_project, indent=2)}
        
        PORTFOLIO SUMMARY:
        {json.dumps(portfolio_summary, indent=2)}
        
        Analyze:
        1. Portfolio diversification impact
        2. Potential synergies with existing projects
        3. Resource allocation conflicts
        4. Market positioning strategy
        5. Technology stack compatibility
        
        Return JSON with:
        {{
            "portfolio_fit_score": 0.0-1.0,
            "diversification_impact": "positive|neutral|negative",
            "synergies": [
                {{
                    "project": "project-name",
                    "synergy_type": "technical|market|resource",
                    "description": "how they complement"
                }}
            ],
            "conflicts": [
                {{
                    "project": "project-name", 
                    "conflict_type": "market|resource|technical",
                    "description": "nature of conflict"
                }}
            ],
            "strategic_value": "high|medium|low",
            "recommendation": "Specific portfolio strategy recommendation"
        }}
        """
        
        try:
            response = await self.ai_brain.execute_capability(
                'problem_analysis', 
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                result_text = response.get('result', '')
                
                # Use same JSON parsing logic
                content = result_text.replace('```json', '').replace('```', '').strip()
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to find JSON object with balanced braces
                    start_idx = content.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        end_idx = -1
                        
                        for i in range(start_idx, len(content)):
                            char = content[i]
                            
                            if escape_next:
                                escape_next = False
                                continue
                            
                            if char == '\\' and in_string:
                                escape_next = True
                                continue
                            
                            if char == '"':
                                in_string = not in_string
                                continue
                            
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                        
                        if end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                            
                            return json.loads(json_str)
                    
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {e}")
            
        return {
            'portfolio_fit_score': 0.5,
            'recommendation': 'Unable to perform portfolio analysis'
        }
        
    def get_metadata_for_update(self, new_project: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced metadata for the new project including duplicate analysis.
        
        Args:
            new_project: New project details
            
        Returns:
            Enhanced metadata dictionary
        """
        # Start with standard metadata
        metadata = {
            'repository_name': new_project.get('name', ''),
            'display_name': new_project.get('display_name', new_project.get('name', '')),
            'description': new_project.get('description', ''),
            'tech_stack': new_project.get('tech_stack', 'Laravel + React + MySQL + Redis + laravel-echo-server'),
            'project_type': new_project.get('project_type', 'Web Application'),
            'key_features': new_project.get('initial_features', []),
            'created_at': datetime.now().isoformat(),
            'created_by': 'CWMAI System',
            
            # Enhanced fields for better duplicate detection
            'problem_domain': new_project.get('problem_statement', ''),
            'target_audience': new_project.get('target_market', ''),
            'monetization_model': new_project.get('monetization_model', ''),
            'core_functionality': self._extract_core_functionality(new_project),
            'unique_value_proposition': new_project.get('unique_value_proposition', ''),
            'market_category': new_project.get('market_category', '')
        }
        
        return metadata
        
    def _extract_core_functionality(self, project: Dict[str, Any]) -> str:
        """Extract a summary of core functionality from project details."""
        # Combine key aspects
        elements = []
        
        if project.get('problem_statement'):
            elements.append(f"Solves: {project['problem_statement']}")
            
        if project.get('solution_approach'):
            elements.append(f"Approach: {project['solution_approach']}")
            
        if project.get('initial_features'):
            top_features = project['initial_features'][:3]
            elements.append(f"Core features: {', '.join(top_features)}")
            
        return ' | '.join(elements)