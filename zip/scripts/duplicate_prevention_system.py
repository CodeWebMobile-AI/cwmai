"""
Enhanced Duplicate Prevention System

This module provides intelligent duplicate detection for projects by:
1. Reading metadata.json from local repositories
2. Performing semantic similarity checks
3. Comparing problem domains and target audiences
4. Validating against existing features
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import difflib
import re

@dataclass
class ProjectMetadata:
    """Structured project metadata from metadata.json"""
    repository_name: str
    display_name: str
    description: str
    tech_stack: str
    project_type: str
    key_features: List[str]
    created_at: str
    created_by: str
    problem_domain: Optional[str] = None
    target_audience: Optional[str] = None
    monetization_model: Optional[str] = None
    core_functionality: Optional[str] = None
    
    @classmethod
    def from_file(cls, metadata_path: Path) -> Optional['ProjectMetadata']:
        """Load metadata from file."""
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return cls(
                    repository_name=data.get('repository_name', ''),
                    display_name=data.get('display_name', ''),
                    description=data.get('description', ''),
                    tech_stack=data.get('tech_stack', ''),
                    project_type=data.get('project_type', ''),
                    key_features=data.get('key_features', []),
                    created_at=data.get('created_at', ''),
                    created_by=data.get('created_by', ''),
                    problem_domain=data.get('problem_domain'),
                    target_audience=data.get('target_audience'),
                    monetization_model=data.get('monetization_model'),
                    core_functionality=data.get('core_functionality')
                )
        except Exception as e:
            logging.error(f"Failed to load metadata from {metadata_path}: {e}")
            return None


class DuplicatePreventionSystem:
    """Advanced duplicate detection system for projects."""
    
    def __init__(self, local_repos_path: Path, ai_brain=None):
        """Initialize the duplicate prevention system.
        
        Args:
            local_repos_path: Path to local repositories directory
            ai_brain: Optional AI brain for advanced similarity checks
        """
        self.local_repos_path = local_repos_path
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Cache loaded metadata
        self._metadata_cache: Dict[str, ProjectMetadata] = {}
        self._last_scan: Optional[datetime] = None
        
        # Similarity thresholds
        self.name_similarity_threshold = 0.8
        self.description_similarity_threshold = 0.7
        self.feature_overlap_threshold = 0.6
        self.domain_match_threshold = 0.9
        
    def load_all_metadata(self, force_reload: bool = False) -> Dict[str, ProjectMetadata]:
        """Load metadata from all local repositories.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary of repository_name -> ProjectMetadata
        """
        if not force_reload and self._metadata_cache:
            return self._metadata_cache
            
        self._metadata_cache.clear()
        
        if not self.local_repos_path.exists():
            self.logger.warning(f"Local repos path does not exist: {self.local_repos_path}")
            return self._metadata_cache
            
        for repo_dir in self.local_repos_path.iterdir():
            if not repo_dir.is_dir():
                continue
                
            metadata_path = repo_dir / 'metadata.json'
            if metadata_path.exists():
                metadata = ProjectMetadata.from_file(metadata_path)
                if metadata:
                    self._metadata_cache[metadata.repository_name] = metadata
                    
        self._last_scan = datetime.now()
        self.logger.info(f"Loaded metadata for {len(self._metadata_cache)} projects")
        return self._metadata_cache
        
    def check_duplicate(self, new_project: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a new project would be a duplicate.
        
        Args:
            new_project: Dictionary with project details including:
                - name: Project name
                - description: Project description
                - problem_statement: Problem being solved
                - target_market: Target audience
                - initial_features: List of features
                - monetization_model: How it makes money
                
        Returns:
            Dictionary with:
                - is_duplicate: Boolean indicating if duplicate
                - similar_projects: List of similar existing projects
                - similarity_reasons: Detailed reasons for similarity
                - recommendation: What to do (create new, enhance existing, etc.)
        """
        # Load all existing metadata
        existing_metadata = self.load_all_metadata()
        
        similar_projects = []
        
        for repo_name, metadata in existing_metadata.items():
            similarity_score, reasons = self._calculate_similarity(new_project, metadata)
            
            if similarity_score > 0.5:  # Significant similarity
                similar_projects.append({
                    'repository_name': repo_name,
                    'display_name': metadata.display_name,
                    'description': metadata.description,
                    'similarity_score': similarity_score,
                    'reasons': reasons
                })
                
        # Sort by similarity score
        similar_projects.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Determine if it's a duplicate
        is_duplicate = False
        recommendation = "create_new"
        
        if similar_projects:
            top_similarity = similar_projects[0]['similarity_score']
            
            if top_similarity > 0.9:
                is_duplicate = True
                recommendation = "exact_duplicate"
            elif top_similarity > 0.75:
                is_duplicate = True
                recommendation = "enhance_existing"
            elif top_similarity > 0.6:
                recommendation = "differentiate_more"
                
        return {
            'is_duplicate': is_duplicate,
            'similar_projects': similar_projects[:5],  # Top 5 similar
            'similarity_reasons': self._summarize_similarities(similar_projects[:3]),
            'recommendation': recommendation,
            'suggested_differentiators': self._suggest_differentiators(new_project, similar_projects[:3])
        }
        
    def _calculate_similarity(self, new_project: Dict[str, Any], 
                            existing: ProjectMetadata) -> Tuple[float, List[str]]:
        """Calculate similarity between new project and existing project.
        
        Returns:
            Tuple of (similarity_score, reasons)
        """
        scores = []
        reasons = []
        
        # 1. Name similarity
        name_sim = self._string_similarity(
            new_project.get('name', '').lower(),
            existing.display_name.lower()
        )
        if name_sim > 0.4:  # Lower threshold for name matching
            scores.append(name_sim * 0.15)  # 15% weight
            reasons.append(f"Similar name ({name_sim:.0%} match)")
            
        # 2. Description similarity
        desc_sim = self._string_similarity(
            new_project.get('description', '').lower(),
            existing.description.lower()
        )
        if desc_sim > 0.3:  # Lower threshold for description
            scores.append(desc_sim * 0.2)  # 20% weight
            reasons.append(f"Similar description ({desc_sim:.0%} match)")
            
        # 3. Problem domain similarity
        # Check both problem_statement and description against existing description
        problem_score = 0.0
        if new_project.get('problem_statement'):
            problem_sim = self._semantic_similarity(
                new_project['problem_statement'],
                existing.description
            )
            problem_score = max(problem_score, problem_sim)
            
        if new_project.get('description'):
            desc_problem_sim = self._semantic_similarity(
                new_project['description'],
                existing.description
            )
            problem_score = max(problem_score, desc_problem_sim)
            
        if problem_score > 0.3:  # Lower threshold
            scores.append(problem_score * 0.35)  # 35% weight - most important
            reasons.append("Solving similar problem")
                
        # 4. Target audience similarity
        # Extract target audience from description if not explicitly provided
        target_from_desc = self._extract_target_from_description(existing.description)
        target_sim = self._target_audience_similarity(
            new_project.get('target_market', ''),
            target_from_desc
        )
        if target_sim > 0.4:
            scores.append(target_sim * 0.15)  # 15% weight
            reasons.append("Same target audience")
            
        # 5. Feature overlap
        feature_overlap = self._feature_overlap(
            new_project.get('initial_features', []),
            existing.key_features
        )
        if feature_overlap > 0.2:  # Lower threshold
            scores.append(feature_overlap * 0.15)  # 15% weight
            reasons.append(f"{int(feature_overlap * 100)}% feature overlap")
            
        # Calculate total score
        total_score = sum(scores) if scores else 0.0
        
        return min(total_score, 1.0), reasons
        
    def _extract_target_from_description(self, description: str) -> str:
        """Extract target audience from description text."""
        # Common patterns in descriptions
        patterns = [
            r'for\s+([\w\s]+?)(?:\.|,|:|$)',  # "for X"
            r'helps?\s+([\w\s]+?)(?:\.|,|:|$)',  # "helps X"
            r'designed for\s+([\w\s]+?)(?:\.|,|:|$)',  # "designed for X"
            r'perfect for\s+([\w\s]+?)(?:\.|,|:|$)',  # "perfect for X"
            r'targeting\s+([\w\s]+?)(?:\.|,|:|$)',  # "targeting X"
        ]
        
        desc_lower = description.lower()
        for pattern in patterns:
            match = re.search(pattern, desc_lower)
            if match:
                return match.group(1).strip()
                
        # If no pattern found, extract from keywords
        if 'students' in desc_lower:
            return 'students'
        elif 'professionals' in desc_lower:
            return 'professionals'
        elif 'businesses' in desc_lower:
            return 'businesses'
        elif 'families' in desc_lower:
            return 'families'
        elif 'developers' in desc_lower:
            return 'developers'
            
        return description[:50]  # Return first part of description as fallback
        
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using sequence matching."""
        return difflib.SequenceMatcher(None, str1, str2).ratio()
        
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        # Simple keyword-based similarity
        # Can be enhanced with AI brain if available
        
        # Extract key terms
        terms1 = self._extract_key_terms(text1.lower())
        terms2 = self._extract_key_terms(text2.lower())
        
        if not terms1 or not terms2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = terms1.intersection(terms2)
        union = terms1.union(terms2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'that', 'this', 'those', 'these', 'is',
            'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'need', 'platform', 'system', 'application', 'app', 'software'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter and return
        return {word for word in words if len(word) > 3 and word not in stop_words}
        
    def _target_audience_similarity(self, target1: str, target2: str) -> float:
        """Check if target audiences are similar."""
        # Common audience patterns
        audience_groups = {
            'business': ['business', 'company', 'enterprise', 'corporate', 'b2b'],
            'consumer': ['consumer', 'customer', 'individual', 'personal', 'b2c'],
            'developer': ['developer', 'programmer', 'engineer', 'technical'],
            'small_business': ['small business', 'smb', 'startup', 'entrepreneur'],
            'healthcare': ['healthcare', 'medical', 'health', 'patient', 'doctor'],
            'education': ['education', 'student', 'teacher', 'school', 'learning'],
            'finance': ['finance', 'financial', 'banking', 'investment', 'money']
        }
        
        # Find matching groups
        groups1 = set()
        groups2 = set()
        
        target1_lower = target1.lower()
        target2_lower = target2.lower()
        
        for group, keywords in audience_groups.items():
            if any(keyword in target1_lower for keyword in keywords):
                groups1.add(group)
            if any(keyword in target2_lower for keyword in keywords):
                groups2.add(group)
                
        # If same groups, high similarity
        if groups1 and groups2 and groups1.intersection(groups2):
            return 0.9
            
        # Otherwise check string similarity
        return self._string_similarity(target1, target2)
        
    def _feature_overlap(self, features1: List[str], features2: List[str]) -> float:
        """Calculate feature overlap between two feature lists."""
        if not features1 or not features2:
            return 0.0
            
        # Extract key terms from each feature
        terms1 = set()
        terms2 = set()
        
        for feature in features1:
            terms1.update(self._extract_key_terms(feature.lower()))
            
        for feature in features2:
            terms2.update(self._extract_key_terms(feature.lower()))
            
        if not terms1 or not terms2:
            return 0.0
            
        # Calculate overlap
        intersection = terms1.intersection(terms2)
        smaller_set = min(len(terms1), len(terms2))
        
        return len(intersection) / smaller_set if smaller_set > 0 else 0.0
        
    def _summarize_similarities(self, similar_projects: List[Dict[str, Any]]) -> str:
        """Create a summary of why projects are similar."""
        if not similar_projects:
            return "No similar projects found."
            
        summaries = []
        for project in similar_projects[:3]:
            reasons = project['reasons']
            score = project['similarity_score']
            name = project['display_name']
            
            summary = f"'{name}' ({score:.0%} similar): {', '.join(reasons)}"
            summaries.append(summary)
            
        return " | ".join(summaries)
        
    def _suggest_differentiators(self, new_project: Dict[str, Any], 
                                similar_projects: List[Dict[str, Any]]) -> List[str]:
        """Suggest ways to differentiate from similar projects."""
        suggestions = []
        
        if not similar_projects:
            return suggestions
            
        # Analyze what makes projects similar
        all_reasons = []
        for project in similar_projects:
            all_reasons.extend(project['reasons'])
            
        # Suggest differentiators based on similarity reasons
        if any('name' in reason.lower() for reason in all_reasons):
            suggestions.append("Choose a more distinctive project name")
            
        if any('description' in reason.lower() for reason in all_reasons):
            suggestions.append("Focus on unique value proposition in description")
            
        if any('problem' in reason.lower() for reason in all_reasons):
            suggestions.append("Target a more specific niche or use case")
            
        if any('audience' in reason.lower() for reason in all_reasons):
            suggestions.append("Consider a different market segment or geographic focus")
            
        if any('feature' in reason.lower() for reason in all_reasons):
            suggestions.append("Add innovative features not present in existing solutions")
            
        # Add general suggestions
        suggestions.extend([
            "Implement a unique monetization model",
            "Use different technology or architecture approach",
            "Focus on integration capabilities others lack",
            "Target an underserved segment of the market"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions
        
    async def validate_with_ai(self, new_project: Dict[str, Any], 
                              similar_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI for advanced duplicate validation.
        
        Args:
            new_project: New project details
            similar_projects: List of similar existing projects
            
        Returns:
            AI validation results
        """
        if not self.ai_brain:
            return {
                'ai_validation': False,
                'reason': 'AI brain not available'
            }
            
        prompt = f"""
        Analyze if this new project idea is truly different from existing projects:
        
        New Project:
        - Name: {new_project.get('name')}
        - Description: {new_project.get('description')}
        - Problem: {new_project.get('problem_statement')}
        - Target Market: {new_project.get('target_market')}
        - Features: {json.dumps(new_project.get('initial_features', []))}
        
        Similar Existing Projects:
        {json.dumps(similar_projects, indent=2)}
        
        Determine:
        1. Is this genuinely different or just a slight variation?
        2. Does it solve a unique aspect of the problem?
        3. Would users see value in having both solutions?
        4. What unique value does the new project bring?
        
        Return JSON with:
        {{
            "is_unique": true/false,
            "uniqueness_score": 0.0-1.0,
            "unique_aspects": ["aspect1", "aspect2"],
            "overlap_concerns": ["concern1", "concern2"],
            "recommendation": "proceed" or "reconsider" or "pivot",
            "pivot_suggestions": ["suggestion1", "suggestion2"] if applicable
        }}
        """
        
        try:
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                result = response.get('result', '')
                # Parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                    
        except Exception as e:
            self.logger.error(f"AI validation failed: {e}")
            
        return {
            'ai_validation': False,
            'reason': 'Failed to get AI validation'
        }
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the metadata cache."""
        return {
            'total_projects': len(self._metadata_cache),
            'last_scan': self._last_scan.isoformat() if self._last_scan else None,
            'projects_by_type': self._count_by_attribute('project_type'),
            'projects_by_tech': self._count_by_attribute('tech_stack'),
            'cache_size_bytes': sum(
                len(json.dumps(m.__dict__)) for m in self._metadata_cache.values()
            )
        }
        
    def _count_by_attribute(self, attribute: str) -> Dict[str, int]:
        """Count projects by attribute value."""
        counts = {}
        for metadata in self._metadata_cache.values():
            value = getattr(metadata, attribute, None)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return counts
        
    def clear_cache(self):
        """Clear the metadata cache."""
        self._metadata_cache.clear()
        self._last_scan = None
        self.logger.info("Metadata cache cleared")