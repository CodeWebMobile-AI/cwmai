"""
Research Summary Manager

Handles all research summary operations with validation, deduplication,
and quality control to maintain clean and useful research data.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class ValidatedProblem:
    """A validated problem entry with all required fields."""
    problem: str
    audience: str
    opportunity: str
    market_size: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'problem': self.problem,
            'audience': self.audience,
            'opportunity': self.opportunity,
            'market_size': self.market_size,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['ValidatedProblem']:
        """Create from dictionary with validation."""
        # Check all required fields are present and non-empty
        required_fields = ['problem', 'audience', 'opportunity', 'market_size', 'timestamp']
        for field in required_fields:
            if not data.get(field) or str(data.get(field)).strip() == '':
                return None
        
        return cls(
            problem=data['problem'].strip(),
            audience=data['audience'].strip(),
            opportunity=data['opportunity'].strip(),
            market_size=data['market_size'].strip(),
            timestamp=data['timestamp']
        )


class ResearchSummaryManager:
    """Manages research summary with validation and deduplication."""
    
    def __init__(self, summary_file: str = "research_knowledge/research_summary.json"):
        """Initialize the summary manager."""
        self.summary_file = Path(summary_file)
        self.logger = logging.getLogger(__name__)
        self.summary_data = self._load_summary()
        
        # Similarity threshold for duplicate detection (0-1)
        self.similarity_threshold = 0.7
        
        # Audience normalization map
        self.audience_map = {
            'small business owners': ['small businesses', 'small business', 'smb'],
            'freelancers': ['freelancer', 'independent contractors', 'contractors', 'gig workers'],
            'remote workers': ['remote teams', 'distributed teams', 'remote employees'],
            'developers': ['software developers', 'programmers', 'coders', 'engineers'],
            'students': ['learners', 'online learners', 'k-12 students'],
            'older adults': ['seniors', 'elderly', '65+', 'aging adults']
        }
        
    def _load_summary(self) -> Dict[str, Any]:
        """Load existing summary or create new one."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading summary: {e}")
        
        # Return default structure
        return {
            'total_research_sessions': 0,
            'all_sectors_discovered': [],
            'all_problems_discovered': [],
            'research_files': [],
            'total_unique_sectors': 0,
            'total_problems_discovered': 0,
            'last_research': None
        }
    
    def _save_summary(self):
        """Save summary to file."""
        try:
            # Ensure directory exists
            self.summary_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.summary_file, 'w') as f:
                json.dump(self.summary_data, f, indent=2)
                
            self.logger.info(f"Saved research summary to {self.summary_file}")
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
    
    def _normalize_audience(self, audience: str) -> str:
        """Normalize audience segment names."""
        if not audience:
            return ""
            
        # First normalize the text
        normalized = ' '.join(audience.strip().split())
        
        # Capitalize each word properly
        words = normalized.split()
        capitalized = []
        
        # Special cases that should remain uppercase
        uppercase_words = {'AI', 'SEO', 'SEM', 'B2B', 'B2C', 'API', 'CRM', 'ERP', 'HR', 'IT', 'UI', 'UX', 'K-12'}
        
        for i, word in enumerate(words):
            # Check if it's a known uppercase word (case-insensitive check)
            if word.upper() in uppercase_words:
                capitalized.append(word.upper())
            # Always capitalize first word
            elif i == 0:
                capitalized.append(word.capitalize())
            # Keep connector words lowercase unless at the start
            elif word.lower() in ['and', 'or', 'for', 'with', 'in', 'on', 'to', 'the', 'of']:
                capitalized.append(word.lower())
            else:
                capitalized.append(word.capitalize())
        
        normalized = ' '.join(capitalized)
        
        # Check against normalization map
        audience_lower = normalized.lower()
        for canonical, variations in self.audience_map.items():
            if audience_lower == canonical:
                # Already normalized, just return it
                return normalized
            for variation in variations:
                if variation in audience_lower:
                    # Replace with canonical form and re-normalize the capitalization
                    canonical_normalized = ' '.join(canonical.split())
                    words = canonical_normalized.split()
                    capitalized = []
                    
                    for i, word in enumerate(words):
                        if word.upper() in uppercase_words:
                            capitalized.append(word.upper())
                        elif i == 0:
                            capitalized.append(word.capitalize())
                        elif word.lower() in ['and', 'or', 'for', 'with', 'in', 'on', 'to', 'the', 'of']:
                            capitalized.append(word.lower())
                        else:
                            capitalized.append(word.capitalize())
                    
                    return ' '.join(capitalized)
        
        return normalized
    
    def _standardize_market_size(self, market_size: str) -> str:
        """Standardize market size format."""
        if not market_size or market_size.strip() == '':
            return "Under Analysis"
        
        market_lower = market_size.lower().strip()
        
        # Handle descriptive sizes
        if market_lower in ['large', 'small', 'medium', 'massive', 'huge', 'significant']:
            return market_lower.capitalize()
        
        # Handle "Large and growing" type patterns
        if 'and' in market_lower:
            parts = market_lower.split('and')
            capitalized_parts = [part.strip().capitalize() for part in parts]
            return ' and '.join(capitalized_parts)
        
        # Extract numeric values
        numbers = re.findall(r'\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)', market_size, re.IGNORECASE)
        if numbers:
            value, unit = numbers[0]
            unit_full = 'Billion' if unit.upper() in ['B', 'BILLION'] else 'Million'
            
            # Handle ranges
            if '-' in market_size or 'to' in market_size.lower():
                # Try to find second number
                all_numbers = re.findall(r'\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)', market_size, re.IGNORECASE)
                if len(all_numbers) >= 2:
                    val1, unit1 = all_numbers[0]
                    val2, unit2 = all_numbers[1]
                    unit1_full = 'Billion' if unit1.upper() in ['B', 'BILLION'] else 'Million'
                    unit2_full = 'Billion' if unit2.upper() in ['B', 'BILLION'] else 'Million'
                    return f"${val1} {unit1_full} - ${val2} {unit2_full}"
            
            return f"${value} {unit_full}"
        
        # Default formats
        if market_size.strip() == "$1B+":
            return "Under Analysis"
        
        return market_size
    
    def _is_valid_problem(self, problem: dict) -> bool:
        """Check if a problem entry is valid."""
        # Check required fields
        required_fields = ['problem', 'audience', 'market_size']
        for field in required_fields:
            if field not in problem:
                return False
            if not problem[field] or problem[field].strip() == '':
                return False
        
        # Check problem description is substantial
        if len(problem['problem'].strip()) < 10:
            return False
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _is_duplicate_problem(self, new_problem: ValidatedProblem) -> bool:
        """Check if a problem is duplicate of existing ones."""
        for existing in self.summary_data.get('all_problems_discovered', []):
            # Skip invalid entries
            validated_existing = ValidatedProblem.from_dict(existing)
            if not validated_existing:
                continue
            
            # Check problem similarity
            problem_similarity = self._calculate_similarity(
                new_problem.problem, 
                validated_existing.problem
            )
            
            # Check audience similarity
            audience_similarity = 1.0 if (
                self._normalize_audience(new_problem.audience) == 
                self._normalize_audience(validated_existing.audience)
            ) else 0.0
            
            # Combined similarity
            combined_similarity = (problem_similarity * 0.7 + audience_similarity * 0.3)
            
            if combined_similarity >= self.similarity_threshold:
                self.logger.info(
                    f"Found duplicate problem (similarity: {combined_similarity:.2f}): "
                    f"{new_problem.problem[:50]}..."
                )
                return True
        
        return False
    
    def add_research_session(self, 
                           opportunities: List[Any], 
                           research_file: str,
                           sectors: Set[str]) -> Dict[str, Any]:
        """Add a new research session with validation and deduplication.
        
        Args:
            opportunities: List of market opportunities
            research_file: Name of the research file
            sectors: Set of sectors/audiences discovered
            
        Returns:
            Statistics about what was added
        """
        stats = {
            'problems_added': 0,
            'problems_rejected': 0,
            'problems_duplicate': 0,
            'sectors_added': 0,
            'validation_errors': []
        }
        
        # Update session count
        self.summary_data['total_research_sessions'] += 1
        self.summary_data['last_research'] = datetime.now(timezone.utc).isoformat()
        
        # Add research file if not already present
        if research_file not in self.summary_data['research_files']:
            self.summary_data['research_files'].append(research_file)
        
        # Process opportunities
        for opp in opportunities:
            # Create validated problem
            problem_data = {
                'problem': getattr(opp, 'problem_description', ''),
                'audience': getattr(opp, 'affected_audience', ''),
                'opportunity': getattr(opp, 'opportunity', ''),
                'market_size': self._standardize_market_size(getattr(opp, 'market_size', '')),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            validated_problem = ValidatedProblem.from_dict(problem_data)
            
            if not validated_problem:
                stats['problems_rejected'] += 1
                stats['validation_errors'].append(f"Missing required fields in opportunity")
                continue
            
            # Check for duplicates
            if self._is_duplicate_problem(validated_problem):
                stats['problems_duplicate'] += 1
                continue
            
            # Add to problems
            self.summary_data['all_problems_discovered'].append(validated_problem.to_dict())
            stats['problems_added'] += 1
        
        # Process sectors with normalization
        current_sectors = set(self.summary_data.get('all_sectors_discovered', []))
        original_count = len(current_sectors)
        
        for sector in sectors:
            normalized = self._normalize_audience(sector)
            current_sectors.add(normalized)
        
        stats['sectors_added'] = len(current_sectors) - original_count
        
        # Update summary data
        self.summary_data['all_sectors_discovered'] = sorted(list(current_sectors))
        self.summary_data['total_unique_sectors'] = len(current_sectors)
        self.summary_data['total_problems_discovered'] = len(
            self.summary_data['all_problems_discovered']
        )
        
        # Save updated summary
        self._save_summary()
        
        self.logger.info(
            f"Research session added: {stats['problems_added']} problems, "
            f"{stats['problems_duplicate']} duplicates, "
            f"{stats['problems_rejected']} rejected"
        )
        
        return stats
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_sessions': self.summary_data['total_research_sessions'],
            'total_problems': self.summary_data['total_problems_discovered'],
            'unique_sectors': self.summary_data['total_unique_sectors'],
            'last_research': self.summary_data.get('last_research', 'Never')
        }
    
    def clean_existing_data(self) -> Dict[str, Any]:
        """Clean existing summary data (remove empty, deduplicate, etc.)."""
        stats = {
            'problems_removed': 0,
            'problems_deduplicated': 0,
            'sectors_deduplicated': 0
        }
        
        # Clean problems
        cleaned_problems = []
        seen_problems = set()
        
        for problem_data in self.summary_data.get('all_problems_discovered', []):
            # Validate
            validated = ValidatedProblem.from_dict(problem_data)
            if not validated:
                stats['problems_removed'] += 1
                continue
            
            # Create unique key for deduplication
            key = f"{validated.problem[:50]}_{self._normalize_audience(validated.audience)}"
            if key in seen_problems:
                stats['problems_deduplicated'] += 1
                continue
            
            seen_problems.add(key)
            
            # Standardize fields
            validated.audience = self._normalize_audience(validated.audience)
            validated.market_size = self._standardize_market_size(validated.market_size)
            
            cleaned_problems.append(validated.to_dict())
        
        # Clean sectors
        original_sectors = len(self.summary_data.get('all_sectors_discovered', []))
        normalized_sectors = set()
        
        for sector in self.summary_data.get('all_sectors_discovered', []):
            normalized = self._normalize_audience(sector)
            normalized_sectors.add(normalized)
        
        stats['sectors_deduplicated'] = original_sectors - len(normalized_sectors)
        
        # Update summary
        self.summary_data['all_problems_discovered'] = cleaned_problems
        self.summary_data['all_sectors_discovered'] = sorted(list(normalized_sectors))
        self.summary_data['total_unique_sectors'] = len(normalized_sectors)
        self.summary_data['total_problems_discovered'] = len(cleaned_problems)
        
        # Save cleaned data
        self._save_summary()
        
        return stats