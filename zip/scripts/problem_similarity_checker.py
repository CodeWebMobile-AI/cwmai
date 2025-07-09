"""
Problem Similarity Checker

Advanced similarity detection for market problems to prevent
duplicate research and identify related opportunities.
"""

import re
from typing import List, Tuple, Set, Dict, Any, Optional
from difflib import SequenceMatcher
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from similarity_config import SimilarityConfig, default_config
except ImportError:
    # Fallback if config not available
    default_config = None


class ProblemSimilarityChecker:
    """Check similarity between market problems using multiple techniques."""
    
    def __init__(self, config: Optional[Dict] = None, domain: str = 'general'):
        """Initialize with configuration.
        
        Args:
            config: Optional configuration dictionary. If None, uses default config
            domain: Domain to use for default configuration ('general', 'business', 'technical')
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config:
            self.config = config
        elif default_config:
            self.config = default_config.get_config_for_domain(domain)
        else:
            # Fallback minimal configuration
            self.config = self._get_minimal_config()
        
        # Extract configuration values
        self.threshold = self.config.get('threshold', 0.7)
        self.weights = self.config.get('weights', {
            'jaccard': 0.5,
            'sequence': 0.35,
            'phrase': 0.15
        })
        self.stop_words = self.config.get('stop_words', set())
        self.synonyms = self.config.get('synonyms', {})
        self.phrase_patterns = self.config.get('phrase_patterns', {})
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            self.logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= weight_sum
    
    def _get_minimal_config(self) -> Dict:
        """Get minimal fallback configuration."""
        return {
            'threshold': 0.7,
            'weights': {
                'jaccard': 0.5,
                'sequence': 0.35,
                'phrase': 0.15
            },
            'stop_words': {
                'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are',
                'been', 'by', 'for', 'from', 'has', 'have', 'in', 'of', 'or', 'to',
                'was', 'will', 'with', 'their', 'they', 'this', 'that', 'these'
            },
            'synonyms': {},
            'phrase_patterns': {
                'problem_indicators': [r'problem', r'issue', r'difficulty', r'challenge']
            }
        }
    
    def update_config(self, updates: Dict):
        """Update configuration dynamically."""
        if 'threshold' in updates:
            self.threshold = updates['threshold']
        
        if 'weights' in updates:
            self.weights.update(updates['weights'])
            # Re-normalize weights
            weight_sum = sum(self.weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                for key in self.weights:
                    self.weights[key] /= weight_sum
        
        if 'stop_words' in updates:
            if isinstance(updates['stop_words'], set):
                self.stop_words = updates['stop_words']
            else:
                self.stop_words.update(updates['stop_words'])
        
        if 'synonyms' in updates:
            for key, values in updates['synonyms'].items():
                if key in self.synonyms:
                    self.synonyms[key].extend(values)
                    self.synonyms[key] = list(set(self.synonyms[key]))
                else:
                    self.synonyms[key] = values
        
        if 'phrase_patterns' in updates:
            self.phrase_patterns.update(updates['phrase_patterns'])
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for similarity comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Split into words and remove stop words
        words = [w for w in text.split() if w and w not in self.stop_words]
        
        return words
    
    def _expand_with_synonyms(self, words: List[str]) -> Set[str]:
        """Expand word list with synonyms from configuration."""
        expanded = set(words)
        
        # Use configured synonyms
        for word in words:
            word_lower = word.lower()
            
            # Check if word is a key
            if word_lower in self.synonyms:
                expanded.update(self.synonyms[word_lower])
            
            # Check if word is a synonym value
            for key, synonym_list in self.synonyms.items():
                if word_lower in [s.lower() for s in synonym_list]:
                    expanded.add(key)
                    expanded.update(synonym_list)
        
        return {w.lower() for w in expanded}  # Normalize to lowercase
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two problem descriptions.
        
        Uses multiple techniques:
        1. Jaccard similarity on key words
        2. Sequence matching for phrase similarity
        3. Synonym expansion for semantic similarity
        
        Returns:
            Similarity score between 0 and 1
        """
        # Preprocess texts
        words1 = self._preprocess_text(text1)
        words2 = self._preprocess_text(text2)
        
        # Expand with synonyms
        expanded1 = self._expand_with_synonyms(words1)
        expanded2 = self._expand_with_synonyms(words2)
        
        # Calculate Jaccard similarity
        intersection = len(expanded1.intersection(expanded2))
        union = len(expanded1.union(expanded2))
        jaccard_score = intersection / union if union > 0 else 0
        
        # Calculate sequence similarity
        sequence_score = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Calculate key phrase similarity
        phrases1 = self._extract_key_phrases(text1)
        phrases2 = self._extract_key_phrases(text2)
        phrase_score = self._calculate_phrase_similarity(phrases1, phrases2)
        
        # Weighted combination using configurable weights
        final_score = (
            jaccard_score * self.weights.get('jaccard', 0.5) +
            sequence_score * self.weights.get('sequence', 0.35) +
            phrase_score * self.weights.get('phrase', 0.15)
        )
        
        return final_score
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text using configured patterns."""
        phrases = []
        text_lower = text.lower()
        
        # Use configured patterns
        for category, patterns in self.phrase_patterns.items():
            for pattern in patterns:
                # Add capture group if not present
                if '(' not in pattern:
                    pattern = pattern + r'.*?(\w+ \w+)'
                
                try:
                    matches = re.findall(pattern, text_lower)
                    phrases.extend(matches)
                except re.error:
                    self.logger.warning(f"Invalid regex pattern: {pattern}")
        
        # If no patterns configured, do basic extraction
        if not phrases and not self.phrase_patterns:
            # Extract bigrams after removing stop words
            words = self._preprocess_text(text)
            for i in range(len(words) - 1):
                if words[i] not in self.stop_words and words[i+1] not in self.stop_words:
                    phrases.append(f"{words[i]} {words[i+1]}")
        
        return phrases
    
    def _calculate_phrase_similarity(self, phrases1: List[str], phrases2: List[str]) -> float:
        """Calculate similarity between phrase lists."""
        if not phrases1 or not phrases2:
            return 0.0
        
        max_scores = []
        for p1 in phrases1:
            scores = [SequenceMatcher(None, p1, p2).ratio() for p2 in phrases2]
            if scores:
                max_scores.append(max(scores))
        
        return sum(max_scores) / len(max_scores) if max_scores else 0.0
    
    def find_similar_problems(self, 
                            new_problem: str, 
                            existing_problems: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        """Find similar problems from existing list.
        
        Args:
            new_problem: New problem description
            existing_problems: List of existing problems with 'problem' field
            
        Returns:
            List of (index, similarity_score) tuples for similar problems
        """
        similar = []
        
        for i, existing in enumerate(existing_problems):
            if 'problem' not in existing:
                continue
            
            score = self.calculate_similarity(new_problem, existing['problem'])
            
            if score >= self.threshold:
                similar.append((i, score))
        
        # Sort by similarity score (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar
    
    def is_duplicate(self, 
                    problem1: Dict[str, Any], 
                    problem2: Dict[str, Any]) -> bool:
        """Check if two problems are duplicates considering multiple factors.
        
        Args:
            problem1: First problem with 'problem' and 'audience' fields
            problem2: Second problem with 'problem' and 'audience' fields
            
        Returns:
            True if problems are duplicates
        """
        # Check problem similarity
        problem_score = self.calculate_similarity(
            problem1.get('problem', ''),
            problem2.get('problem', '')
        )
        
        # Check audience match
        audience1 = problem1.get('audience', '').lower()
        audience2 = problem2.get('audience', '').lower()
        
        # Exact audience match boosts similarity
        audience_boost = 0.2 if audience1 == audience2 else 0.0
        
        # Similar audience gives smaller boost
        if audience1 != audience2:
            audience_score = self.calculate_similarity(audience1, audience2)
            audience_boost = audience_score * 0.1
        
        # Combined score
        total_score = problem_score + audience_boost
        
        return total_score >= self.threshold
    
    def group_similar_problems(self, 
                             problems: List[Dict[str, Any]]) -> List[List[int]]:
        """Group similar problems together.
        
        Args:
            problems: List of problems with 'problem' field
            
        Returns:
            List of groups, where each group is a list of problem indices
        """
        groups = []
        assigned = set()
        
        for i, problem in enumerate(problems):
            if i in assigned:
                continue
            
            # Start new group
            group = [i]
            assigned.add(i)
            
            # Find similar problems
            for j in range(i + 1, len(problems)):
                if j in assigned:
                    continue
                
                if self.is_duplicate(problem, problems[j]):
                    group.append(j)
                    assigned.add(j)
            
            groups.append(group)
        
        return groups
    
    def get_similarity_report(self, 
                            problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a similarity report for a list of problems.
        
        Args:
            problems: List of problems to analyze
            
        Returns:
            Report with similarity statistics
        """
        groups = self.group_similar_problems(problems)
        
        # Find most common problem themes
        all_words = []
        for problem in problems:
            words = self._preprocess_text(problem.get('problem', ''))
            all_words.extend(words)
        
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        common_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report = {
            'total_problems': len(problems),
            'unique_groups': len(groups),
            'duplicate_count': len(problems) - len(groups),
            'duplication_rate': (len(problems) - len(groups)) / len(problems) * 100 if problems else 0,
            'largest_group_size': max(len(g) for g in groups) if groups else 0,
            'common_themes': common_themes,
            'similarity_threshold': self.threshold
        }
        
        return report