"""
System-Aligned Dynamic Diversity Enhancer

Learns and adapts diversity strategies from the actual CWMAI system patterns
without any hard-coded domains or categories.
"""

import re
import json
import random
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
from datetime import datetime
import hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


class SystemAlignedDiversityEnhancer:
    """Diversity enhancer that learns from system patterns dynamically."""
    
    def __init__(self, system_state_path: str = "system_state.json"):
        self.logger = logging.getLogger(__name__)
        self.system_state_path = system_state_path
        
        # Dynamic learning components
        self.learned_patterns = defaultdict(float)
        self.concept_embeddings = {}
        self.cluster_centers = []
        self.improvement_history = []
        
        # Adaptive parameters
        self.similarity_threshold = 0.7
        self.learning_rate = 0.1
        self.exploration_bonus = 0.3
        
        # Text processing
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Load system knowledge
        self._load_system_knowledge()
        
    def _load_system_knowledge(self):
        """Load knowledge from system state and existing improvements."""
        try:
            # Load system state
            if Path(self.system_state_path).exists():
                with open(self.system_state_path, 'r') as f:
                    system_state = json.load(f)
                    
                # Learn from existing improvements
                if 'evolutionary_improvements' in system_state:
                    for imp in system_state['evolutionary_improvements']:
                        self.improvement_history.append(imp)
                        
                # Learn from tool patterns
                if 'generated_tools' in system_state:
                    for tool in system_state['generated_tools']:
                        self._learn_from_pattern(tool)
                        
            # Scan for existing improvement files
            self._scan_improvement_files()
            
        except Exception as e:
            self.logger.warning(f"Could not load system knowledge: {e}")
            
    def _scan_improvement_files(self):
        """Scan system for existing improvements to learn patterns."""
        improvement_dirs = [
            Path("scripts/improvements"),
            Path("scripts/custom_tools"),
            Path("scripts/workflows")
        ]
        
        for dir_path in improvement_dirs:
            if dir_path.exists():
                for file_path in dir_path.glob("*.py"):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Extract patterns from file content
                            self._learn_from_code(content, file_path.stem)
                    except:
                        pass
                        
    def _learn_from_pattern(self, pattern: Dict):
        """Learn from a system pattern (tool, improvement, etc)."""
        # Extract text representation
        text = f"{pattern.get('name', '')} {pattern.get('type', '')} {pattern.get('description', '')}"
        
        # Update pattern frequencies
        tokens = self._tokenize(text)
        for token in tokens:
            self.learned_patterns[token] += self.learning_rate
            
    def _learn_from_code(self, code: str, name: str):
        """Learn patterns from actual code implementations."""
        # Extract docstrings and comments
        docstring_pattern = r'"""(.*?)"""'
        comment_pattern = r'#\s*(.+)'
        
        docstrings = re.findall(docstring_pattern, code, re.DOTALL)
        comments = re.findall(comment_pattern, code)
        
        # Extract function/class names
        function_pattern = r'def\s+(\w+)'
        class_pattern = r'class\s+(\w+)'
        
        functions = re.findall(function_pattern, code)
        classes = re.findall(class_pattern, code)
        
        # Learn from all extracted text
        all_text = ' '.join(docstrings + comments + functions + classes)
        if all_text:
            self._learn_from_pattern({'name': name, 'description': all_text})
            
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into meaningful units."""
        # Remove special characters but keep technical terms
        text = re.sub(r'[^\w\s_-]', ' ', text.lower())
        
        # Split on whitespace and filter
        tokens = text.split()
        
        # Filter out very short tokens and pure numbers
        filtered = []
        for token in tokens:
            if len(token) > 2 and not token.isdigit():
                # Split camelCase and snake_case
                parts = re.split(r'[_-]|(?=[A-Z])', token)
                filtered.extend([p.lower() for p in parts if len(p) > 2])
                
        return filtered
        
    def calculate_diversity_score(self, population: List[Dict]) -> float:
        """Calculate diversity using learned system patterns."""
        if len(population) < 2:
            return 0.0
            
        # Create text representations
        texts = []
        for imp in population:
            text = self._create_text_representation(imp)
            texts.append(text)
            
        try:
            # Compute TF-IDF vectors
            if len(texts) >= 2:
                vectors = self.vectorizer.fit_transform(texts)
                
                # Calculate pairwise similarities
                similarities = cosine_similarity(vectors)
                
                # Diversity is inverse of average similarity
                avg_similarity = (similarities.sum() - len(texts)) / (len(texts) * (len(texts) - 1))
                diversity = 1.0 - avg_similarity
                
                return max(0.0, min(1.0, diversity))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Diversity calculation fallback: {e}")
            # Fallback to simple uniqueness measure
            unique_names = len(set(imp.get('name', str(i)) for i, imp in enumerate(population)))
            return unique_names / len(population)
            
    def _create_text_representation(self, improvement: Dict) -> str:
        """Create comprehensive text representation of improvement."""
        components = []
        
        # Add all text fields
        for field in ['name', 'type', 'description', 'approach']:
            if field in improvement:
                components.append(str(improvement[field]))
                
        # Add metrics and integration points
        if 'metrics' in improvement:
            components.extend([str(m) for m in improvement['metrics']])
            
        if 'integration_points' in improvement:
            components.extend([str(ip) for ip in improvement['integration_points']])
            
        # Add any code snippets
        if 'code' in improvement:
            # Extract meaningful parts from code
            code = improvement['code']
            # Get function names
            functions = re.findall(r'def\s+(\w+)', code)
            components.extend(functions)
            
        return ' '.join(components)
        
    def discover_clusters(self, population: List[Dict]) -> Dict[str, List[int]]:
        """Discover natural clusters in the population."""
        if len(population) < 2:
            return {'cluster_0': list(range(len(population)))}
            
        texts = [self._create_text_representation(imp) for imp in population]
        
        try:
            # Create vectors
            vectors = self.vectorizer.fit_transform(texts)
            
            # Use DBSCAN for clustering (doesn't require predefined number of clusters)
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            labels = clustering.fit_predict(vectors.toarray())
            
            # Group by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                cluster_name = f'cluster_{label}' if label >= 0 else 'outliers'
                clusters[cluster_name].append(idx)
                
            return dict(clusters)
            
        except Exception as e:
            self.logger.debug(f"Clustering fallback: {e}")
            # Fallback: each improvement is its own cluster
            return {f'cluster_{i}': [i] for i in range(len(population))}
            
    def enforce_diversity(self, population: List[Dict], target_diversity: float = 0.6) -> List[Dict]:
        """Enforce diversity using system-learned patterns."""
        current_diversity = self.calculate_diversity_score(population)
        
        if current_diversity >= target_diversity:
            return population
            
        self.logger.info(f"Enforcing diversity: current={current_diversity:.3f}, target={target_diversity}")
        
        # Discover clusters
        clusters = self.discover_clusters(population)
        
        # Find over-represented clusters
        cluster_sizes = {name: len(indices) for name, indices in clusters.items()}
        avg_size = len(population) / max(len(clusters), 1)
        
        # Identify which improvements to diversify
        diverse_population = []
        cluster_usage = defaultdict(int)
        max_per_cluster = int(avg_size * 1.5)
        
        for idx, imp in enumerate(population):
            # Find which cluster this improvement belongs to
            imp_cluster = None
            for cluster_name, indices in clusters.items():
                if idx in indices:
                    imp_cluster = cluster_name
                    break
                    
            if imp_cluster and cluster_usage[imp_cluster] < max_per_cluster:
                diverse_population.append(imp)
                cluster_usage[imp_cluster] += 1
            else:
                # Create diverse variant
                new_imp = self._create_diverse_variant(imp, population)
                diverse_population.append(new_imp)
                
        return diverse_population
        
    def _create_diverse_variant(self, original: Dict, population: List[Dict]) -> Dict:
        """Create a diverse variant using system knowledge."""
        # Analyze what's common in current population
        all_tokens = []
        for imp in population:
            text = self._create_text_representation(imp)
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            
        token_freq = Counter(all_tokens)
        common_tokens = set([token for token, freq in token_freq.most_common(10)])
        
        # Find less common patterns from system knowledge
        uncommon_patterns = []
        for pattern, weight in self.learned_patterns.items():
            if pattern not in common_tokens and weight > 0:
                uncommon_patterns.append((pattern, weight))
                
        # Sort by weight and select diverse concepts
        uncommon_patterns.sort(key=lambda x: x[1], reverse=True)
        selected_patterns = [p[0] for p in uncommon_patterns[:5]]
        
        # Create variant
        timestamp = datetime.now().isoformat()
        variant_id = hashlib.md5(f"diverse_{timestamp}".encode()).hexdigest()[:8]
        
        variant = original.copy()
        variant['id'] = variant_id
        variant['name'] = f"Enhanced_{original.get('name', 'Solution')}_{variant_id[:4]}"
        variant['diversity_enforced'] = True
        
        # Enhance with uncommon patterns
        if selected_patterns:
            variant['enhanced_concepts'] = selected_patterns
            concepts_text = ' '.join(selected_patterns)
            variant['description'] = f"{original.get('description', '')} Enhanced with: {concepts_text}"
            
        return variant
        
    def calculate_novelty_bonus(self, improvement: Dict) -> float:
        """Calculate novelty based on system knowledge."""
        text = self._create_text_representation(improvement)
        tokens = set(self._tokenize(text))
        
        # Calculate how many tokens are new or rare
        novelty_score = 0.0
        known_tokens = 0
        
        for token in tokens:
            if token in self.learned_patterns:
                # Lower weight means more novel
                weight = self.learned_patterns[token]
                novelty_score += 1.0 / (1.0 + weight)
                known_tokens += 1
            else:
                # Completely new token
                novelty_score += 1.0
                
        # Normalize
        if len(tokens) > 0:
            novelty_ratio = novelty_score / len(tokens)
        else:
            novelty_ratio = 0.0
            
        # Apply exploration bonus
        bonus = novelty_ratio * self.exploration_bonus
        
        # Extra bonus for introducing many new concepts
        if len(tokens) > 0 and known_tokens / len(tokens) < 0.3:
            bonus += 0.1
            
        return min(bonus, 0.5)
        
    def update_learning(self, improvement: Dict, success: bool):
        """Update learned patterns based on improvement success."""
        text = self._create_text_representation(improvement)
        tokens = self._tokenize(text)
        
        # Adjust learning based on success
        adjustment = self.learning_rate if success else -self.learning_rate * 0.5
        
        for token in tokens:
            self.learned_patterns[token] += adjustment
            # Keep weights positive
            if self.learned_patterns[token] < 0:
                self.learned_patterns[token] = 0
                
    def get_diversity_report(self, population: List[Dict]) -> Dict[str, Any]:
        """Generate diversity report aligned with system patterns."""
        diversity_score = self.calculate_diversity_score(population)
        clusters = self.discover_clusters(population)
        
        # Analyze pattern distribution
        all_tokens = []
        for imp in population:
            text = self._create_text_representation(imp)
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            
        token_freq = Counter(all_tokens)
        
        # Find patterns from system knowledge not in current population
        missing_patterns = []
        for pattern, weight in self.learned_patterns.items():
            if weight > 0.5 and pattern not in token_freq:
                missing_patterns.append(pattern)
                
        return {
            'diversity_score': diversity_score,
            'num_clusters': len(clusters),
            'cluster_sizes': {name: len(indices) for name, indices in clusters.items()},
            'unique_patterns': len(set(all_tokens)),
            'most_common_patterns': token_freq.most_common(10),
            'missing_system_patterns': missing_patterns[:10],
            'recommendations': self._generate_recommendations(clusters, token_freq, missing_patterns)
        }
        
    def _generate_recommendations(self, clusters: Dict, token_freq: Counter, missing_patterns: List[str]) -> List[str]:
        """Generate recommendations based on system alignment."""
        recommendations = []
        
        # Check cluster balance
        cluster_sizes = [len(indices) for indices in clusters.values()]
        if cluster_sizes:
            max_size = max(cluster_sizes)
            avg_size = sum(cluster_sizes) / len(cluster_sizes)
            if max_size > avg_size * 2:
                recommendations.append(f"Rebalance clusters - largest is {max_size/avg_size:.1f}x average")
                
        # Check for overused patterns
        if token_freq:
            most_common = token_freq.most_common(1)[0]
            if most_common[1] > len(token_freq) * 0.2:
                recommendations.append(f"Reduce use of '{most_common[0]}' pattern")
                
        # Suggest missing patterns
        if missing_patterns:
            suggestions = missing_patterns[:3]
            recommendations.append(f"Explore patterns: {', '.join(suggestions)}")
            
        # Check overall diversity
        if len(clusters) < 3:
            recommendations.append("Increase solution diversity - too few distinct clusters")
            
        return recommendations