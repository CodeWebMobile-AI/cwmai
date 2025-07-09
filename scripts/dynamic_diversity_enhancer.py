"""
Dynamic Diversity Enhancement System

Automatically discovers domains and patterns from the population itself
rather than using predefined categories.
"""

import re
import random
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
from datetime import datetime
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


class DynamicDiversityEnhancer:
    """Enhances diversity by dynamically discovering domains from the population."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Dynamic domain discovery
        self.discovered_domains = {}  # domain_id -> keywords
        self.domain_signatures = {}   # domain_id -> signature
        self.improvement_domains = defaultdict(list)  # improvement_id -> domains
        
        # Pattern tracking
        self.keyword_frequency = Counter()
        self.bigram_frequency = Counter()
        self.concept_clusters = defaultdict(set)
        
        # Diversity metrics
        self.similarity_threshold = 0.7
        self.min_domain_size = 3  # Minimum keywords to form a domain
        
        # NLP tools
        self._init_nlp_tools()
        
        # Historical tracking
        self.generation_history = []
        self.domain_evolution = defaultdict(list)
        
    def _init_nlp_tools(self):
        """Initialize NLP tools for text processing."""
        try:
            # Download required NLTK data if not present
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            self.stemmer = PorterStemmer()
            
            # Add technical stop words
            self.tech_stop_words = {
                'system', 'method', 'function', 'class', 'module', 'implement',
                'create', 'add', 'use', 'make', 'get', 'set', 'update', 'new',
                'data', 'process', 'service', 'application', 'software'
            }
            self.stop_words.update(self.tech_stop_words)
            
        except Exception as e:
            self.logger.warning(f"NLP tools initialization partial: {e}")
            self.stop_words = set()
            self.stemmer = None
            
    def discover_domains(self, population: List[Dict]) -> Dict[str, Set[str]]:
        """Dynamically discover domains from the current population."""
        # Extract all text from improvements
        all_text = []
        for imp in population:
            text = f"{imp.get('name', '')} {imp.get('type', '')} {imp.get('description', '')} {imp.get('approach', '')}"
            all_text.append(text.lower())
            
        # Extract keywords and concepts
        keyword_groups = defaultdict(set)
        
        for idx, text in enumerate(all_text):
            keywords = self._extract_keywords(text)
            
            # Group by significant keywords
            for keyword in keywords:
                if len(keyword) > 3:  # Skip very short words
                    keyword_groups[keyword].add(idx)
                    
        # Find keyword clusters (potential domains)
        domains = {}
        domain_id = 0
        
        # Use co-occurrence to find domains
        for keyword, imp_indices in keyword_groups.items():
            if len(imp_indices) >= 2:  # Keyword appears in multiple improvements
                # Find related keywords
                related_keywords = set([keyword])
                
                for idx in imp_indices:
                    text = all_text[idx]
                    text_keywords = self._extract_keywords(text)
                    
                    # Find keywords that co-occur
                    for other_keyword in text_keywords:
                        if other_keyword != keyword:
                            # Check if this keyword co-occurs frequently
                            co_occurrence = len(keyword_groups[other_keyword].intersection(imp_indices))
                            if co_occurrence >= len(imp_indices) * 0.5:  # 50% co-occurrence
                                related_keywords.add(other_keyword)
                                
                # Create domain if we have enough related keywords
                if len(related_keywords) >= self.min_domain_size:
                    domain_name = f"domain_{domain_id}"
                    domains[domain_name] = related_keywords
                    domain_id += 1
                    
        # Merge overlapping domains
        merged_domains = self._merge_similar_domains(domains)
        
        # Update discovered domains
        self.discovered_domains.update(merged_domains)
        
        return merged_domains
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove punctuation and tokenize
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
            
        keywords = []
        
        for token in tokens:
            # Skip short tokens and numbers
            if len(token) <= 2 or token.isdigit():
                continue
                
            # Skip stop words
            if token.lower() in self.stop_words:
                continue
                
            # Stem the word if possible
            if self.stemmer:
                try:
                    stemmed = self.stemmer.stem(token.lower())
                    keywords.append(stemmed)
                except:
                    keywords.append(token.lower())
            else:
                keywords.append(token.lower())
                
        return keywords
        
    def _merge_similar_domains(self, domains: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Merge domains that have significant overlap."""
        merged = {}
        used_domains = set()
        
        domain_items = list(domains.items())
        
        for i, (domain1_name, domain1_keywords) in enumerate(domain_items):
            if domain1_name in used_domains:
                continue
                
            merged_keywords = domain1_keywords.copy()
            
            for j, (domain2_name, domain2_keywords) in enumerate(domain_items[i+1:], i+1):
                if domain2_name in used_domains:
                    continue
                    
                # Calculate overlap
                overlap = len(domain1_keywords.intersection(domain2_keywords))
                total = len(domain1_keywords.union(domain2_keywords))
                
                if total > 0 and overlap / total > 0.3:  # 30% overlap
                    merged_keywords.update(domain2_keywords)
                    used_domains.add(domain2_name)
                    
            # Name domain based on most common keyword
            if merged_keywords:
                most_common = Counter(merged_keywords).most_common(1)[0][0]
                domain_name = f"{most_common}_domain"
                merged[domain_name] = merged_keywords
                used_domains.add(domain1_name)
                
        return merged
        
    def calculate_diversity_score(self, population: List[Dict]) -> float:
        """Calculate diversity score based on discovered domains."""
        if len(population) < 2:
            return 0.0
            
        # Discover domains if not already done
        if not self.discovered_domains:
            self.discover_domains(population)
            
        # Classify improvements into domains
        domain_assignments = self._classify_improvements(population)
        
        # Calculate domain diversity
        domain_counts = Counter(domain_assignments.values())
        num_active_domains = len(domain_counts)
        max_possible_domains = min(len(population), len(self.discovered_domains) + 1)  # +1 for 'general'
        
        domain_diversity = num_active_domains / max_possible_domains if max_possible_domains > 0 else 0
        
        # Calculate approach diversity
        approaches = [str(imp.get('approach', '')) for imp in population]
        unique_approaches = len(set(approaches))
        approach_diversity = unique_approaches / len(approaches) if approaches else 0
        
        # Calculate semantic diversity (how different the improvements are)
        semantic_diversity = self._calculate_semantic_diversity(population)
        
        # Combined score
        diversity = (
            domain_diversity * 0.4 +
            approach_diversity * 0.3 +
            semantic_diversity * 0.3
        )
        
        return min(1.0, diversity)
        
    def _classify_improvements(self, population: List[Dict]) -> Dict[str, str]:
        """Classify each improvement into discovered domains."""
        classifications = {}
        
        for imp in population:
            imp_id = imp.get('id', imp.get('name', str(hash(str(imp)))))
            text = f"{imp.get('name', '')} {imp.get('type', '')} {imp.get('description', '')} {imp.get('approach', '')}"
            keywords = set(self._extract_keywords(text.lower()))
            
            best_domain = 'general'
            best_score = 0
            
            for domain_name, domain_keywords in self.discovered_domains.items():
                # Calculate overlap score
                overlap = len(keywords.intersection(domain_keywords))
                if overlap > best_score:
                    best_score = overlap
                    best_domain = domain_name
                    
            classifications[imp_id] = best_domain
            
        return classifications
        
    def _calculate_semantic_diversity(self, population: List[Dict]) -> float:
        """Calculate how semantically different the improvements are."""
        if len(population) < 2:
            return 0.0
            
        signatures = []
        for imp in population:
            sig = self._create_semantic_signature(imp)
            signatures.append(sig)
            
        # Calculate pairwise distances
        total_distance = 0
        comparisons = 0
        
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                distance = 1.0 - self._calculate_similarity(signatures[i], signatures[j])
                total_distance += distance
                comparisons += 1
                
        return total_distance / comparisons if comparisons > 0 else 0.0
        
    def _create_semantic_signature(self, improvement: Dict) -> str:
        """Create a semantic signature for comparison."""
        # Extract all meaningful text
        components = [
            improvement.get('type', ''),
            improvement.get('approach', ''),
            ' '.join(improvement.get('metrics', [])),
            ' '.join(str(ip) for ip in improvement.get('integration_points', []))
        ]
        
        # Extract keywords from all components
        all_keywords = []
        for component in components:
            keywords = self._extract_keywords(str(component).lower())
            all_keywords.extend(keywords)
            
        return ' '.join(all_keywords)
        
    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate Jaccard similarity between signatures."""
        words1 = set(sig1.split())
        words2 = set(sig2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def enforce_diversity(self, population: List[Dict], target_diversity: float = 0.6) -> List[Dict]:
        """Enforce diversity by modifying population."""
        current_diversity = self.calculate_diversity_score(population)
        
        if current_diversity >= target_diversity:
            return population
            
        self.logger.info(f"Enforcing diversity: current={current_diversity:.3f}, target={target_diversity}")
        
        # Classify improvements
        classifications = self._classify_improvements(population)
        domain_counts = Counter(classifications.values())
        
        # Find over-represented domains
        avg_per_domain = len(population) / max(len(domain_counts), 1)
        over_represented = [domain for domain, count in domain_counts.items() 
                           if count > avg_per_domain * 1.5]
        
        # Replace some over-represented improvements
        diverse_population = []
        domain_usage = defaultdict(int)
        max_per_domain = int(avg_per_domain * 1.2)  # Allow 20% over average
        
        for imp in population:
            imp_id = imp.get('id', imp.get('name', str(hash(str(imp)))))
            domain = classifications.get(imp_id, 'general')
            
            if domain_usage[domain] < max_per_domain:
                diverse_population.append(imp)
                domain_usage[domain] += 1
            else:
                # Create diverse replacement
                new_imp = self._create_diverse_improvement(imp, avoided_domains=over_represented)
                diverse_population.append(new_imp)
                
        return diverse_population
        
    def _create_diverse_improvement(self, original: Dict, avoided_domains: List[str]) -> Dict:
        """Create improvement in underrepresented domain."""
        # Find underrepresented concepts from population history
        underrepresented_keywords = self._find_underrepresented_concepts()
        
        # Generate new improvement focusing on these concepts
        timestamp = datetime.now().isoformat()
        random_id = hashlib.md5(f"diverse_{timestamp}".encode()).hexdigest()[:8]
        
        # Use original as template but modify key aspects
        new_improvement = original.copy()
        new_improvement['id'] = random_id
        new_improvement['name'] = f"Diverse_{original.get('name', 'Solution')}_{random_id[:4]}"
        new_improvement['diversity_enforced'] = True
        new_improvement['target_keywords'] = list(underrepresented_keywords)[:5]
        
        # Modify description to include underrepresented concepts
        if underrepresented_keywords:
            concepts = ', '.join(list(underrepresented_keywords)[:3])
            new_improvement['description'] = f"Enhanced solution incorporating {concepts}"
            
        return new_improvement
        
    def _find_underrepresented_concepts(self) -> Set[str]:
        """Find concepts that are underrepresented in current population."""
        # This would ideally look at historical data
        # For now, return some diverse technical concepts
        diverse_concepts = {
            'resilience', 'scalability', 'federation', 'consensus',
            'replication', 'sharding', 'versioning', 'migration',
            'rollback', 'observability', 'tracing', 'profiling',
            'debugging', 'simulation', 'emulation', 'virtualization',
            'sandboxing', 'isolation', 'orchestration', 'choreography'
        }
        
        # Remove concepts already well-represented
        for domain_keywords in self.discovered_domains.values():
            diverse_concepts -= domain_keywords
            
        return diverse_concepts
        
    def calculate_novelty_bonus(self, improvement: Dict) -> float:
        """Calculate novelty bonus for unique improvements."""
        # Extract improvement signature
        signature = self._create_semantic_signature(improvement)
        keywords = set(signature.split())
        
        # Calculate how many new concepts this introduces
        all_existing_keywords = set()
        for domain_keywords in self.discovered_domains.values():
            all_existing_keywords.update(domain_keywords)
            
        new_keywords = keywords - all_existing_keywords
        novelty_ratio = len(new_keywords) / len(keywords) if keywords else 0
        
        # Bonus for introducing new concepts
        novelty_bonus = novelty_ratio * 0.3
        
        # Extra bonus for completely new domain
        imp_id = improvement.get('id', improvement.get('name', str(hash(str(improvement)))))
        classifications = self._classify_improvements([improvement])
        if classifications.get(imp_id) == 'general' and novelty_ratio > 0.5:
            novelty_bonus += 0.2
            
        return min(novelty_bonus, 0.5)
        
    def get_diversity_report(self, population: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive diversity report."""
        # Ensure domains are discovered
        if not self.discovered_domains:
            self.discover_domains(population)
            
        # Classify improvements
        classifications = self._classify_improvements(population)
        domain_counts = Counter(classifications.values())
        
        # Find most common concepts
        all_keywords = []
        for imp in population:
            text = f"{imp.get('name', '')} {imp.get('type', '')} {imp.get('description', '')}"
            keywords = self._extract_keywords(text.lower())
            all_keywords.extend(keywords)
            
        keyword_counts = Counter(all_keywords)
        
        return {
            'diversity_score': self.calculate_diversity_score(population),
            'discovered_domains': len(self.discovered_domains),
            'domain_distribution': dict(domain_counts),
            'most_common_domain': domain_counts.most_common(1)[0] if domain_counts else ('none', 0),
            'unique_improvements': len(set(imp.get('name', '') for imp in population)),
            'top_concepts': keyword_counts.most_common(10),
            'recommendations': self._get_diversity_recommendations(domain_counts, keyword_counts)
        }
        
    def _get_diversity_recommendations(self, domain_counts: Counter, keyword_counts: Counter) -> List[str]:
        """Generate actionable diversity recommendations."""
        recommendations = []
        
        # Check domain concentration
        total = sum(domain_counts.values())
        for domain, count in domain_counts.items():
            if total > 0 and count / total > 0.3:
                recommendations.append(f"Reduce focus on {domain} (currently {count/total:.1%})")
                
        # Suggest underexplored concepts
        underrepresented = self._find_underrepresented_concepts()
        if underrepresented:
            concepts = list(underrepresented)[:3]
            recommendations.append(f"Explore concepts: {', '.join(concepts)}")
            
        # Check for low diversity
        if len(domain_counts) < 3:
            recommendations.append("Increase domain diversity - too few distinct domains")
            
        return recommendations