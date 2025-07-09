"""
Diversity Enhancement System

Implements smart mechanisms to ensure exploration beyond local optima
and prevent convergence on similar solution types.
"""

import re
import random
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import logging
from datetime import datetime
import hashlib


class DiversityEnhancer:
    """Enhances diversity in evolutionary population through multiple strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Track solution domains to prevent over-concentration
        self.domain_tracker = defaultdict(int)
        self.solution_signatures = set()
        self.generation_history = []
        
        # Domain keywords for classification
        self.domain_keywords = {
            'caching': ['cache', 'lru', 'ttl', 'memoiz', 'redis', 'memory'],
            'security': ['auth', 'encrypt', 'secure', 'protect', 'validate', 'sanitize'],
            'performance': ['optimize', 'fast', 'speed', 'efficient', 'parallel', 'async'],
            'monitoring': ['monitor', 'track', 'metric', 'log', 'observ', 'telemetry'],
            'api': ['api', 'rest', 'graphql', 'endpoint', 'route', 'http'],
            'database': ['db', 'sql', 'query', 'orm', 'migration', 'index'],
            'testing': ['test', 'unit', 'integration', 'mock', 'assert', 'coverage'],
            'ai_ml': ['ai', 'ml', 'neural', 'predict', 'train', 'model'],
            'workflow': ['workflow', 'pipeline', 'orchestrat', 'automat', 'flow'],
            'communication': ['message', 'queue', 'event', 'pubsub', 'webhook', 'socket'],
            'data_processing': ['transform', 'etl', 'process', 'parse', 'convert'],
            'ui_ux': ['ui', 'ux', 'interface', 'component', 'render', 'display'],
            'infrastructure': ['deploy', 'container', 'kubernetes', 'cloud', 'server'],
            'documentation': ['doc', 'comment', 'readme', 'swagger', 'openapi']
        }
        
        # Novelty rewards for unexplored domains
        self.domain_novelty_scores = defaultdict(lambda: 1.0)
        
        # Similarity threshold for considering solutions too similar
        self.similarity_threshold = 0.7
        
    def calculate_diversity_score(self, population: List[Dict]) -> float:
        """Calculate overall diversity score of population."""
        if len(population) < 2:
            return 0.0
            
        # Domain diversity
        domains = [self._classify_domain(imp) for imp in population]
        domain_diversity = len(set(domains)) / len(self.domain_keywords)
        
        # Name diversity (avoid similar names)
        names = [imp.get('name', '').lower() for imp in population]
        name_diversity = len(set(names)) / len(names) if names else 0
        
        # Approach diversity
        approaches = [str(imp.get('approach', '')) for imp in population]
        approach_diversity = len(set(approaches)) / len(approaches) if approaches else 0
        
        # Type diversity
        types = [imp.get('type', 'unknown') for imp in population]
        type_diversity = len(set(types)) / len(types) if types else 0
        
        # Combined diversity score
        diversity = (
            domain_diversity * 0.4 +  # Most important
            name_diversity * 0.2 +
            approach_diversity * 0.2 +
            type_diversity * 0.2
        )
        
        return min(1.0, diversity)
        
    def enforce_diversity(self, population: List[Dict], target_diversity: float = 0.6) -> List[Dict]:
        """Enforce diversity in population by modifying or replacing similar solutions."""
        current_diversity = self.calculate_diversity_score(population)
        
        if current_diversity >= target_diversity:
            return population
            
        self.logger.info(f"Enforcing diversity: current={current_diversity:.3f}, target={target_diversity}")
        
        # Identify over-represented domains
        domain_counts = Counter(self._classify_domain(imp) for imp in population)
        max_per_domain = max(2, len(population) // len(self.domain_keywords))
        
        # Replace over-represented improvements
        diverse_population = []
        domain_usage = defaultdict(int)
        
        for imp in population:
            domain = self._classify_domain(imp)
            if domain_usage[domain] < max_per_domain:
                diverse_population.append(imp)
                domain_usage[domain] += 1
            else:
                # Force a different domain
                new_imp = self._create_diverse_improvement(imp, excluded_domains=[domain])
                diverse_population.append(new_imp)
                
        return diverse_population
        
    def _classify_domain(self, improvement: Dict) -> str:
        """Classify improvement into a domain based on keywords."""
        text = f"{improvement.get('name', '')} {improvement.get('description', '')} {improvement.get('approach', '')}".lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            domain_scores[domain] = score
            
        # Return domain with highest score, or 'general' if no matches
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
                
        return 'general'
        
    def _create_diverse_improvement(self, original: Dict, excluded_domains: List[str]) -> Dict:
        """Create a new improvement in a different domain."""
        # Select an underrepresented domain
        available_domains = [d for d in self.domain_keywords.keys() if d not in excluded_domains]
        weights = [self.domain_novelty_scores[d] for d in available_domains]
        
        if not available_domains:
            available_domains = list(self.domain_keywords.keys())
            weights = [1.0] * len(available_domains)
            
        selected_domain = random.choices(available_domains, weights=weights)[0]
        
        # Create improvement focused on selected domain
        domain_templates = {
            'security': {
                'names': ['AuthenticationEnhancer', 'EncryptionOptimizer', 'SecurityAuditor', 'AccessControlManager'],
                'types': ['security', 'authentication', 'encryption', 'validation'],
                'approaches': ['Implement OAuth2 with JWT tokens', 'Add end-to-end encryption', 'Create security audit logs']
            },
            'monitoring': {
                'names': ['MetricsCollector', 'PerformanceTracker', 'SystemMonitor', 'AlertManager'],
                'types': ['monitoring', 'observability', 'metrics', 'alerting'],
                'approaches': ['Implement Prometheus metrics', 'Add distributed tracing', 'Create real-time dashboards']
            },
            'ai_ml': {
                'names': ['PredictiveAnalyzer', 'MLPipeline', 'NeuralOptimizer', 'AIAssistant'],
                'types': ['machine_learning', 'ai_enhancement', 'prediction', 'optimization'],
                'approaches': ['Use LSTM for prediction', 'Implement recommendation engine', 'Add sentiment analysis']
            },
            'workflow': {
                'names': ['WorkflowOrchestrator', 'PipelineManager', 'TaskAutomation', 'ProcessOptimizer'],
                'types': ['workflow', 'automation', 'orchestration', 'pipeline'],
                'approaches': ['Create DAG-based workflows', 'Implement event-driven automation', 'Add parallel processing']
            },
            'data_processing': {
                'names': ['DataTransformer', 'ETLPipeline', 'StreamProcessor', 'DataValidator'],
                'types': ['data_processing', 'etl', 'streaming', 'transformation'],
                'approaches': ['Implement stream processing', 'Create data validation rules', 'Add batch processing']
            }
        }
        
        template = domain_templates.get(selected_domain, {
            'names': [f'{selected_domain.title()}Enhancement'],
            'types': [selected_domain],
            'approaches': [f'Implement {selected_domain} improvements']
        })
        
        # Generate unique improvement
        timestamp = datetime.now().isoformat()
        return {
            'name': random.choice(template['names']) + f'_v{random.randint(1, 100)}',
            'type': random.choice(template['types']),
            'description': f'Enhanced {selected_domain} solution for better system capabilities',
            'approach': random.choice(template['approaches']),
            'metrics': [f'{selected_domain}_metric_1', f'{selected_domain}_metric_2'],
            'integration_points': ['system_core', selected_domain],
            'fitness': 0.0,
            'domain': selected_domain,
            'diversity_enforced': True,
            'id': hashlib.md5(f"{selected_domain}_{timestamp}".encode()).hexdigest()[:8]
        }
        
    def calculate_novelty_bonus(self, improvement: Dict) -> float:
        """Calculate novelty bonus for improvements in underexplored domains."""
        domain = self._classify_domain(improvement)
        
        # Update domain tracker
        self.domain_tracker[domain] += 1
        
        # Calculate novelty based on inverse frequency
        total_improvements = sum(self.domain_tracker.values())
        domain_frequency = self.domain_tracker[domain] / total_improvements if total_improvements > 0 else 1
        
        # Novelty bonus: higher for less frequent domains
        novelty_bonus = 1.0 - domain_frequency
        
        # Extra bonus for never-seen domains
        if self.domain_tracker[domain] == 1:
            novelty_bonus += 0.2
            
        return min(novelty_bonus * 0.3, 0.5)  # Cap at 0.5 bonus
        
    def prevent_duplicate_patterns(self, new_improvement: Dict, existing_population: List[Dict]) -> bool:
        """Check if new improvement is too similar to existing ones."""
        new_signature = self._create_signature(new_improvement)
        
        for existing in existing_population:
            existing_signature = self._create_signature(existing)
            similarity = self._calculate_similarity(new_signature, existing_signature)
            
            if similarity > self.similarity_threshold:
                self.logger.debug(f"Rejecting similar improvement: {new_improvement['name']} (similarity={similarity:.2f})")
                return False
                
        return True
        
    def _create_signature(self, improvement: Dict) -> str:
        """Create a signature for improvement comparison."""
        # Extract key features
        features = [
            improvement.get('type', ''),
            improvement.get('approach', ''),
            ' '.join(improvement.get('metrics', [])),
            ' '.join(improvement.get('integration_points', []))
        ]
        
        return ' '.join(features).lower()
        
    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures."""
        words1 = set(sig1.split())
        words2 = set(sig2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def get_diversity_report(self, population: List[Dict]) -> Dict[str, Any]:
        """Generate detailed diversity report."""
        domains = [self._classify_domain(imp) for imp in population]
        domain_counts = Counter(domains)
        
        return {
            'diversity_score': self.calculate_diversity_score(population),
            'domain_distribution': dict(domain_counts),
            'most_common_domain': domain_counts.most_common(1)[0] if domain_counts else ('none', 0),
            'unique_types': len(set(imp.get('type', 'unknown') for imp in population)),
            'unique_approaches': len(set(str(imp.get('approach', '')) for imp in population)),
            'recommendations': self._get_diversity_recommendations(domain_counts)
        }
        
    def _get_diversity_recommendations(self, domain_counts: Counter) -> List[str]:
        """Get recommendations for improving diversity."""
        recommendations = []
        
        # Check for over-concentration
        total = sum(domain_counts.values())
        for domain, count in domain_counts.items():
            if count / total > 0.3:  # More than 30% in one domain
                recommendations.append(f"Reduce focus on {domain} (currently {count/total:.1%})")
                
        # Check for missing domains
        missing_domains = set(self.domain_keywords.keys()) - set(domain_counts.keys())
        if missing_domains:
            recommendations.append(f"Explore domains: {', '.join(list(missing_domains)[:3])}")
            
        return recommendations