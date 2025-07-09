"""
Similarity Configuration

Configuration file for problem similarity checking.
This allows customization without modifying the core similarity checker code.
"""

from typing import Dict, List, Set


class SimilarityConfig:
    """Configuration for similarity checking."""
    
    def __init__(self):
        """Initialize with default configuration."""
        # Similarity thresholds
        self.thresholds = {
            'default': 0.7,
            'strict': 0.85,
            'loose': 0.6
        }
        
        # Component weights
        self.weights = {
            'default': {
                'jaccard': 0.5,
                'sequence': 0.35,
                'phrase': 0.15
            },
            'text_heavy': {
                'jaccard': 0.3,
                'sequence': 0.5,
                'phrase': 0.2
            },
            'keyword_heavy': {
                'jaccard': 0.6,
                'sequence': 0.2,
                'phrase': 0.2
            }
        }
        
        # Language-specific stop words
        self.stop_words = {
            'english': self._get_english_stop_words(),
            'minimal': {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an'}
        }
        
        # Domain configurations
        self.domains = {
            'business': {
                'synonyms': {
                    'company': ['business', 'organization', 'enterprise', 'firm'],
                    'customer': ['client', 'user', 'consumer', 'buyer'],
                    'payment': ['invoice', 'billing', 'transaction'],
                },
                'key_terms': ['revenue', 'profit', 'cost', 'market', 'sales']
            },
            'technical': {
                'synonyms': {
                    'software': ['application', 'app', 'program', 'tool'],
                    'bug': ['error', 'issue', 'problem', 'defect'],
                    'database': ['db', 'datastore', 'storage'],
                },
                'key_terms': ['api', 'backend', 'frontend', 'server', 'client']
            },
            'general': {
                'synonyms': {
                    'problem': ['issue', 'challenge', 'difficulty', 'concern'],
                    'solution': ['answer', 'resolution', 'fix', 'remedy'],
                    'manage': ['handle', 'control', 'organize', 'coordinate']
                },
                'key_terms': []
            }
        }
        
        # Phrase patterns by category
        self.phrase_patterns = {
            'problem_indicators': [
                r'(?:difficulty|struggle|challenge|problem|issue)s? (?:in |with |of )?',
                r'(?:lack|absence) of',
                r'(?:unable|fail|cannot) to',
                r'(?:hard|difficult) to'
            ],
            'need_indicators': [
                r'need(?:s)? (?:to |for )?',
                r'require(?:s|ment)? (?:to |for )?',
                r'want(?:s)? (?:to )?'
            ],
            'solution_indicators': [
                r'(?:solve|fix|address|handle) (?:the )?',
                r'(?:improve|enhance|optimize) (?:the )?',
                r'(?:automate|streamline) (?:the )?'
            ]
        }
    
    def _get_english_stop_words(self) -> Set[str]:
        """Get comprehensive English stop words."""
        return {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are',
            'been', 'by', 'for', 'from', 'has', 'have', 'in', 'of', 'or', 'to',
            'was', 'will', 'with', 'their', 'they', 'this', 'that', 'these',
            'those', 'it', 'its', 'can', 'could', 'would', 'should', 'may',
            'might', 'must', 'shall', 'will', 'need', 'needs', 'require',
            'be', 'being', 'had', 'do', 'does', 'did', 'doing', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'them', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'but'
        }
    
    def get_config_for_domain(self, domain: str = 'general') -> Dict:
        """Get configuration for a specific domain."""
        if domain not in self.domains:
            domain = 'general'
        
        return {
            'stop_words': self.stop_words.get('english', set()),
            'synonyms': self.domains[domain]['synonyms'],
            'key_terms': self.domains[domain]['key_terms'],
            'weights': self.weights.get('default'),
            'threshold': self.thresholds.get('default'),
            'phrase_patterns': self.phrase_patterns
        }
    
    def merge_configs(self, *configs: Dict) -> Dict:
        """Merge multiple configurations."""
        merged = {
            'stop_words': set(),
            'synonyms': {},
            'key_terms': [],
            'weights': self.weights['default'].copy(),
            'threshold': self.thresholds['default'],
            'phrase_patterns': {}
        }
        
        for config in configs:
            if 'stop_words' in config:
                merged['stop_words'].update(config['stop_words'])
            
            if 'synonyms' in config:
                for key, values in config['synonyms'].items():
                    if key in merged['synonyms']:
                        merged['synonyms'][key].extend(values)
                        merged['synonyms'][key] = list(set(merged['synonyms'][key]))
                    else:
                        merged['synonyms'][key] = values
            
            if 'key_terms' in config:
                merged['key_terms'].extend(config['key_terms'])
                merged['key_terms'] = list(set(merged['key_terms']))
            
            if 'weights' in config:
                merged['weights'].update(config['weights'])
            
            if 'threshold' in config:
                merged['threshold'] = config['threshold']
            
            if 'phrase_patterns' in config:
                for category, patterns in config['phrase_patterns'].items():
                    if category in merged['phrase_patterns']:
                        merged['phrase_patterns'][category].extend(patterns)
                    else:
                        merged['phrase_patterns'][category] = patterns
        
        return merged


# Default instance
default_config = SimilarityConfig()