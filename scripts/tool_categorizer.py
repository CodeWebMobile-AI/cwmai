"""
Tool Categorizer Module

Dynamically discovers categories from the file system and learns
patterns to categorize new tools appropriately.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

class ToolCategorizer:
    """Dynamically categorizes tools based on existing structure and learned patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.custom_tools_dir = Path("scripts/custom_tools")
        self.state_file = Path("scripts/tool_categorizer_state.json")
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Discover existing categories
        self.categories = self._discover_categories()
        
        # Learn patterns from existing tools
        self._learn_from_existing_tools()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load categorizer state from file."""
        if self.state_file.exists():
            try:
                loaded_state = json.loads(self.state_file.read_text())
                # Convert regular dicts back to defaultdicts
                state = {
                    'learned_patterns': defaultdict(lambda: defaultdict(list)),
                    'category_stats': defaultdict(lambda: {'count': 0, 'confidence_sum': 0}),
                    'new_categories_created': loaded_state.get('new_categories_created', []),
                    'last_updated': loaded_state.get('last_updated', datetime.now().isoformat())
                }
                
                # Populate defaultdicts with loaded data
                for k, v in loaded_state.get('learned_patterns', {}).items():
                    for k2, v2 in v.items():
                        state['learned_patterns'][k][k2] = v2
                        
                for k, v in loaded_state.get('category_stats', {}).items():
                    state['category_stats'][k] = v
                    
                return state
            except Exception as e:
                self.logger.error(f"Error loading state: {e}")
                
        return {
            'learned_patterns': defaultdict(lambda: defaultdict(list)),
            'category_stats': defaultdict(lambda: {'count': 0, 'confidence_sum': 0}),
            'new_categories_created': [],
            'last_updated': datetime.now().isoformat()
        }
        
    def _save_state(self):
        """Save categorizer state to file."""
        try:
            # Convert defaultdicts to regular dicts for JSON serialization
            state_to_save = {
                'learned_patterns': {k: dict(v) for k, v in self.state['learned_patterns'].items()},
                'category_stats': {k: v for k, v in self.state['category_stats'].items()},
                'new_categories_created': self.state['new_categories_created'],
                'last_updated': datetime.now().isoformat()
            }
            self.state_file.write_text(json.dumps(state_to_save, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            
    def _discover_categories(self) -> Dict[str, Dict[str, Any]]:
        """Dynamically discover categories from the file system."""
        categories = {}
        
        # Check for subdirectories in custom_tools
        if self.custom_tools_dir.exists():
            for item in self.custom_tools_dir.iterdir():
                if item.is_dir() and not item.name.startswith('__'):
                    category_name = item.name
                    
                    # Count tools in this category
                    tool_count = len(list(item.glob("*.py"))) - (1 if (item / "__init__.py").exists() else 0)
                    
                    categories[category_name] = {
                        'path': item,
                        'tool_count': tool_count,
                        'patterns': [],  # Will be learned
                        'imports': [],   # Will be learned
                        'description': f"Tools in {category_name} category"
                    }
                    
        # Add 'general' for root-level tools
        root_tools = len([f for f in self.custom_tools_dir.glob("*.py") if f.name != "__init__.py"])
        if root_tools > 0:
            categories['general'] = {
                'path': self.custom_tools_dir,
                'tool_count': root_tools,
                'patterns': [],
                'imports': [],
                'description': "General purpose tools"
            }
            
        self.logger.info(f"Discovered {len(categories)} categories: {list(categories.keys())}")
        return categories
        
    def _learn_from_existing_tools(self):
        """Learn patterns from existing categorized tools."""
        for category_name, category_info in self.categories.items():
            path = category_info['path']
            
            # Find all tools in this category
            if category_name == 'general':
                tool_files = [f for f in path.glob("*.py") if f.name != "__init__.py"]
            else:
                tool_files = list(path.glob("*.py"))
                tool_files = [f for f in tool_files if f.name != "__init__.py"]
                
            # Learn from each tool
            learned_patterns = defaultdict(int)
            learned_imports = defaultdict(int)
            
            for tool_file in tool_files:
                try:
                    code = tool_file.read_text()
                    tool_name = tool_file.stem
                    
                    # Extract patterns from tool name
                    words = re.findall(r'\w+', tool_name.lower())
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            learned_patterns[word] += 1
                            
                    # Extract description
                    desc_match = re.search(r'__description__\s*=\s*["\'](.+?)["\']', code)
                    if desc_match:
                        desc_words = re.findall(r'\w+', desc_match.group(1).lower())
                        for word in desc_words:
                            if len(word) > 3:  # Skip short words in descriptions
                                learned_patterns[word] += 0.5  # Lower weight for description
                                
                    # Extract imports
                    import_matches = re.findall(r'(?:from|import)\s+(\S+)', code)
                    for imp in import_matches:
                        base_module = imp.split('.')[0]
                        if base_module not in ['scripts', 'os', 'sys', 're', 'json']:  # Skip common imports
                            learned_imports[base_module] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error learning from {tool_file}: {e}")
                    
            # Store learned patterns (keep top patterns)
            if learned_patterns:
                top_patterns = sorted(learned_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
                category_info['patterns'] = [p[0] for p in top_patterns]
                self.state['learned_patterns'][category_name]['patterns'] = category_info['patterns']
                
            if learned_imports:
                top_imports = sorted(learned_imports.items(), key=lambda x: x[1], reverse=True)[:10]
                category_info['imports'] = [i[0] for i in top_imports]
                self.state['learned_patterns'][category_name]['imports'] = category_info['imports']
                
        self._save_state()
        
    def categorize_tool(self, tool_name: str, description: str = "", 
                       code: str = "", imports: List[str] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Determine the category for a tool using learned patterns.
        
        Args:
            tool_name: Name of the tool
            description: Tool description
            code: Tool source code
            imports: List of imports used in the tool
            
        Returns:
            Tuple of (category_name, confidence_score, analysis_details)
        """
        scores = defaultdict(float)
        analysis = {
            'tool_name': tool_name,
            'matched_patterns': defaultdict(list),
            'matched_imports': defaultdict(list),
            'reasons': []
        }
        
        # Extract imports from code if not provided
        if imports is None and code:
            imports = self._extract_imports(code)
            
        tool_name_lower = tool_name.lower()
        desc_lower = description.lower() if description else ""
        
        # Score based on learned patterns
        for category_name, category_info in self.categories.items():
            # Skip empty categories unless they have patterns
            if category_info['tool_count'] == 0 and not category_info['patterns']:
                continue
                
            # Check name patterns
            for pattern in category_info['patterns']:
                if pattern in tool_name_lower:
                    scores[category_name] += 3.0
                    analysis['matched_patterns'][category_name].append(pattern)
                elif pattern in desc_lower:
                    scores[category_name] += 1.5
                    analysis['matched_patterns'][category_name].append(f"desc:{pattern}")
                    
            # Check import patterns
            if imports and category_info['imports']:
                for imp in imports:
                    imp_base = imp.split('.')[0].lower()
                    if imp_base in category_info['imports']:
                        scores[category_name] += 2.0
                        analysis['matched_imports'][category_name].append(imp)
                        
            # Bonus for categories with many tools (established categories)
            if category_info['tool_count'] > 5:
                scores[category_name] += 0.5
                
        # Determine best category
        if not scores:
            # No matches - suggest a new category or use general
            suggested_new = self._suggest_new_category(tool_name, description, code)
            if suggested_new:
                category = suggested_new
                confidence = 0.6
                analysis['reasons'].append(f"New category suggested based on tool characteristics")
            else:
                category = 'general'
                confidence = 0.4
                analysis['reasons'].append("No specific matches, using general category")
        else:
            # Get category with highest score
            category = max(scores.items(), key=lambda x: x[1])[0]
            max_score = scores[category]
            
            # Calculate confidence
            if max_score >= 5.0:
                confidence = 0.9
            elif max_score >= 3.0:
                confidence = 0.7
            elif max_score >= 1.5:
                confidence = 0.5
            else:
                confidence = 0.3
                
            # Add reasoning
            if analysis['matched_patterns'][category]:
                analysis['reasons'].append(f"Matched patterns: {', '.join(set(analysis['matched_patterns'][category]))}")
            if analysis['matched_imports'][category]:
                analysis['reasons'].append(f"Matched imports: {', '.join(analysis['matched_imports'][category])}")
                
        # Update stats
        self.state['category_stats'][category]['count'] += 1
        self.state['category_stats'][category]['confidence_sum'] += confidence
        
        analysis['confidence'] = confidence
        analysis['category'] = category
        analysis['scores'] = dict(scores)
        
        return category, confidence, analysis
        
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        import_pattern = re.compile(r'(?:from|import)\s+(\S+)')
        
        for match in import_pattern.finditer(code):
            imports.append(match.group(1))
            
        return imports
        
    def _suggest_new_category(self, tool_name: str, description: str, code: str) -> Optional[str]:
        """
        Suggest a new category name based on tool characteristics.
        Uses pattern analysis to determine if a new category is warranted.
        """
        tool_name_lower = tool_name.lower()
        all_text = f"{tool_name_lower} {description.lower()} {code.lower()}"
        
        # Extract meaningful words from tool name (handle snake_case)
        # Split by underscore first, then extract words
        tool_parts = tool_name_lower.split('_')
        words = []
        for part in tool_parts:
            if len(part) >= 3:  # Words with 3+ chars
                words.append(part)
        
        # Analyze if tool has a strong domain focus
        word_frequencies = defaultdict(int)
        
        # Count occurrences of each word in the full text
        for word in words:
            if word not in ['tool', 'create', 'get', 'set', 'make', 'add', 'new']:  # Skip generic verbs
                count = len(re.findall(r'\b' + word + r'\b', all_text))
                if count >= 3:  # Word appears multiple times
                    word_frequencies[word] = count
                    
        # Look for the most dominant word as potential category
        if word_frequencies:
            # Get the most frequent meaningful word
            dominant_word = max(word_frequencies.items(), key=lambda x: x[1])[0]
            
            # Check if this could be a valid category
            if self._is_valid_category_name(dominant_word):
                # Ensure it's sufficiently different from existing categories
                for existing_cat in self.categories:
                    if (dominant_word in existing_cat or existing_cat in dominant_word or
                        self._similar_words(dominant_word, existing_cat)):
                        return None  # Too similar to existing category
                        
                return dominant_word
                
        # If we have multiple related tools waiting to be categorized,
        # learn from patterns in uncategorized tools
        uncategorized_patterns = self._analyze_uncategorized_tools()
        if uncategorized_patterns:
            for pattern, count in uncategorized_patterns.items():
                if count >= 3 and pattern in tool_name_lower:
                    return pattern
                    
        return None
        
    def _is_valid_category_name(self, name: str) -> bool:
        """Check if a name is valid for a category."""
        # Must be meaningful length
        if len(name) < 3 or len(name) > 20:
            return False
            
        # Must be a noun-like word (not a verb)
        verb_suffixes = ['ing', 'ed', 'ize', 'ify']
        for suffix in verb_suffixes:
            if name.endswith(suffix):
                return False
                
        # Should not be a common programming term
        common_terms = {'function', 'class', 'method', 'variable', 'script', 'module', 'package'}
        if name in common_terms:
            return False
            
        return True
        
    def _similar_words(self, word1: str, word2: str) -> bool:
        """Check if two words are similar (share common root)."""
        # Check if one is substring of other (with some tolerance)
        if len(word1) > 4 and len(word2) > 4:
            if word1[:4] == word2[:4]:  # Same prefix
                return True
                
        # Check Levenshtein-like similarity for short words
        if abs(len(word1) - len(word2)) <= 2:
            common_chars = sum(1 for c in word1 if c in word2)
            if common_chars >= min(len(word1), len(word2)) - 1:
                return True
                
        return False
        
    def _analyze_uncategorized_tools(self) -> Dict[str, int]:
        """Analyze patterns in tools that are in the general category."""
        patterns = defaultdict(int)
        
        if 'general' in self.categories:
            general_path = self.categories['general']['path']
            tool_files = [f for f in general_path.glob("*.py") if f.name != "__init__.py"]
            
            for tool_file in tool_files:
                tool_name = tool_file.stem.lower()
                # Extract meaningful words
                words = re.findall(r'\b\w{4,}\b', tool_name)
                for word in words:
                    if word not in ['tool', 'create', 'make', 'general']:
                        patterns[word] += 1
                        
        # Return patterns that appear in multiple tools
        return {k: v for k, v in patterns.items() if v >= 2}
        
    def _is_new_domain(self, compound: str, full_text: str) -> bool:
        """Check if a compound term represents a new domain."""
        # Must appear multiple times to be significant
        if full_text.count(compound.replace('_', '')) < 2:
            return False
            
        # Must not be too similar to existing categories
        for category in self.categories:
            if compound.startswith(category) or category.startswith(compound):
                return False
                
        return True
        
    def ensure_category_exists(self, category: str) -> Path:
        """Ensure a category directory exists and has __init__.py."""
        if category == 'general':
            return self.custom_tools_dir
            
        category_path = self.custom_tools_dir / category
        
        if not category_path.exists():
            category_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py
            init_file = category_path / "__init__.py"
            init_file.write_text(f'"""Tools for {category} category."""\n')
            
            # Add to our categories
            self.categories[category] = {
                'path': category_path,
                'tool_count': 0,
                'patterns': [],
                'imports': [],
                'description': f"Tools for {category}"
            }
            
            # Track new category creation
            if category not in self.state['new_categories_created']:
                self.state['new_categories_created'].append(category)
                self._save_state()
                
            self.logger.info(f"Created new category: {category}")
            
        return category_path
        
    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about categorization."""
        stats = {
            'total_categories': len(self.categories),
            'categories': {},
            'new_categories_created': self.state['new_categories_created'],
            'last_updated': self.state.get('last_updated', 'unknown')
        }
        
        for cat_name, cat_info in self.categories.items():
            cat_stats = self.state['category_stats'].get(cat_name, {'count': 0, 'confidence_sum': 0})
            avg_confidence = cat_stats['confidence_sum'] / max(cat_stats['count'], 1)
            
            stats['categories'][cat_name] = {
                'existing_tools': cat_info['tool_count'],
                'categorized_count': cat_stats['count'],
                'average_confidence': avg_confidence,
                'learned_patterns': cat_info['patterns'][:5],  # Top 5
                'learned_imports': cat_info['imports'][:5]     # Top 5
            }
            
        return stats