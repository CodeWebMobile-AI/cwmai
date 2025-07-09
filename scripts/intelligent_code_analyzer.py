"""
Intelligent Code Analyzer
=========================

Analyzes code relationships, understands file dependencies, and provides
deep insights for tool generation - going beyond Claude Code's capabilities.
"""

import ast
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx


class IntelligentCodeAnalyzer:
    """Analyzes code to understand relationships and patterns deeply."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.file_graph = nx.DiGraph()  # Directed graph of file dependencies
        self.class_map = {}  # Map of classes to their files
        self.function_map = {}  # Map of functions to their files
        self.import_map = defaultdict(set)  # Map of what each file imports
        self.pattern_library = {}  # Library of common patterns
        
    async def analyze_grep_results(self, grep_results: List[Dict[str, Any]], user_intent: str) -> Dict[str, Any]:
        """Intelligently analyze grep results to find the most relevant files."""
        
        # Score each file based on relevance
        file_scores = {}
        file_contexts = {}
        
        for result in grep_results:
            file_path = result['file']
            line_content = result['content']
            line_no = result['line']
            
            # Calculate relevance score
            score = 0
            context_hints = []
            
            # Check if it's a definition (class, function, etc)
            if re.match(r'^(class|def|async def)\s+\w+', line_content.strip()):
                score += 10
                context_hints.append("definition")
            
            # Check if it's in a main implementation file
            if not file_path.startswith('test_') and not file_path.endswith('_test.py'):
                score += 5
                
            # Check for specific patterns based on intent
            intent_keywords = self._extract_intent_keywords(user_intent)
            for keyword in intent_keywords:
                if keyword.lower() in line_content.lower():
                    score += 3
                    
            # Bonus for files with certain patterns
            if 'generator' in file_path.lower() and 'generate' in user_intent.lower():
                score += 8
                context_hints.append("generator")
            
            if 'analyzer' in file_path.lower() and 'analyze' in user_intent.lower():
                score += 8
                context_hints.append("analyzer")
                
            if 'creator' in file_path.lower() and 'create' in user_intent.lower():
                score += 8
                context_hints.append("creator")
            
            # Store score and context
            if file_path not in file_scores or score > file_scores[file_path]:
                file_scores[file_path] = score
                file_contexts[file_path] = {
                    'score': score,
                    'hints': context_hints,
                    'key_line': line_content,
                    'line_no': line_no
                }
        
        # Sort files by score
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze relationships between top files
        top_files = [f[0] for f in sorted_files[:5]]
        relationships = await self._analyze_file_relationships(top_files)
        
        return {
            'primary_files': top_files[:3],
            'supporting_files': top_files[3:],
            'file_contexts': file_contexts,
            'relationships': relationships,
            'recommended_read_order': self._determine_read_order(relationships, top_files)
        }
    
    async def analyze_code_patterns(self, file_path: str) -> Dict[str, Any]:
        """Deep analysis of code patterns in a file."""
        
        full_path = self.project_root / file_path
        if not full_path.exists():
            return {'error': 'File not found'}
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except:
            # If AST parsing fails, fall back to regex
            return self._analyze_with_regex(content)
        
        analysis = {
            'imports': [],
            'classes': [],
            'functions': [],
            'patterns': [],
            'dependencies': set(),
            'exports': [],
            'key_methods': []
        }
        
        # Analyze AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append({
                        'module': alias.name,
                        'alias': alias.asname
                    })
                    analysis['dependencies'].add(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    analysis['imports'].append({
                        'module': f"{module}.{alias.name}" if module else alias.name,
                        'from': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })
                    analysis['dependencies'].add(module)
                    
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'methods': [],
                    'base_classes': [self._get_name(base) for base in node.bases],
                    'decorators': [self._get_name(dec) for dec in node.decorator_list],
                    'docstring': ast.get_docstring(node)
                }
                
                # Get methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'decorators': [self._get_name(dec) for dec in item.decorator_list]
                        }
                        class_info['methods'].append(method_info)
                        
                        # Mark key methods
                        if item.name in ['__init__', 'execute', 'run', 'process', 'generate', 'analyze']:
                            analysis['key_methods'].append(f"{node.name}.{item.name}")
                
                analysis['classes'].append(class_info)
                
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Top-level functions
                if not self._is_nested(node):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'decorators': [self._get_name(dec) for dec in node.decorator_list],
                        'docstring': ast.get_docstring(node)
                    }
                    analysis['functions'].append(func_info)
                    
                    # Check if it's exported
                    if node.name.endswith('_tool') or not node.name.startswith('_'):
                        analysis['exports'].append(node.name)
        
        # Identify patterns
        analysis['patterns'] = self._identify_patterns(analysis, content)
        
        # Convert set to list for JSON serialization
        analysis['dependencies'] = list(analysis['dependencies'])
        
        return analysis
    
    async def suggest_code_reuse(self, intent: str, analyzed_files: Dict[str, Dict]) -> Dict[str, Any]:
        """Suggest specific code to reuse based on analysis."""
        
        suggestions = {
            'classes_to_import': [],
            'functions_to_reuse': [],
            'patterns_to_follow': [],
            'example_usage': []
        }
        
        # Analyze intent
        intent_lower = intent.lower()
        
        for file_path, analysis in analyzed_files.items():
            # Suggest classes
            for class_info in analysis.get('classes', []):
                class_name = class_info['name']
                
                # Check if class is relevant to intent
                if self._is_relevant_to_intent(class_name, class_info.get('docstring', ''), intent):
                    suggestions['classes_to_import'].append({
                        'class': class_name,
                        'from': file_path.replace('/', '.').replace('.py', ''),
                        'methods': [m['name'] for m in class_info.get('methods', [])[:5]],
                        'usage_hint': self._generate_usage_hint(class_info)
                    })
            
            # Suggest functions
            for func_info in analysis.get('functions', []):
                if self._is_relevant_to_intent(func_info['name'], func_info.get('docstring', ''), intent):
                    suggestions['functions_to_reuse'].append({
                        'function': func_info['name'],
                        'from': file_path.replace('/', '.').replace('.py', ''),
                        'args': func_info.get('args', []),
                        'is_async': func_info.get('is_async', False)
                    })
            
            # Suggest patterns
            for pattern in analysis.get('patterns', []):
                if pattern.get('relevance_score', 0) > 0.5:
                    suggestions['patterns_to_follow'].append(pattern)
        
        # Generate example usage
        suggestions['example_usage'] = self._generate_example_usage(suggestions, intent)
        
        return suggestions
    
    def _extract_intent_keywords(self, intent: str) -> List[str]:
        """Extract meaningful keywords from user intent."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'to', 'for', 'and', 'or', 'but', 'in', 'on', 'at', 'with', 'by'}
        words = re.findall(r'\w+', intent.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add variations
        variations = []
        for keyword in keywords:
            variations.append(keyword)
            # Add common variations
            if keyword.endswith('e'):
                variations.append(keyword[:-1] + 'ing')  # create -> creating
            elif keyword.endswith('y'):
                variations.append(keyword[:-1] + 'ies')  # quality -> qualities
            
        return list(set(variations))
    
    async def _analyze_file_relationships(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Analyze how files are related to each other."""
        relationships = defaultdict(list)
        
        for file_path in file_paths:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check imports
            import_pattern = re.compile(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)')
            for match in import_pattern.finditer(content):
                imported = match.group(1) or match.group(2)
                
                # Check if any of our files is being imported
                for other_file in file_paths:
                    if other_file != file_path:
                        other_module = other_file.replace('/', '.').replace('.py', '')
                        if imported.endswith(other_module.split('.')[-1]):
                            relationships[file_path].append(f"imports from {other_file}")
                            relationships[other_file].append(f"imported by {file_path}")
            
            # Check class inheritance
            class_pattern = re.compile(r'class\s+(\w+)\s*\(([^)]+)\)')
            for match in class_pattern.finditer(content):
                class_name = match.group(1)
                base_classes = match.group(2).split(',')
                
                for base in base_classes:
                    base = base.strip()
                    # Check if base class is defined in our files
                    for other_file in file_paths:
                        if other_file != file_path:
                            other_full = self.project_root / other_file
                            if other_full.exists():
                                with open(other_full, 'r') as of:
                                    if f'class {base}' in of.read():
                                        relationships[file_path].append(f"inherits {base} from {other_file}")
                                        relationships[other_file].append(f"base class for {file_path}")
        
        return dict(relationships)
    
    def _determine_read_order(self, relationships: Dict[str, List[str]], files: List[str]) -> List[str]:
        """Determine optimal order to read files based on dependencies."""
        # Build dependency graph
        graph = nx.DiGraph()
        
        for file in files:
            graph.add_node(file)
        
        for file, rels in relationships.items():
            for rel in rels:
                if 'imports from' in rel:
                    other = rel.replace('imports from ', '')
                    if other in files:
                        graph.add_edge(other, file)  # other should be read first
                elif 'base class for' in rel:
                    other = rel.replace('base class for ', '')
                    if other in files:
                        graph.add_edge(file, other)  # this file should be read first
        
        # Try topological sort
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            # If there are cycles, just return original order
            return files
    
    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, str):
            return node
        return str(node)
    
    def _is_nested(self, node) -> bool:
        """Check if a function is nested inside a class or another function."""
        # This is a simplified check - in real implementation would walk the AST
        return False
    
    def _analyze_with_regex(self, content: str) -> Dict[str, Any]:
        """Fallback analysis using regex when AST parsing fails."""
        analysis = {
            'imports': [],
            'classes': [],
            'functions': [],
            'patterns': [],
            'dependencies': [],
            'exports': []
        }
        
        # Extract imports
        import_pattern = re.compile(r'^(?:from\s+([\w.]+)\s+)?import\s+(.+)$', re.MULTILINE)
        for match in import_pattern.finditer(content):
            from_module = match.group(1)
            imports = match.group(2)
            
            if from_module:
                analysis['dependencies'].append(from_module)
            
            for imp in imports.split(','):
                imp = imp.strip()
                if ' as ' in imp:
                    name, alias = imp.split(' as ')
                    analysis['imports'].append({
                        'name': name.strip(),
                        'alias': alias.strip(),
                        'from': from_module
                    })
                else:
                    analysis['imports'].append({
                        'name': imp,
                        'from': from_module
                    })
        
        # Extract classes
        class_pattern = re.compile(r'^class\s+(\w+)(?:\(([^)]*)\))?:\s*$', re.MULTILINE)
        for match in class_pattern.finditer(content):
            analysis['classes'].append({
                'name': match.group(1),
                'base_classes': [b.strip() for b in (match.group(2) or '').split(',') if b.strip()]
            })
        
        # Extract functions
        func_pattern = re.compile(r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
        for match in func_pattern.finditer(content):
            analysis['functions'].append({
                'name': match.group(1),
                'args': [a.strip().split(':')[0].strip() for a in match.group(2).split(',') if a.strip()]
            })
        
        return analysis
    
    def _identify_patterns(self, analysis: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Identify common patterns in the code."""
        patterns = []
        
        # Pattern: Tool definition
        if any(func['name'].endswith('_tool') for func in analysis.get('functions', [])):
            patterns.append({
                'type': 'tool_pattern',
                'description': 'Follows tool naming convention with _tool suffix',
                'relevance_score': 0.9
            })
        
        # Pattern: Async operations
        if any(func.get('is_async') for func in analysis.get('functions', [])):
            patterns.append({
                'type': 'async_pattern',
                'description': 'Uses async/await pattern',
                'relevance_score': 0.7
            })
        
        # Pattern: Error handling
        if 'try:' in content and 'except' in content:
            patterns.append({
                'type': 'error_handling',
                'description': 'Implements try/except error handling',
                'relevance_score': 0.8
            })
        
        # Pattern: State management
        if 'StateManager' in content or 'state_manager' in content:
            patterns.append({
                'type': 'state_management',
                'description': 'Uses StateManager for persistence',
                'relevance_score': 0.8
            })
        
        return patterns
    
    def _is_relevant_to_intent(self, name: str, docstring: str, intent: str) -> bool:
        """Check if a class/function is relevant to the user's intent."""
        intent_lower = intent.lower()
        name_lower = name.lower()
        doc_lower = docstring.lower() if docstring else ''
        
        # Direct name match
        intent_words = set(re.findall(r'\w+', intent_lower))
        name_words = set(re.findall(r'\w+', name_lower))
        
        if len(intent_words & name_words) > 0:
            return True
        
        # Docstring match
        if doc_lower and any(word in doc_lower for word in intent_words if len(word) > 3):
            return True
        
        # Semantic matches
        semantic_matches = {
            'check': ['verify', 'validate', 'ensure', 'test'],
            'create': ['generate', 'make', 'build', 'construct'],
            'analyze': ['examine', 'inspect', 'review', 'assess'],
            'update': ['modify', 'change', 'edit', 'patch']
        }
        
        for intent_word in intent_words:
            if intent_word in semantic_matches:
                for synonym in semantic_matches[intent_word]:
                    if synonym in name_lower or synonym in doc_lower:
                        return True
        
        return False
    
    def _generate_usage_hint(self, class_info: Dict[str, Any]) -> str:
        """Generate a usage hint for a class."""
        class_name = class_info['name']
        
        # Check for common patterns
        if 'Generator' in class_name:
            return f"{class_name}().generate(...)"
        elif 'Analyzer' in class_name:
            return f"{class_name}().analyze(...)"
        elif 'Manager' in class_name:
            return f"{class_name}().manage(...)"
        elif 'Creator' in class_name:
            return f"{class_name}().create(...)"
        
        # Check methods
        methods = class_info.get('methods', [])
        key_methods = ['execute', 'run', 'process', 'perform']
        
        for method in methods:
            if method['name'] in key_methods:
                return f"{class_name}().{method['name']}(...)"
        
        # Default
        if methods:
            first_public_method = next((m['name'] for m in methods if not m['name'].startswith('_')), None)
            if first_public_method:
                return f"{class_name}().{first_public_method}(...)"
        
        return f"{class_name}()"
    
    def _generate_example_usage(self, suggestions: Dict[str, Any], intent: str) -> List[str]:
        """Generate example usage based on suggestions."""
        examples = []
        
        # Example imports
        if suggestions['classes_to_import']:
            for cls in suggestions['classes_to_import'][:2]:
                examples.append(f"from {cls['from']} import {cls['class']}")
        
        if suggestions['functions_to_reuse']:
            for func in suggestions['functions_to_reuse'][:2]:
                examples.append(f"from {func['from']} import {func['function']}")
        
        examples.append("")  # Blank line
        
        # Example usage
        if suggestions['classes_to_import']:
            cls = suggestions['classes_to_import'][0]
            examples.append(f"# Using {cls['class']}")
            examples.append(f"instance = {cls['usage_hint']}")
            
        if suggestions['functions_to_reuse']:
            func = suggestions['functions_to_reuse'][0]
            if func['is_async']:
                examples.append(f"# Using {func['function']}")
                examples.append(f"result = await {func['function']}({', '.join(func['args'])})")
            else:
                examples.append(f"# Using {func['function']}")
                examples.append(f"result = {func['function']}({', '.join(func['args'])})")
        
        return examples