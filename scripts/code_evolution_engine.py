"""
Code Evolution Engine

Continuously evolves the codebase by:
1. Analyzing code patterns and anti-patterns
2. Refactoring for better performance and maintainability
3. Learning from successful code changes
4. Propagating improvements across the codebase
5. Generating better implementations based on usage
"""

import ast
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict
import difflib

from scripts.http_ai_client import HTTPAIClient
from scripts.code_analyzer import CodeAnalyzer
from scripts.smart_tool_generator import SmartToolGenerator


@dataclass
class CodePattern:
    """Represents a code pattern (good or bad)."""
    pattern_type: str  # 'best_practice', 'anti_pattern', 'optimization'
    description: str
    detection_regex: Optional[str]
    ast_pattern: Optional[Dict[str, Any]]
    occurrences: List[Dict[str, Any]]  # file, line, context
    replacement_template: Optional[str]
    impact: str  # 'performance', 'maintainability', 'reliability'
    confidence: float


@dataclass 
class CodeEvolution:
    """Represents an evolution/improvement to code."""
    file_path: str
    original_code: str
    evolved_code: str
    description: str
    improvements: List[str]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    confidence: float
    applied: bool = False


class CodeEvolutionEngine:
    """Evolves code continuously for better performance and maintainability."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
        self.code_analyzer = CodeAnalyzer()
        self.tool_generator = SmartToolGenerator()
        # Git integration removed - not needed for core functionality
        
        # Pattern database
        self.known_patterns = self._load_initial_patterns()
        self.discovered_patterns = []
        self.evolution_history = []
        
        # Metrics tracking
        self.code_metrics = defaultdict(dict)
        self.improvement_metrics = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'performance_improvements': 0,
            'reliability_improvements': 0,
            'maintainability_improvements': 0
        }
        
        # Configuration
        self.config = {
            'scan_interval': 600,  # 10 minutes
            'max_evolutions_per_cycle': 3,
            'min_confidence_threshold': 0.7,
            'performance_improvement_threshold': 0.1,  # 10% improvement
            'test_before_apply': True
        }
        
        self.is_running = False
        
    def _load_initial_patterns(self) -> List[CodePattern]:
        """Load initial known patterns."""
        return [
            CodePattern(
                pattern_type='anti_pattern',
                description='Synchronous I/O in async function',
                detection_regex=r'async def.*?[\s\S]*?(?:open\(|read\(|write\()',
                ast_pattern=None,
                occurrences=[],
                replacement_template='Use aiofiles for async file operations',
                impact='performance',
                confidence=0.9
            ),
            CodePattern(
                pattern_type='anti_pattern',
                description='Not using context managers for resources',
                detection_regex=r'(\w+)\s*=\s*open\([^)]+\)(?!.*?with)',
                ast_pattern=None,
                occurrences=[],
                replacement_template='Use with statement for resource management',
                impact='reliability',
                confidence=0.9
            ),
            CodePattern(
                pattern_type='optimization',
                description='List comprehension instead of loop append',
                detection_regex=r'for\s+\w+\s+in\s+.*:\s*\n\s*\w+\.append\(',
                ast_pattern=None,
                occurrences=[],
                replacement_template='Use list comprehension',
                impact='performance',
                confidence=0.8
            ),
            CodePattern(
                pattern_type='best_practice',
                description='Type hints for function parameters',
                detection_regex=r'def\s+\w+\([^:)]+\)(?!.*->)',
                ast_pattern=None,
                occurrences=[],
                replacement_template='Add type hints',
                impact='maintainability',
                confidence=0.7
            )
        ]
        
    async def start(self):
        """Start the code evolution engine."""
        print("Starting Code Evolution Engine")
        self.is_running = True
        asyncio.create_task(self._evolution_loop())
        
    async def stop(self):
        """Stop the evolution engine."""
        self.is_running = False
        
    async def _evolution_loop(self):
        """Main evolution loop."""
        while self.is_running:
            try:
                await self._run_evolution_cycle()
                await asyncio.sleep(self.config['scan_interval'])
            except Exception as e:
                self.logger.error(f"Error in evolution loop: {e}")
                await asyncio.sleep(60)
                
    async def _run_evolution_cycle(self):
        """Run a single evolution cycle."""
        print("Running code evolution cycle")
        
        # 1. Scan codebase for patterns
        patterns_found = await self._scan_codebase()
        
        # 2. Analyze code metrics
        metrics = await self._analyze_code_metrics()
        
        # 3. Identify evolution opportunities
        opportunities = await self._identify_evolution_opportunities(patterns_found, metrics)
        
        # 4. Generate evolved code
        evolutions = await self._generate_evolutions(opportunities)
        
        # 5. Test and apply evolutions
        applied = await self._apply_evolutions(evolutions)
        
        # 6. Learn from results
        await self._learn_from_evolutions(applied)
        
        print(f"Evolution cycle complete: {len(applied)} evolutions applied")
        
    async def _scan_codebase(self) -> Dict[str, List[CodePattern]]:
        """Scan codebase for patterns."""
        patterns_by_file = defaultdict(list)
        
        # Scan Python files
        for py_file in Path('scripts').rglob('*.py'):
            if py_file.name.startswith('_'):
                continue
                
            try:
                content = py_file.read_text()
                
                # Check each known pattern
                for pattern in self.known_patterns + self.discovered_patterns:
                    if pattern.detection_regex:
                        matches = re.finditer(pattern.detection_regex, content, re.MULTILINE)
                        for match in matches:
                            pattern.occurrences.append({
                                'file': str(py_file),
                                'line': content[:match.start()].count('\n') + 1,
                                'context': match.group(0)
                            })
                            patterns_by_file[str(py_file)].append(pattern)
                            
                # AST-based pattern detection
                if content:
                    ast_patterns = await self._detect_ast_patterns(py_file, content)
                    patterns_by_file[str(py_file)].extend(ast_patterns)
                    
            except Exception as e:
                self.logger.error(f"Error scanning {py_file}: {e}")
                
        return patterns_by_file
        
    async def _detect_ast_patterns(self, file_path: Path, content: str) -> List[CodePattern]:
        """Detect patterns using AST analysis."""
        patterns = []
        
        try:
            tree = ast.parse(content)
            
            # Detect functions without docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        patterns.append(CodePattern(
                            pattern_type='best_practice',
                            description='Function without docstring',
                            detection_regex=None,
                            ast_pattern={'type': 'FunctionDef', 'has_docstring': False},
                            occurrences=[{
                                'file': str(file_path),
                                'line': node.lineno,
                                'context': node.name
                            }],
                            replacement_template='Add descriptive docstring',
                            impact='maintainability',
                            confidence=0.8
                        ))
                        
                    # Detect overly complex functions
                    complexity = self._calculate_complexity(node)
                    if complexity > 10:
                        patterns.append(CodePattern(
                            pattern_type='anti_pattern',
                            description='High complexity function',
                            detection_regex=None,
                            ast_pattern={'type': 'FunctionDef', 'complexity': complexity},
                            occurrences=[{
                                'file': str(file_path),
                                'line': node.lineno,
                                'context': f"{node.name} (complexity: {complexity})"
                            }],
                            replacement_template='Refactor into smaller functions',
                            impact='maintainability',
                            confidence=0.7
                        ))
                        
        except Exception as e:
            self.logger.error(f"AST analysis failed for {file_path}: {e}")
            
        return patterns
        
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += len(child.items)
                
        return complexity
        
    async def _analyze_code_metrics(self) -> Dict[str, Any]:
        """Analyze overall code metrics."""
        metrics = {
            'total_files': 0,
            'total_lines': 0,
            'avg_complexity': 0,
            'test_coverage': 0,
            'code_duplication': 0,
            'type_hint_coverage': 0
        }
        
        complexities = []
        total_functions = 0
        functions_with_hints = 0
        
        for py_file in Path('scripts').rglob('*.py'):
            if py_file.name.startswith('_'):
                continue
                
            try:
                content = py_file.read_text()
                metrics['total_files'] += 1
                metrics['total_lines'] += len(content.splitlines())
                
                # Analyze with AST
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        complexities.append(self._calculate_complexity(node))
                        
                        # Check for type hints
                        if node.returns or any(arg.annotation for arg in node.args.args):
                            functions_with_hints += 1
                            
            except:
                pass
                
        if complexities:
            metrics['avg_complexity'] = sum(complexities) / len(complexities)
            
        if total_functions > 0:
            metrics['type_hint_coverage'] = functions_with_hints / total_functions
            
        # Store metrics for comparison
        self.code_metrics[datetime.now(timezone.utc).isoformat()] = metrics
        
        return metrics
        
    async def _identify_evolution_opportunities(self, patterns: Dict[str, List[CodePattern]], 
                                              metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for code evolution."""
        opportunities = []
        
        # Pattern-based opportunities
        for file_path, file_patterns in patterns.items():
            for pattern in file_patterns:
                if pattern.confidence >= self.config['min_confidence_threshold']:
                    opportunities.append({
                        'type': 'pattern',
                        'file': file_path,
                        'pattern': pattern,
                        'priority': pattern.confidence * (2.0 if pattern.impact == 'performance' else 1.0)
                    })
                    
        # Metric-based opportunities
        if metrics['avg_complexity'] > 8:
            # Find most complex files
            complex_files = await self._find_complex_files()
            for file_info in complex_files[:3]:
                opportunities.append({
                    'type': 'complexity',
                    'file': file_info['file'],
                    'description': 'Reduce complexity',
                    'priority': 1.5
                })
                
        # AI-suggested improvements
        ai_suggestions = await self._get_ai_improvement_suggestions(metrics)
        opportunities.extend(ai_suggestions)
        
        # Sort by priority
        opportunities.sort(key=lambda x: x.get('priority', 1.0), reverse=True)
        
        return opportunities
        
    async def _find_complex_files(self) -> List[Dict[str, Any]]:
        """Find files with highest complexity."""
        file_complexities = []
        
        for py_file in Path('scripts').rglob('*.py'):
            if py_file.name.startswith('_'):
                continue
                
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                
                max_complexity = 0
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_complexity(node)
                        max_complexity = max(max_complexity, complexity)
                        
                if max_complexity > 0:
                    file_complexities.append({
                        'file': str(py_file),
                        'max_complexity': max_complexity
                    })
                    
            except:
                pass
                
        return sorted(file_complexities, key=lambda x: x['max_complexity'], reverse=True)
        
    async def _get_ai_improvement_suggestions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get AI suggestions for improvements."""
        prompt = f"""Analyze these code metrics and suggest improvements:

Metrics:
{json.dumps(metrics, indent=2)}

Recent patterns found:
{json.dumps([p.description for p in self.discovered_patterns[-5:]], indent=2)}

Suggest 3-5 specific improvements that would have the most impact. For each:
1. What file or component to improve
2. What specific change to make
3. Expected impact
4. Implementation approach

Return as JSON list.
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        suggestions = []
        try:
            ai_suggestions = json.loads(response.get('content', '[]'))
            for suggestion in ai_suggestions:
                suggestions.append({
                    'type': 'ai_suggestion',
                    'file': suggestion.get('file', 'unknown'),
                    'description': suggestion.get('change', ''),
                    'impact': suggestion.get('impact', ''),
                    'approach': suggestion.get('approach', ''),
                    'priority': 1.2
                })
        except:
            pass
            
        return suggestions
        
    async def _generate_evolutions(self, opportunities: List[Dict[str, Any]]) -> List[CodeEvolution]:
        """Generate code evolutions for opportunities."""
        evolutions = []
        
        for opp in opportunities[:self.config['max_evolutions_per_cycle']]:
            try:
                if opp['type'] == 'pattern':
                    evolution = await self._evolve_for_pattern(opp['file'], opp['pattern'])
                elif opp['type'] == 'complexity':
                    evolution = await self._evolve_for_complexity(opp['file'])
                elif opp['type'] == 'ai_suggestion':
                    evolution = await self._evolve_for_suggestion(opp)
                else:
                    continue
                    
                if evolution and evolution.confidence >= self.config['min_confidence_threshold']:
                    evolutions.append(evolution)
                    
            except Exception as e:
                self.logger.error(f"Failed to generate evolution: {e}")
                
        return evolutions
        
    async def _evolve_for_pattern(self, file_path: str, pattern: CodePattern) -> Optional[CodeEvolution]:
        """Generate evolution to fix a pattern."""
        try:
            content = Path(file_path).read_text()
            
            # Use AI to generate improved code
            prompt = f"""Improve this code by fixing: {pattern.description}

File: {file_path}
Pattern found: {pattern.replacement_template}

Code section:
```python
{content}
```

Generate the complete improved code that:
1. Fixes the identified pattern
2. Maintains all functionality
3. Improves {pattern.impact}
4. Follows Python best practices

Return only the improved code.
"""
            
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            if response.get('content'):
                evolved_code = self._extract_code(response['content'])
                
                # Calculate metrics
                metrics_before = await self._calculate_code_metrics(content)
                metrics_after = await self._calculate_code_metrics(evolved_code)
                
                return CodeEvolution(
                    file_path=file_path,
                    original_code=content,
                    evolved_code=evolved_code,
                    description=f"Fix: {pattern.description}",
                    improvements=[pattern.replacement_template],
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    confidence=pattern.confidence
                )
                
        except Exception as e:
            self.logger.error(f"Failed to evolve for pattern: {e}")
            
        return None
        
    async def _evolve_for_complexity(self, file_path: str) -> Optional[CodeEvolution]:
        """Refactor complex code."""
        try:
            content = Path(file_path).read_text()
            
            # Find most complex function
            tree = ast.parse(content)
            most_complex = None
            max_complexity = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    if complexity > max_complexity:
                        max_complexity = complexity
                        most_complex = node
                        
            if not most_complex:
                return None
                
            # Extract function code
            func_lines = content.splitlines()[most_complex.lineno-1:most_complex.end_lineno]
            func_code = '\n'.join(func_lines)
            
            # Use AI to refactor
            prompt = f"""Refactor this complex function (complexity: {max_complexity}) into simpler, more maintainable code:

```python
{func_code}
```

Requirements:
1. Break into smaller functions
2. Reduce nesting levels
3. Improve readability
4. Maintain exact functionality
5. Add clear docstrings

Return the refactored code with any helper functions.
"""
            
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            if response.get('content'):
                refactored = self._extract_code(response['content'])
                
                # Replace in original code
                evolved_code = content.replace(func_code, refactored)
                
                metrics_before = await self._calculate_code_metrics(content)
                metrics_after = await self._calculate_code_metrics(evolved_code)
                
                return CodeEvolution(
                    file_path=file_path,
                    original_code=content,
                    evolved_code=evolved_code,
                    description=f"Refactor complex function: {most_complex.name}",
                    improvements=['Reduced complexity', 'Improved readability'],
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    confidence=0.8
                )
                
        except Exception as e:
            self.logger.error(f"Failed to evolve for complexity: {e}")
            
        return None
        
    async def _evolve_for_suggestion(self, suggestion: Dict[str, Any]) -> Optional[CodeEvolution]:
        """Implement an AI suggestion."""
        try:
            file_path = suggestion['file']
            if not Path(file_path).exists():
                # Find most likely file
                possible_files = list(Path('scripts').rglob(f"*{Path(file_path).name}"))
                if possible_files:
                    file_path = str(possible_files[0])
                else:
                    return None
                    
            content = Path(file_path).read_text()
            
            # Generate evolution
            prompt = f"""Implement this improvement:

Change: {suggestion['description']}
Impact: {suggestion['impact']}
Approach: {suggestion['approach']}

Current code:
```python
{content}
```

Generate the improved code following the suggested approach.
"""
            
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            if response.get('content'):
                evolved_code = self._extract_code(response['content'])
                
                metrics_before = await self._calculate_code_metrics(content)
                metrics_after = await self._calculate_code_metrics(evolved_code)
                
                return CodeEvolution(
                    file_path=file_path,
                    original_code=content,
                    evolved_code=evolved_code,
                    description=suggestion['description'],
                    improvements=[suggestion['impact']],
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    confidence=0.7
                )
                
        except Exception as e:
            self.logger.error(f"Failed to evolve for suggestion: {e}")
            
        return None
        
    def _extract_code(self, response: str) -> str:
        """Extract code from AI response."""
        # Find code blocks
        if '```python' in response:
            match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif '```' in response:
            match = re.search(r'```\n(.*?)```', response, re.DOTALL)
            if match:
                return match.group(1).strip()
                
        # If no code blocks, assume entire response is code
        return response.strip()
        
    async def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate metrics for code."""
        metrics = {
            'lines': len(code.splitlines()),
            'complexity': 0,
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'docstring_coverage': 0
        }
        
        try:
            tree = ast.parse(code)
            
            functions_with_docstrings = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['functions'] += 1
                    metrics['complexity'] += self._calculate_complexity(node)
                    if ast.get_docstring(node):
                        functions_with_docstrings += 1
                elif isinstance(node, ast.ClassDef):
                    metrics['classes'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics['imports'] += 1
                    
            if metrics['functions'] > 0:
                metrics['avg_complexity'] = metrics['complexity'] / metrics['functions']
                metrics['docstring_coverage'] = functions_with_docstrings / metrics['functions']
                
        except:
            pass
            
        return metrics
        
    async def _apply_evolutions(self, evolutions: List[CodeEvolution]) -> List[CodeEvolution]:
        """Apply evolutions to code."""
        applied = []
        
        for evolution in evolutions:
            try:
                # Test if we should
                if self.config['test_before_apply']:
                    if not await self._test_evolution(evolution):
                        self.logger.warning(f"Evolution failed testing: {evolution.description}")
                        continue
                        
                # Check improvement threshold
                if not self._meets_improvement_threshold(evolution):
                    print(f"Evolution doesn't meet threshold: {evolution.description}")
                    continue
                    
                # Apply the evolution
                print(f"Applying evolution: {evolution.description}")
                
                # Backup original
                backup_path = Path(f"{evolution.file_path}.backup")
                backup_path.write_text(evolution.original_code)
                
                # Apply evolution
                Path(evolution.file_path).write_text(evolution.evolved_code)
                
                # Mark as applied
                evolution.applied = True
                applied.append(evolution)
                
                # Update metrics
                self.improvement_metrics['total_evolutions'] += 1
                self.improvement_metrics['successful_evolutions'] += 1
                
                if 'performance' in evolution.description.lower():
                    self.improvement_metrics['performance_improvements'] += 1
                elif 'reliability' in evolution.description.lower():
                    self.improvement_metrics['reliability_improvements'] += 1
                else:
                    self.improvement_metrics['maintainability_improvements'] += 1
                    
                # Store in history
                self.evolution_history.append(evolution)
                
                # Clean up backup if successful
                backup_path.unlink()
                
            except Exception as e:
                self.logger.error(f"Failed to apply evolution: {e}")
                # Restore from backup if exists
                backup_path = Path(f"{evolution.file_path}.backup")
                if backup_path.exists():
                    Path(evolution.file_path).write_text(backup_path.read_text())
                    backup_path.unlink()
                    
        return applied
        
    async def _test_evolution(self, evolution: CodeEvolution) -> bool:
        """Test if evolution maintains functionality."""
        # Simple syntax check
        try:
            ast.parse(evolution.evolved_code)
        except SyntaxError:
            return False
            
        # Import check
        test_file = Path(f"{evolution.file_path}.test")
        try:
            test_file.write_text(evolution.evolved_code)
            
            # Try to import
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            test_file.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"Evolution test failed: {e}")
            if test_file.exists():
                test_file.unlink()
            return False
            
    def _meets_improvement_threshold(self, evolution: CodeEvolution) -> bool:
        """Check if evolution meets improvement threshold."""
        before = evolution.metrics_before
        after = evolution.metrics_after
        
        # Complexity improvement
        if before.get('avg_complexity', 0) > 0:
            complexity_improvement = (before['avg_complexity'] - after.get('avg_complexity', before['avg_complexity'])) / before['avg_complexity']
            if complexity_improvement > self.config['performance_improvement_threshold']:
                return True
                
        # Docstring coverage improvement
        doc_improvement = after.get('docstring_coverage', 0) - before.get('docstring_coverage', 0)
        if doc_improvement > 0.2:
            return True
            
        # Line reduction (simpler code)
        if after['lines'] < before['lines'] * 0.8:
            return True
            
        # Trust high confidence evolutions
        if evolution.confidence >= 0.85:
            return True
            
        return False
        
    async def _learn_from_evolutions(self, applied: List[CodeEvolution]):
        """Learn from applied evolutions."""
        for evolution in applied:
            # Extract patterns from successful evolutions
            patterns = await self._extract_evolution_patterns(evolution)
            
            for pattern in patterns:
                # Check if pattern is new
                is_new = True
                for known in self.discovered_patterns:
                    if known.description == pattern.description:
                        # Update confidence
                        known.confidence = min(1.0, known.confidence + 0.05)
                        is_new = False
                        break
                        
                if is_new:
                    self.discovered_patterns.append(pattern)
                    print(f"Discovered new pattern: {pattern.description}")
                    
        # Save learnings
        await self._save_evolution_state()
        
    async def _extract_evolution_patterns(self, evolution: CodeEvolution) -> List[CodePattern]:
        """Extract reusable patterns from evolution."""
        patterns = []
        
        # Generate diff
        diff = list(difflib.unified_diff(
            evolution.original_code.splitlines(),
            evolution.evolved_code.splitlines(),
            lineterm=''
        ))
        
        # Analyze changes
        prompt = f"""Analyze this code evolution and extract reusable patterns:

Evolution: {evolution.description}
Improvements: {json.dumps(evolution.improvements)}

Diff:
{chr(10).join(diff[:100])}  # First 100 lines

Extract patterns that could be applied elsewhere:
1. What was the problem pattern?
2. What is the solution pattern?
3. How to detect similar cases?
4. What is the general fix?

Return as JSON list of patterns.
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            extracted = json.loads(response.get('content', '[]'))
            for item in extracted:
                patterns.append(CodePattern(
                    pattern_type='discovered',
                    description=item.get('problem', ''),
                    detection_regex=item.get('detection', ''),
                    ast_pattern=None,
                    occurrences=[],
                    replacement_template=item.get('solution', ''),
                    impact=evolution.improvements[0].split()[0].lower() if evolution.improvements else 'general',
                    confidence=0.6
                ))
        except:
            pass
            
        return patterns
        
    async def _save_evolution_state(self):
        """Save evolution state."""
        state = {
            'discovered_patterns': [
                {
                    'description': p.description,
                    'impact': p.impact,
                    'confidence': p.confidence,
                    'occurrences_count': len(p.occurrences)
                }
                for p in self.discovered_patterns
            ],
            'improvement_metrics': self.improvement_metrics,
            'recent_evolutions': [
                {
                    'file': e.file_path,
                    'description': e.description,
                    'improvements': e.improvements,
                    'applied': e.applied
                }
                for e in self.evolution_history[-20:]
            ],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        state_path = Path('code_evolution_state.json')
        state_path.write_text(json.dumps(state, indent=2))
        
    async def suggest_evolution(self, file_path: str) -> Optional[CodeEvolution]:
        """Suggest evolution for a specific file."""
        # Analyze file
        patterns = await self._scan_file(file_path)
        
        if patterns:
            # Use highest priority pattern
            pattern = max(patterns, key=lambda p: p.confidence)
            return await self._evolve_for_pattern(file_path, pattern)
            
        # Try complexity reduction
        return await self._evolve_for_complexity(file_path)
        
    async def _scan_file(self, file_path: str) -> List[CodePattern]:
        """Scan a single file for patterns."""
        patterns = []
        
        try:
            content = Path(file_path).read_text()
            
            for pattern in self.known_patterns + self.discovered_patterns:
                if pattern.detection_regex:
                    if re.search(pattern.detection_regex, content):
                        patterns.append(pattern)
                        
            # AST patterns
            patterns.extend(await self._detect_ast_patterns(Path(file_path), content))
            
        except Exception as e:
            self.logger.error(f"Failed to scan {file_path}: {e}")
            
        return patterns
        
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'total_patterns_known': len(self.known_patterns),
            'patterns_discovered': len(self.discovered_patterns),
            'evolutions_applied': self.improvement_metrics['successful_evolutions'],
            'performance_improvements': self.improvement_metrics['performance_improvements'],
            'reliability_improvements': self.improvement_metrics['reliability_improvements'],
            'maintainability_improvements': self.improvement_metrics['maintainability_improvements'],
            'success_rate': (
                self.improvement_metrics['successful_evolutions'] / 
                max(self.improvement_metrics['total_evolutions'], 1)
            )
        }


# Global instance
code_evolution_engine = None

def get_evolution_engine() -> CodeEvolutionEngine:
    """Get or create the global evolution engine."""
    global code_evolution_engine
    if code_evolution_engine is None:
        code_evolution_engine = CodeEvolutionEngine()
    return code_evolution_engine