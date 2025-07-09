import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import ast
import asyncio
from collections import defaultdict
import networkx as nx

from scripts.ai_brain import AIBrain
from scripts.redis_integration import get_redis_client
from scripts.code_analyzer import CodeAnalyzer
from scripts.pattern_learning_engine import PatternLearningEngine
from scripts.knowledge_synthesis_engine import KnowledgeSynthesisEngine

logger = logging.getLogger(__name__)


class AutoArchitectureEvolution:
    """Automatically evolve system architecture for better maintainability and performance"""
    
    def __init__(self):
        self.ai_brain = AIBrain()
        self.redis_client = get_redis_client()
        self.code_analyzer = CodeAnalyzer()
        self.pattern_engine = PatternLearningEngine()
        self.synthesis_engine = KnowledgeSynthesisEngine()
        self.architecture_state_file = "architecture_evolution_state.json"
        self.dependency_graph = nx.DiGraph()
        self.load_architecture_state()
        
    def load_architecture_state(self):
        """Load current architecture understanding"""
        if Path(self.architecture_state_file).exists():
            with open(self.architecture_state_file, 'r') as f:
                self.architecture_state = json.load(f)
        else:
            self.architecture_state = {
                "modules": {},
                "dependencies": {},
                "complexity_scores": {},
                "refactoring_history": [],
                "architecture_patterns": {},
                "last_analysis": datetime.now().isoformat()
            }
    
    def save_architecture_state(self):
        """Persist architecture state"""
        self.architecture_state["last_analysis"] = datetime.now().isoformat()
        with open(self.architecture_state_file, 'w') as f:
            json.dump(self.architecture_state, f, indent=2)
    
    async def analyze_architecture(self) -> Dict[str, Any]:
        """Analyze current system architecture"""
        logger.info("Analyzing system architecture for improvement opportunities")
        
        # Scan codebase structure
        codebase_structure = await self._scan_codebase_structure()
        
        # Build dependency graph
        dependencies = await self._analyze_dependencies(codebase_structure)
        
        # Calculate complexity metrics
        complexity_metrics = await self._calculate_complexity_metrics(codebase_structure)
        
        # Identify architectural smells
        arch_smells = self._identify_architectural_smells(dependencies, complexity_metrics)
        
        # Find improvement opportunities
        improvements = await self._find_improvement_opportunities(arch_smells)
        
        # Update architecture state
        self.architecture_state["modules"] = codebase_structure
        self.architecture_state["dependencies"] = dependencies
        self.architecture_state["complexity_scores"] = complexity_metrics
        
        self.save_architecture_state()
        
        return {
            "structure": codebase_structure,
            "dependencies": dependencies,
            "complexity": complexity_metrics,
            "architectural_smells": arch_smells,
            "improvements": improvements
        }
    
    async def evolve_architecture(self) -> Dict[str, Any]:
        """Automatically refactor and improve architecture"""
        logger.info("Evolving system architecture")
        
        # Get current architecture analysis
        analysis = await self.analyze_architecture()
        
        # Prioritize improvements
        prioritized_improvements = self._prioritize_improvements(analysis["improvements"])
        
        # Generate refactoring plans
        refactoring_plans = []
        for improvement in prioritized_improvements[:5]:  # Top 5 improvements
            plan = await self._generate_refactoring_plan(improvement)
            if plan:
                refactoring_plans.append(plan)
        
        # Apply safe refactorings
        applied_refactorings = []
        for plan in refactoring_plans:
            if plan["risk_level"] == "low":
                result = await self._apply_refactoring(plan)
                if result:
                    applied_refactorings.append(result)
                    
        # Update architecture patterns
        new_patterns = await self._extract_architecture_patterns(applied_refactorings)
        self.architecture_state["architecture_patterns"].update(new_patterns)
        
        # Record refactoring history
        self.architecture_state["refactoring_history"].extend([
            {
                "timestamp": datetime.now().isoformat(),
                "refactoring": ref["type"],
                "impact": ref.get("impact", "unknown")
            }
            for ref in applied_refactorings
        ])
        
        self.save_architecture_state()
        
        return {
            "analysis": analysis,
            "refactoring_plans": refactoring_plans,
            "applied_refactorings": applied_refactorings,
            "new_patterns": new_patterns
        }
    
    async def _scan_codebase_structure(self) -> Dict[str, Any]:
        """Scan and understand codebase structure"""
        structure = {
            "modules": {},
            "packages": {},
            "total_files": 0,
            "total_lines": 0
        }
        
        # Scan Python files
        for py_file in Path("scripts").rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Parse AST
                tree = ast.parse(content)
                
                module_info = {
                    "path": str(py_file),
                    "classes": [],
                    "functions": [],
                    "imports": [],
                    "lines": len(content.splitlines()),
                    "complexity": 0
                }
                
                # Extract module components
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        module_info["classes"].append({
                            "name": node.name,
                            "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            "lines": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        })
                    elif isinstance(node, ast.FunctionDef):
                        module_info["functions"].append({
                            "name": node.name,
                            "args": len(node.args.args),
                            "lines": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        })
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            module_info["imports"].append(alias.name)
                            
                # Calculate cyclomatic complexity
                module_info["complexity"] = self._calculate_cyclomatic_complexity(tree)
                
                module_name = py_file.stem
                structure["modules"][module_name] = module_info
                structure["total_files"] += 1
                structure["total_lines"] += module_info["lines"]
                
                # Group by package
                package = py_file.parent.name
                if package not in structure["packages"]:
                    structure["packages"][package] = []
                structure["packages"][package].append(module_name)
                
            except Exception as e:
                logger.error(f"Error analyzing {py_file}: {e}")
                
        return structure
    
    async def _analyze_dependencies(self, structure: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze module dependencies"""
        dependencies = defaultdict(list)
        
        # Build dependency graph
        for module_name, module_info in structure.get("modules", {}).items():
            imports = module_info.get("imports", [])
            
            for imp in imports:
                # Internal dependencies
                if imp.startswith("scripts."):
                    dep_module = imp.split(".")[-1]
                    if dep_module in structure["modules"]:
                        dependencies[module_name].append(dep_module)
                        self.dependency_graph.add_edge(module_name, dep_module)
                # External dependencies
                else:
                    dependencies[module_name].append(f"external:{imp}")
                    
        return dict(dependencies)
    
    async def _calculate_complexity_metrics(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various complexity metrics"""
        metrics = {}
        
        for module_name, module_info in structure.get("modules", {}).items():
            metrics[module_name] = {
                "cyclomatic_complexity": module_info.get("complexity", 0),
                "lines_of_code": module_info.get("lines", 0),
                "number_of_classes": len(module_info.get("classes", [])),
                "number_of_functions": len(module_info.get("functions", [])),
                "dependency_count": len(module_info.get("imports", [])),
                "cohesion_score": self._calculate_cohesion(module_info),
                "coupling_score": self._calculate_coupling(module_name)
            }
            
        # Calculate aggregate metrics
        metrics["aggregate"] = {
            "total_complexity": sum(m["cyclomatic_complexity"] for m in metrics.values() if isinstance(m, dict)),
            "average_complexity": sum(m["cyclomatic_complexity"] for m in metrics.values() if isinstance(m, dict)) / len(metrics) if metrics else 0,
            "highest_complexity_modules": sorted(
                [(k, v["cyclomatic_complexity"]) for k, v in metrics.items() if isinstance(v, dict)],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
        return metrics
    
    def _identify_architectural_smells(self, dependencies: Dict[str, List[str]], 
                                     complexity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify architectural anti-patterns and smells"""
        smells = []
        
        # God Class/Module (too complex)
        for module, metrics in complexity.items():
            if isinstance(metrics, dict) and metrics.get("cyclomatic_complexity", 0) > 20:
                smells.append({
                    "type": "god_module",
                    "module": module,
                    "severity": "high",
                    "details": f"Complexity: {metrics['cyclomatic_complexity']}"
                })
                
        # Circular Dependencies
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    smells.append({
                        "type": "circular_dependency",
                        "modules": cycle,
                        "severity": "high",
                        "details": f"Circular dependency: {' -> '.join(cycle)}"
                    })
        except:
            pass
            
        # High Coupling
        for module, deps in dependencies.items():
            if len(deps) > 10:
                smells.append({
                    "type": "high_coupling",
                    "module": module,
                    "severity": "medium",
                    "details": f"Depends on {len(deps)} modules"
                })
                
        # Low Cohesion
        for module, metrics in complexity.items():
            if isinstance(metrics, dict) and metrics.get("cohesion_score", 1) < 0.3:
                smells.append({
                    "type": "low_cohesion",
                    "module": module,
                    "severity": "medium",
                    "details": "Module has unrelated responsibilities"
                })
                
        # Dead Code (no dependencies)
        all_modules = set(complexity.keys()) - {"aggregate"}
        referenced_modules = set()
        for deps in dependencies.values():
            referenced_modules.update(d for d in deps if not d.startswith("external:"))
            
        dead_modules = all_modules - referenced_modules
        for module in dead_modules:
            if module not in ["__init__", "test_"]:  # Skip init and test files
                smells.append({
                    "type": "dead_code",
                    "module": module,
                    "severity": "low",
                    "details": "Module not referenced anywhere"
                })
                
        return smells
    
    async def _find_improvement_opportunities(self, smells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find specific improvement opportunities based on smells"""
        improvements = []
        
        for smell in smells:
            if smell["type"] == "god_module":
                # Suggest splitting
                prompt = f"""
                Module {smell['module']} is too complex ({smell['details']}).
                Suggest how to split it into smaller, focused modules.
                
                Return as JSON: {{
                    "improvement": "split_module",
                    "target": "module_name",
                    "suggested_modules": ["module1", "module2"],
                    "rationale": "explanation"
                }}
                """
                
                result = await self.ai_brain.process(prompt)
                try:
                    improvement = json.loads(result)
                    improvement["addresses"] = smell
                    improvements.append(improvement)
                except:
                    pass
                    
            elif smell["type"] == "circular_dependency":
                improvements.append({
                    "improvement": "break_circular_dependency",
                    "target": smell["modules"],
                    "suggestion": "Introduce interface or move shared code to new module",
                    "addresses": smell
                })
                
            elif smell["type"] == "high_coupling":
                improvements.append({
                    "improvement": "reduce_coupling",
                    "target": smell["module"],
                    "suggestion": "Use dependency injection or interfaces",
                    "addresses": smell
                })
                
            elif smell["type"] == "low_cohesion":
                improvements.append({
                    "improvement": "increase_cohesion",
                    "target": smell["module"],
                    "suggestion": "Group related functionality together",
                    "addresses": smell
                })
                
        return improvements
    
    def _prioritize_improvements(self, improvements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize improvements by impact and risk"""
        for improvement in improvements:
            # Calculate priority score
            severity_scores = {"high": 3, "medium": 2, "low": 1}
            base_score = severity_scores.get(
                improvement.get("addresses", {}).get("severity", "low"), 1
            )
            
            # Adjust based on improvement type
            if improvement["improvement"] == "break_circular_dependency":
                base_score *= 2  # High priority
            elif improvement["improvement"] == "split_module":
                base_score *= 1.5  # Important but risky
                
            improvement["priority_score"] = base_score
            
        # Sort by priority
        return sorted(improvements, key=lambda x: x["priority_score"], reverse=True)
    
    async def _generate_refactoring_plan(self, improvement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate detailed refactoring plan"""
        improvement_type = improvement["improvement"]
        
        plan = {
            "type": improvement_type,
            "target": improvement["target"],
            "steps": [],
            "risk_level": "medium",
            "estimated_impact": "medium",
            "validation": []
        }
        
        if improvement_type == "split_module":
            plan["steps"] = [
                "Analyze module dependencies",
                "Identify cohesive groups of functionality",
                "Create new module files",
                "Move related code to new modules",
                "Update imports in dependent modules",
                "Run tests to ensure functionality"
            ]
            plan["risk_level"] = "high"
            plan["validation"] = ["All tests pass", "No import errors"]
            
        elif improvement_type == "break_circular_dependency":
            plan["steps"] = [
                "Identify shared functionality",
                "Create interface or abstract base",
                "Refactor modules to use interface",
                "Remove direct circular imports",
                "Validate dependency graph"
            ]
            plan["risk_level"] = "medium"
            plan["validation"] = ["No circular dependencies", "Tests pass"]
            
        elif improvement_type == "reduce_coupling":
            plan["steps"] = [
                "Identify tightly coupled components",
                "Introduce abstraction layer",
                "Use dependency injection",
                "Reduce direct dependencies"
            ]
            plan["risk_level"] = "low"
            plan["validation"] = ["Reduced dependency count", "Maintained functionality"]
            
        plan["generated_at"] = datetime.now().isoformat()
        return plan
    
    async def _apply_refactoring(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply a refactoring plan (simulation for safety)"""
        # In a real implementation, this would perform actual refactoring
        # For now, we'll simulate and log the changes
        
        result = {
            "type": plan["type"],
            "target": plan["target"],
            "status": "simulated",  # Change to "applied" when implementing
            "changes": [],
            "impact": {}
        }
        
        # Simulate impact
        if plan["type"] == "reduce_coupling":
            result["impact"] = {
                "coupling_reduction": "20%",
                "maintainability_improvement": "15%"
            }
            result["changes"] = [
                "Introduced IToolExecutor interface",
                "Refactored 3 modules to use interface",
                "Removed 5 direct dependencies"
            ]
            
        elif plan["type"] == "break_circular_dependency":
            result["impact"] = {
                "circular_dependencies_removed": 1,
                "architecture_clarity": "+25%"
            }
            result["changes"] = [
                "Created shared base module",
                "Moved common functionality",
                "Updated imports in affected modules"
            ]
            
        logger.info(f"Simulated refactoring: {result}")
        return result
    
    async def _extract_architecture_patterns(self, refactorings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract reusable architecture patterns from successful refactorings"""
        patterns = {}
        
        for refactoring in refactorings:
            pattern_key = f"{refactoring['type']}_pattern"
            
            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    "description": f"Pattern for {refactoring['type']}",
                    "applications": [],
                    "success_rate": 1.0
                }
                
            patterns[pattern_key]["applications"].append({
                "target": refactoring["target"],
                "impact": refactoring.get("impact", {}),
                "timestamp": datetime.now().isoformat()
            })
            
        return patterns
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of AST"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _calculate_cohesion(self, module_info: Dict[str, Any]) -> float:
        """Calculate module cohesion score"""
        # Simple cohesion metric based on shared method calls
        classes = module_info.get("classes", [])
        functions = module_info.get("functions", [])
        
        if not classes and not functions:
            return 1.0  # Empty module is cohesive
            
        total_components = len(classes) + len(functions)
        if total_components <= 1:
            return 1.0  # Single component is cohesive
            
        # Estimate cohesion (simplified)
        # In reality, would analyze method calls and data sharing
        avg_methods_per_class = sum(len(c.get("methods", [])) for c in classes) / len(classes) if classes else 0
        
        if avg_methods_per_class > 10:
            return 0.3  # Low cohesion - too many responsibilities
        elif avg_methods_per_class > 5:
            return 0.6  # Medium cohesion
        else:
            return 0.9  # High cohesion
            
    def _calculate_coupling(self, module_name: str) -> float:
        """Calculate module coupling score"""
        if module_name not in self.dependency_graph:
            return 0.0
            
        # Coupling based on number of dependencies
        out_degree = self.dependency_graph.out_degree(module_name)
        in_degree = self.dependency_graph.in_degree(module_name)
        
        total_modules = len(self.dependency_graph.nodes())
        if total_modules <= 1:
            return 0.0
            
        # Normalize coupling score
        coupling = (out_degree + in_degree) / (2 * (total_modules - 1))
        return min(coupling, 1.0)