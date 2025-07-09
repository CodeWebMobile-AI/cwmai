#!/usr/bin/env python3
"""
Comprehensive analysis of import usage in the codebase.
Properly handles scripts.module imports and traces the full dependency graph.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional
import json
import re

class ComprehensiveImportAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.zip_dir = self.root_dir / "zip"
        self.used_files: Set[Path] = set()
        self.import_graph: Dict[Path, Set[Path]] = {}
        self.errors: List[Tuple[Path, str]] = []
        self.import_mapping: Dict[str, Path] = {}  # Module name to file path mapping
        
        # Build initial mapping of available modules
        self._build_module_mapping()
        
    def _build_module_mapping(self):
        """Build a mapping of module names to file paths."""
        scripts_dir = self.zip_dir / "scripts"
        if scripts_dir.exists():
            for py_file in scripts_dir.glob("**/*.py"):
                if '__pycache__' not in str(py_file):
                    # Calculate module name
                    rel_path = py_file.relative_to(self.zip_dir)
                    module_parts = list(rel_path.parts[:-1])  # Remove .py file
                    module_parts.append(rel_path.stem)
                    module_name = '.'.join(module_parts)
                    self.import_mapping[module_name] = py_file
                    
                    # Also add without 'scripts.' prefix for direct imports
                    if module_parts[0] == 'scripts' and len(module_parts) > 1:
                        alt_module_name = '.'.join(module_parts[1:])
                        self.import_mapping[alt_module_name] = py_file
        
    def analyze(self):
        """Main analysis function."""
        # Start from the main entry point
        entry_point = self.zip_dir / "run_hybrid_continuous.py"
        
        print(f"Starting comprehensive analysis from: {entry_point}")
        print(f"Found {len(self.import_mapping)} Python modules in the codebase")
        print("=" * 80)
        
        # Trace all imports recursively
        self._trace_imports(entry_point)
        
        # Find all Python files in the project
        all_python_files = self._find_all_python_files()
        
        # Identify unused files
        unused_files = all_python_files - self.used_files
        
        # Create and save detailed report
        report = self._create_detailed_report(unused_files, all_python_files)
        
        with open('comprehensive_unused_files_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Print results
        self._print_results(unused_files, all_python_files)
        
    def _trace_imports(self, file_path: Path, visited: Set[Path] = None, depth: int = 0):
        """Recursively trace all imports from a file."""
        if visited is None:
            visited = set()
            
        if file_path in visited or not file_path.exists():
            return
            
        visited.add(file_path)
        self.used_files.add(file_path)
        
        # Debug print for first few levels
        if depth < 3:
            indent = "  " * depth
            print(f"{indent}Tracing: {file_path.relative_to(self.zip_dir)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract imports
            imports = self._extract_imports(tree, file_path, content)
            self.import_graph[file_path] = imports
            
            # Recursively trace each import
            for imported_file in imports:
                self._trace_imports(imported_file, visited, depth + 1)
                
        except Exception as e:
            self.errors.append((file_path, str(e)))
            
    def _extract_imports(self, tree: ast.AST, current_file: Path, content: str) -> Set[Path]:
        """Extract all imports from an AST."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_import(alias.name, current_file)
                    if resolved:
                        imports.add(resolved)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle 'from module import ...'
                    resolved = self._resolve_import(node.module, current_file)
                    if resolved:
                        imports.add(resolved)
                        
                    # Also check if specific imports are submodules
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        sub_resolved = self._resolve_import(full_name, current_file)
                        if sub_resolved:
                            imports.add(sub_resolved)
                            
        # Also check for dynamic imports
        dynamic_imports = self._find_dynamic_imports(content)
        for module_name in dynamic_imports:
            resolved = self._resolve_import(module_name, current_file)
            if resolved:
                imports.add(resolved)
                
        return imports
        
    def _find_dynamic_imports(self, content: str) -> Set[str]:
        """Find dynamic imports in code."""
        imports = set()
        
        # Patterns for dynamic imports
        patterns = [
            r'import_module\([\'"]([^\'"\)]+)[\'"]\)',
            r'__import__\([\'"]([^\'"\)]+)[\'"]\)',
            r'importlib\.import_module\([\'"]([^\'"\)]+)[\'"]\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)
            
        return imports
        
    def _resolve_import(self, module_name: str, current_file: Path) -> Optional[Path]:
        """Resolve a module name to a file path."""
        # Handle relative imports
        if module_name.startswith('.'):
            # TODO: Handle relative imports properly
            return None
            
        # Check direct mapping
        if module_name in self.import_mapping:
            return self.import_mapping[module_name]
            
        # Try with .py extension
        potential_paths = [
            self.zip_dir / f"{module_name.replace('.', '/')}.py",
            self.zip_dir / module_name.replace('.', '/') / "__init__.py",
        ]
        
        # Special handling for scripts imports
        if not module_name.startswith('scripts.') and '.' in module_name:
            # Try prefixing with scripts
            scripts_module = f"scripts.{module_name}"
            if scripts_module in self.import_mapping:
                return self.import_mapping[scripts_module]
                
        for path in potential_paths:
            if path.exists():
                return path
                
        return None
        
    def _find_all_python_files(self) -> Set[Path]:
        """Find all Python files in the project."""
        python_files = set()
        
        # Include all Python files except test/debug/fix files
        for py_file in self.zip_dir.glob("**/*.py"):
            if not self._should_exclude(py_file):
                python_files.add(py_file)
                    
        return python_files
        
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from analysis."""
        exclude_patterns = [
            '__pycache__',
            'test_',
            'debug_',
            'fix_',
            'example',
            '.backup',
            'monitor_',
            'cleanup_',
            'check_',
            'setup',
            'install',
            'migration',
            'patch_',
            'apply_',
            'update_',
            'convert_',
            'analyze_',
            'verify_',
            'recover_',
            'reset_',
            'clear_',
            'trigger_',
            'force_',
            'start_',
            'stop_',
            'kill_',
            'remove_',
            'purge_',
            'download_',
            'sync_',
            'clone_',
            'compare_',
            'benchmark',
            'demo_',
            'simple_',
            'quick_',
            'temp_',
            'old_',
            'backup_',
            'migrate_',
            'run_',  # Exclude other run scripts
            'detailed_unused_files_report.py',
            'analyze_unused_files.py',
            'comprehensive_import_analyzer.py',  # Exclude self
            'local_repos/',  # Exclude local repository copies
            'utilities/',
            '.git/',
            'venv/',
            'env/',
            '__init__.py'  # Usually empty
        ]
        
        path_str = str(file_path).lower()
        file_name = file_path.name.lower()
        
        # Don't exclude the main entry point
        if file_name == 'run_hybrid_continuous.py' and 'utilities' not in path_str:
            return False
            
        return any(pattern in path_str for pattern in exclude_patterns)
        
    def _create_detailed_report(self, unused_files: Set[Path], all_files: Set[Path]) -> Dict:
        """Create a detailed report of unused files."""
        report = {
            'summary': {
                'total_files': len(all_files),
                'used_files': len(self.used_files),
                'unused_files': len(unused_files),
                'errors': len(self.errors)
            },
            'unused_files_by_category': {},
            'all_unused_files': [],
            'potentially_important_unused': [],
            'used_files': [],
            'import_graph_sample': {},
            'errors': []
        }
        
        # Categorize unused files
        important_patterns = [
            'engine', 'system', 'manager', 'handler', 'processor',
            'analyzer', 'generator', 'builder', 'factory', 'service',
            'optimizer', 'coordinator', 'intelligence', 'discovery'
        ]
        
        for file_path in sorted(unused_files):
            rel_path = file_path.relative_to(self.zip_dir)
            dir_name = str(rel_path.parent)
            
            if dir_name not in report['unused_files_by_category']:
                report['unused_files_by_category'][dir_name] = []
                
            file_info = {
                'path': str(rel_path),
                'name': file_path.name,
                'size': file_path.stat().st_size if file_path.exists() else 0
            }
            
            report['unused_files_by_category'][dir_name].append(file_info)
            report['all_unused_files'].append(str(rel_path))
            
            # Check if potentially important
            file_name_lower = file_path.stem.lower()
            if any(pattern in file_name_lower for pattern in important_patterns):
                report['potentially_important_unused'].append(str(rel_path))
                
        # Add used files
        for file_path in sorted(self.used_files):
            rel_path = file_path.relative_to(self.zip_dir)
            report['used_files'].append(str(rel_path))
            
        # Add sample of import graph (first 10 files)
        for file_path, imports in list(self.import_graph.items())[:10]:
            rel_path = str(file_path.relative_to(self.zip_dir))
            imported_files = [str(f.relative_to(self.zip_dir)) for f in imports]
            report['import_graph_sample'][rel_path] = imported_files
            
        # Add errors
        for file_path, error in self.errors:
            report['errors'].append({
                'file': str(file_path.relative_to(self.zip_dir)),
                'error': error
            })
            
        return report
        
    def _print_results(self, unused_files: Set[Path], all_files: Set[Path]):
        """Print analysis results."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE IMPORT ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"\nTotal Python files found: {len(all_files)}")
        print(f"Used files (imported): {len(self.used_files)}")
        print(f"Unused files: {len(unused_files)}")
        print(f"Import graph entries: {len(self.import_graph)}")
        
        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for file_path, error in self.errors[:5]:
                print(f"  - {file_path.relative_to(self.zip_dir)}: {error}")
                
        # Show sample of import graph
        print("\n" + "=" * 80)
        print("SAMPLE IMPORT GRAPH (first 5 files)")
        print("=" * 80)
        
        for file_path, imports in list(self.import_graph.items())[:5]:
            rel_path = file_path.relative_to(self.zip_dir)
            print(f"\n{rel_path}:")
            for imp in sorted(imports):
                print(f"  → {imp.relative_to(self.zip_dir)}")
                
        print("\n" + "=" * 80)
        print("UNUSED FILES BY CATEGORY")
        print("=" * 80)
        
        # Group by directory
        unused_by_dir: Dict[str, List[Path]] = {}
        for file_path in unused_files:
            dir_name = str(file_path.parent.relative_to(self.zip_dir))
            if dir_name not in unused_by_dir:
                unused_by_dir[dir_name] = []
            unused_by_dir[dir_name].append(file_path)
            
        # Show summary by directory
        for dir_name, files in sorted(unused_by_dir.items()):
            if files:  # Only show directories with unused files
                total_size = sum(f.stat().st_size for f in files if f.exists()) / 1024
                print(f"\n{dir_name}/ ({len(files)} files, {total_size:.1f} KB)")
                
                # Show first 10 files
                for file_path in sorted(files)[:10]:
                    print(f"  - {file_path.name}")
                    
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")
                    
        # Identify potentially important unused files
        print("\n" + "=" * 80)
        print("POTENTIALLY IMPORTANT UNUSED FILES")
        print("=" * 80)
        
        important_patterns = [
            'engine', 'system', 'manager', 'handler', 'processor',
            'analyzer', 'generator', 'builder', 'factory', 'service',
            'optimizer', 'coordinator', 'intelligence', 'discovery'
        ]
        
        important_unused = []
        for file_path in unused_files:
            file_name = file_path.stem.lower()
            if any(pattern in file_name for pattern in important_patterns):
                important_unused.append(file_path)
                
        if important_unused:
            print(f"\nFound {len(important_unused)} potentially important unused files:")
            for file_path in sorted(important_unused)[:20]:
                rel_path = file_path.relative_to(self.zip_dir)
                print(f"  - {rel_path}")
                
            if len(important_unused) > 20:
                print(f"  ... and {len(important_unused) - 20} more")
        else:
            print("  None found")
            
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_unused_size = sum(f.stat().st_size for f in unused_files if f.exists()) / 1024 / 1024
        used_percentage = (len(self.used_files) / len(all_files) * 100) if all_files else 0
        
        print(f"\n📊 Code Usage Statistics:")
        print(f"  - {used_percentage:.1f}% of Python files are actively used")
        print(f"  - {len(unused_files)} files can potentially be removed")
        print(f"  - This would save approximately {total_unused_size:.2f} MB")
        print(f"\n💡 Recommendations:")
        print(f"  1. Review 'potentially important' files before removal")
        print(f"  2. Consider archiving unused files instead of deleting")
        print(f"  3. Some files may be used by external scripts or tools")
        print(f"  4. Check for files that might be imported dynamically")
        
        print(f"\n📄 Full report saved to: comprehensive_unused_files_report.json")


def main():
    """Main entry point."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Run analysis
    analyzer = ComprehensiveImportAnalyzer(script_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()