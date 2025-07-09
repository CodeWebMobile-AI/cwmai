#!/usr/bin/env python3
"""
Create a detailed report of unused files in the codebase.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional
import json

class DetailedImportAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.zip_dir = self.root_dir / "zip"
        self.used_files: Set[Path] = set()
        self.import_graph: Dict[Path, Set[Path]] = {}
        self.errors: List[Tuple[Path, str]] = []
        
    def analyze(self):
        """Main analysis function."""
        # Start from the main entry point
        entry_point = self.zip_dir / "run_hybrid_continuous.py"
        
        # Trace all imports recursively
        self._trace_imports(entry_point)
        
        # Find all Python files in the project
        all_python_files = self._find_all_python_files()
        
        # Identify unused files
        unused_files = all_python_files - self.used_files
        
        # Create detailed report
        report = self._create_detailed_report(unused_files, all_python_files)
        
        # Save report to file
        with open('unused_files_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Print summary and full list
        self._print_detailed_results(report)
        
    def _trace_imports(self, file_path: Path, visited: Set[Path] = None):
        """Recursively trace all imports from a file."""
        if visited is None:
            visited = set()
            
        if file_path in visited or not file_path.exists():
            return
            
        visited.add(file_path)
        self.used_files.add(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content)
            
            # Extract imports
            imports = self._extract_imports(tree, file_path)
            self.import_graph[file_path] = imports
            
            # Recursively trace each import
            for imported_file in imports:
                self._trace_imports(imported_file, visited)
                
        except Exception as e:
            self.errors.append((file_path, str(e)))
            
    def _extract_imports(self, tree: ast.AST, current_file: Path) -> Set[Path]:
        """Extract all imports from an AST."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_path = self._resolve_import(alias.name, current_file)
                    if module_path:
                        imports.add(module_path)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_path = self._resolve_import(node.module, current_file)
                    if module_path:
                        imports.add(module_path)
                        
        # Also check for dynamic imports in strings
        dynamic_imports = self._find_dynamic_imports(content)
        for module_name in dynamic_imports:
            module_path = self._resolve_import(module_name, current_file)
            if module_path:
                imports.add(module_path)
                
        return imports
        
    def _find_dynamic_imports(self, content: str) -> Set[str]:
        """Find dynamic imports in code (e.g., importlib, __import__)."""
        imports = set()
        
        # Look for patterns like importlib.import_module('module.name')
        import re
        patterns = [
            r'import_module\([\'"]([^\'"\)]+)[\'"]\)',
            r'__import__\([\'"]([^\'"\)]+)[\'"]\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)
            
        return imports
        
    def _resolve_import(self, module_name: str, current_file: Path) -> Optional[Path]:
        """Resolve a module name to a file path."""
        # Handle relative imports
        if module_name.startswith('.'):
            return None  # Skip relative imports for now
            
        # Convert module name to path
        parts = module_name.split('.')
        
        # Check if it's a scripts import
        if parts[0] == 'scripts':
            # Look in zip/scripts directory
            path_parts = parts[1:]
            potential_paths = [
                self.zip_dir / 'scripts' / ('/'.join(path_parts) + '.py'),
                self.zip_dir / 'scripts' / '/'.join(path_parts) / '__init__.py'
            ]
            
            for path in potential_paths:
                if path.exists():
                    return path
                    
        # Check if it's a direct file in zip
        potential_path = self.zip_dir / (module_name.replace('.', '/') + '.py')
        if potential_path.exists():
            return potential_path
            
        return None
        
    def _find_all_python_files(self) -> Set[Path]:
        """Find all Python files in the project."""
        python_files = set()
        
        # Focus on scripts directory and root level Python files
        for pattern in ['*.py', 'scripts/**/*.py']:
            for file_path in self.zip_dir.glob(pattern):
                if file_path.is_file() and not self._should_exclude(file_path):
                    python_files.add(file_path)
                    
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
            'run_',  # Exclude other run scripts except main entry
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
            'utilities/'  # Utility scripts that might not be imported
        ]
        
        path_str = str(file_path)
        file_name = file_path.name
        
        # Don't exclude the main entry point
        if file_name == 'run_hybrid_continuous.py':
            return False
            
        return any(pattern in path_str.lower() for pattern in exclude_patterns)
        
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
            'used_files': [],
            'errors': []
        }
        
        # Group unused files by directory
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
            
        # Add used files
        for file_path in sorted(self.used_files):
            rel_path = file_path.relative_to(self.zip_dir)
            report['used_files'].append(str(rel_path))
            
        # Add errors
        for file_path, error in self.errors:
            report['errors'].append({
                'file': str(file_path.relative_to(self.zip_dir)),
                'error': error
            })
            
        return report
        
    def _print_detailed_results(self, report: Dict):
        """Print detailed results."""
        print("\n" + "=" * 80)
        print("DETAILED UNUSED FILES REPORT")
        print("=" * 80)
        
        print(f"\nSummary:")
        print(f"  Total Python files: {report['summary']['total_files']}")
        print(f"  Used files: {report['summary']['used_files']}")
        print(f"  Unused files: {report['summary']['unused_files']}")
        
        print("\n" + "=" * 80)
        print("ALL UNUSED FILES (Complete List)")
        print("=" * 80)
        
        # Print all unused files sorted
        for i, file_path in enumerate(sorted(report['all_unused_files']), 1):
            print(f"{i:3d}. {file_path}")
            
        print("\n" + "=" * 80)
        print("UNUSED FILES GROUPED BY DIRECTORY")
        print("=" * 80)
        
        for dir_name, files in sorted(report['unused_files_by_category'].items()):
            total_size = sum(f['size'] for f in files) / 1024  # KB
            print(f"\n{dir_name}/ ({len(files)} files, {total_size:.1f} KB):")
            for file_info in sorted(files, key=lambda x: x['name']):
                size_kb = file_info['size'] / 1024
                print(f"  - {file_info['name']} ({size_kb:.1f} KB)")
                
        print("\n" + "=" * 80)
        print("Report saved to: unused_files_report.json")
        print("=" * 80)


def main():
    """Main entry point."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Run analysis
    analyzer = DetailedImportAnalyzer(script_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()