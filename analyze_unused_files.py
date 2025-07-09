#!/usr/bin/env python3
"""
Analyze the codebase to find unused Python files.
Starts from run_hybrid_continuous.py and traces all imports.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional
import re

class ImportAnalyzer:
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
        
        print(f"Starting analysis from: {entry_point}")
        print("=" * 80)
        
        # Trace all imports recursively
        self._trace_imports(entry_point)
        
        # Find all Python files in the project
        all_python_files = self._find_all_python_files()
        
        # Identify unused files
        unused_files = all_python_files - self.used_files
        
        # Categorize results
        self._print_results(unused_files, all_python_files)
        
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
            'run_',  # Exclude other run scripts
            'monitor_',
            'cleanup_',
            'check_',
            'setup',
            'install',
            'migration',
            'patch_'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in exclude_patterns)
        
    def _print_results(self, unused_files: Set[Path], all_files: Set[Path]):
        """Print analysis results."""
        # Group unused files by directory
        unused_by_dir: Dict[str, List[Path]] = {}
        
        for file_path in sorted(unused_files):
            rel_path = file_path.relative_to(self.zip_dir)
            dir_name = str(rel_path.parent)
            
            if dir_name not in unused_by_dir:
                unused_by_dir[dir_name] = []
            unused_by_dir[dir_name].append(file_path)
            
        # Print results
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"\nTotal Python files found: {len(all_files)}")
        print(f"Used files: {len(self.used_files)}")
        print(f"Unused files: {len(unused_files)}")
        
        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for file_path, error in self.errors[:5]:
                print(f"  - {file_path.relative_to(self.zip_dir)}: {error}")
                
        print("\n" + "=" * 80)
        print("UNUSED FILES BY CATEGORY")
        print("=" * 80)
        
        # Print unused files grouped by directory
        for dir_name, files in sorted(unused_by_dir.items()):
            print(f"\n{dir_name}/ ({len(files)} files):")
            for file_path in sorted(files)[:10]:  # Limit to first 10 per directory
                rel_path = file_path.relative_to(self.zip_dir)
                print(f"  - {rel_path.name}")
                
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
                
        # Identify potentially important unused files
        print("\n" + "=" * 80)
        print("POTENTIALLY IMPORTANT UNUSED FILES")
        print("=" * 80)
        
        important_patterns = [
            'engine', 'system', 'manager', 'handler', 'processor',
            'analyzer', 'generator', 'builder', 'factory', 'service'
        ]
        
        important_unused = []
        for file_path in unused_files:
            file_name = file_path.stem.lower()
            if any(pattern in file_name for pattern in important_patterns):
                important_unused.append(file_path)
                
        if important_unused:
            for file_path in sorted(important_unused)[:20]:
                rel_path = file_path.relative_to(self.zip_dir)
                print(f"  - {rel_path}")
        else:
            print("  None found")
            
        # Summary recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        total_unused_size = sum(f.stat().st_size for f in unused_files if f.exists()) / 1024 / 1024
        print(f"\n1. {len(unused_files)} unused files can be removed")
        print(f"2. This would save approximately {total_unused_size:.2f} MB")
        print("3. Consider archiving instead of deleting in case they contain useful code")
        print("4. Review 'potentially important' files before removal")


def main():
    """Main entry point."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Run analysis
    analyzer = ImportAnalyzer(script_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()