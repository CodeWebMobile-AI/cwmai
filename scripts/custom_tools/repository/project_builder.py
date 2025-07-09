"""
AI-Generated Tool: project_builder
Description: Build, compile, and manage project build processes
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

__description__ = "Build and compile projects - run build scripts, compile assets, run tests"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["build", "dev", "test", "lint", "format", "compile", "clean", "watch"],
        "description": "Build operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "target": {
        "type": "string",
        "required": False,
        "enum": ["production", "development", "test"],
        "default": "development",
        "description": "Build target environment"
    },
    "watch": {
        "type": "boolean",
        "required": False,
        "default": False,
        "description": "Watch for changes and rebuild"
    },
    "verbose": {
        "type": "boolean",
        "required": False,
        "default": False,
        "description": "Show verbose output"
    }
}
__examples__ = [
    "project_builder(operation='build', repo_name='my-app', target='production')",
    "project_builder(operation='dev', repo_name='my-app')",
    "project_builder(operation='test', repo_name='my-app')",
    "project_builder(operation='lint', repo_name='my-app')",
    "project_builder(operation='watch', repo_name='my-app')"
]

# Base path
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"

async def project_builder(**kwargs) -> Dict[str, Any]:
    """
    Build and compile projects.
    """
    operation = kwargs.get('operation')
    repo_name = kwargs.get('repo_name')
    
    if not operation:
        return {
            "status": "error",
            "message": "operation parameter is required"
        }
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name parameter is required"
        }
    
    # Check if repository exists
    repo_path = REPOSITORIES_PATH / repo_name
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    # Detect project type
    project_type = await _detect_project_type(repo_path)
    if not project_type:
        return {
            "status": "error",
            "message": "Unable to detect project type (no package.json or composer.json found)"
        }
    
    # Route to appropriate operation
    operations = {
        'build': build_project,
        'dev': run_dev_mode,
        'test': run_tests,
        'lint': run_lint,
        'format': run_format,
        'compile': compile_assets,
        'clean': clean_build,
        'watch': watch_project
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        kwargs['repo_path'] = repo_path
        kwargs['project_type'] = project_type
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def build_project(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the project for production or development."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    target = params.get('target', 'development')
    verbose = params.get('verbose', False)
    
    # Determine build command based on project type
    build_commands = await _get_build_commands(repo_path, project_type)
    
    if not build_commands.get('build'):
        return {
            "status": "error",
            "message": "No build script found in package.json or composer.json"
        }
    
    # Set environment
    env = os.environ.copy()
    env['NODE_ENV'] = 'production' if target == 'production' else 'development'
    
    try:
        start_time = datetime.now()
        
        # Run pre-build tasks if any
        if build_commands.get('prebuild'):
            subprocess.run(
                build_commands['prebuild'],
                cwd=repo_path,
                env=env,
                shell=True,
                check=True
            )
        
        # Run build
        result = subprocess.run(
            build_commands['build'],
            cwd=repo_path,
            env=env,
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": "Build failed",
                "error": result.stderr,
                "command": build_commands['build']
            }
        
        duration = (datetime.now() - start_time).seconds
        
        # Check build output
        build_info = await _analyze_build_output(repo_path, project_type)
        
        return {
            "status": "success",
            "message": f"Build completed successfully for {target}",
            "target": target,
            "duration_seconds": duration,
            "build_info": build_info,
            "output": result.stdout[-1000:] if verbose else "Use verbose=True to see output"
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Build process failed: {str(e)}"
        }

async def run_dev_mode(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run project in development mode."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    
    build_commands = await _get_build_commands(repo_path, project_type)
    
    if not build_commands.get('dev'):
        return {
            "status": "error",
            "message": "No dev script found"
        }
    
    return {
        "status": "info",
        "message": "Development mode would start the dev server",
        "command": build_commands['dev'],
        "note": "Use development_server_manager to actually run dev servers"
    }

async def run_tests(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run project tests."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    verbose = params.get('verbose', False)
    
    build_commands = await _get_build_commands(repo_path, project_type)
    
    if not build_commands.get('test'):
        return {
            "status": "error",
            "message": "No test script found"
        }
    
    try:
        result = subprocess.run(
            build_commands['test'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        
        # Parse test results
        test_summary = _parse_test_output(result.stdout, project_type)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "message": "Tests completed" if result.returncode == 0 else "Tests failed",
            "test_summary": test_summary,
            "exit_code": result.returncode,
            "output": result.stdout if verbose else test_summary
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Test execution failed: {str(e)}"
        }

async def run_lint(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run linting on the project."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    
    build_commands = await _get_build_commands(repo_path, project_type)
    
    if not build_commands.get('lint'):
        # Try to detect linter
        linter_cmd = await _detect_linter(repo_path, project_type)
        if not linter_cmd:
            return {
                "status": "error",
                "message": "No lint script or linter found"
            }
    else:
        linter_cmd = build_commands['lint']
    
    try:
        result = subprocess.run(
            linter_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        
        # Parse lint results
        lint_summary = _parse_lint_output(result.stdout, result.stderr)
        
        return {
            "status": "success" if result.returncode == 0 else "warning",
            "message": "Linting completed",
            "issues_found": lint_summary['total_issues'],
            "lint_summary": lint_summary,
            "fixable": lint_summary.get('fixable_issues', 0)
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Linting failed: {str(e)}"
        }

async def run_format(params: Dict[str, Any]) -> Dict[str, Any]:
    """Format code in the project."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    
    build_commands = await _get_build_commands(repo_path, project_type)
    
    formatter_cmd = build_commands.get('format')
    if not formatter_cmd:
        # Try to detect formatter
        formatter_cmd = await _detect_formatter(repo_path, project_type)
        if not formatter_cmd:
            return {
                "status": "error",
                "message": "No format script or formatter found"
            }
    
    try:
        result = subprocess.run(
            formatter_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        
        return {
            "status": "success",
            "message": "Code formatting completed",
            "files_formatted": _count_formatted_files(result.stdout),
            "formatter": _detect_formatter_name(formatter_cmd)
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Formatting failed: {str(e)}"
        }

async def compile_assets(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compile project assets (CSS, JS, etc)."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    
    # Check for asset compilation tools
    if project_type == 'laravel':
        # Laravel Mix/Vite
        if (repo_path / 'vite.config.js').exists():
            cmd = 'npm run build'
        elif (repo_path / 'webpack.mix.js').exists():
            cmd = 'npm run production'
        else:
            return {
                "status": "error",
                "message": "No asset compilation configuration found"
            }
    else:
        # Generic build
        build_commands = await _get_build_commands(repo_path, project_type)
        cmd = build_commands.get('build')
        if not cmd:
            return {
                "status": "error",
                "message": "No build script found for asset compilation"
            }
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": "Asset compilation failed",
                "error": result.stderr
            }
        
        # Check compiled assets
        compiled_assets = await _check_compiled_assets(repo_path, project_type)
        
        return {
            "status": "success",
            "message": "Assets compiled successfully",
            "compiled_assets": compiled_assets
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Asset compilation failed: {str(e)}"
        }

async def clean_build(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean build artifacts."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    
    cleaned = []
    
    # Common build directories to clean
    build_dirs = ['dist', 'build', 'out', '.next', 'public/build', 'public/js', 'public/css']
    
    for dir_name in build_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            import shutil
            shutil.rmtree(dir_path)
            cleaned.append(dir_name)
    
    # Clean cache directories
    cache_dirs = ['.cache', '.parcel-cache', 'node_modules/.cache']
    for dir_name in cache_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists():
            import shutil
            shutil.rmtree(dir_path)
            cleaned.append(dir_name)
    
    return {
        "status": "success",
        "message": "Build artifacts cleaned",
        "cleaned_directories": cleaned,
        "total_cleaned": len(cleaned)
    }

async def watch_project(params: Dict[str, Any]) -> Dict[str, Any]:
    """Set up file watching for automatic rebuilds."""
    repo_path = params['repo_path']
    project_type = params['project_type']
    
    build_commands = await _get_build_commands(repo_path, project_type)
    
    watch_cmd = build_commands.get('watch') or build_commands.get('dev')
    
    if not watch_cmd:
        return {
            "status": "error",
            "message": "No watch script found"
        }
    
    return {
        "status": "info",
        "message": "Watch mode would monitor files for changes",
        "command": watch_cmd,
        "note": "Use development_server_manager to run watch mode with proper process management"
    }

async def _detect_project_type(repo_path: Path) -> Optional[str]:
    """Detect the type of project."""
    if (repo_path / 'composer.json').exists():
        composer_data = json.loads((repo_path / 'composer.json').read_text())
        if 'laravel/framework' in composer_data.get('require', {}):
            return 'laravel'
        return 'php'
    
    if (repo_path / 'package.json').exists():
        package_data = json.loads((repo_path / 'package.json').read_text())
        deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
        
        if 'next' in deps:
            return 'nextjs'
        elif 'react' in deps:
            return 'react'
        elif 'vue' in deps:
            return 'vue'
        elif '@angular/core' in deps:
            return 'angular'
        else:
            return 'node'
    
    return None

async def _get_build_commands(repo_path: Path, project_type: str) -> Dict[str, str]:
    """Get available build commands for the project."""
    commands = {}
    
    if project_type in ['laravel', 'php']:
        # PHP projects might have npm scripts too
        if (repo_path / 'package.json').exists():
            package_data = json.loads((repo_path / 'package.json').read_text())
            scripts = package_data.get('scripts', {})
            commands.update(_extract_npm_scripts(scripts))
    else:
        # Node.js projects
        if (repo_path / 'package.json').exists():
            package_data = json.loads((repo_path / 'package.json').read_text())
            scripts = package_data.get('scripts', {})
            commands.update(_extract_npm_scripts(scripts))
    
    return commands

def _extract_npm_scripts(scripts: Dict[str, str]) -> Dict[str, str]:
    """Extract relevant npm scripts."""
    commands = {}
    
    # Map common script names
    script_mapping = {
        'build': 'build',
        'dev': 'dev',
        'develop': 'dev',
        'start': 'dev',
        'test': 'test',
        'lint': 'lint',
        'format': 'format',
        'prettier': 'format',
        'watch': 'watch',
        'prebuild': 'prebuild'
    }
    
    for script_name, command_key in script_mapping.items():
        if script_name in scripts:
            commands[command_key] = f"npm run {script_name}"
    
    return commands

async def _analyze_build_output(repo_path: Path, project_type: str) -> Dict[str, Any]:
    """Analyze build output directory."""
    build_info = {
        "output_directory": None,
        "files_generated": 0,
        "total_size_mb": 0
    }
    
    # Check common output directories
    output_dirs = ['dist', 'build', 'out', '.next', 'public/build']
    
    for dir_name in output_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            build_info['output_directory'] = dir_name
            
            # Count files and size
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_count += 1
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size
            
            build_info['files_generated'] = file_count
            build_info['total_size_mb'] = round(total_size / 1024 / 1024, 2)
            break
    
    return build_info

def _parse_test_output(output: str, project_type: str) -> Dict[str, Any]:
    """Parse test output to extract summary."""
    summary = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # Try to parse Jest output
    if 'Tests:' in output:
        for line in output.splitlines():
            if line.strip().startswith('Tests:'):
                # Parse Jest summary line
                parts = line.split(',')
                for part in parts:
                    if 'passed' in part:
                        summary['passed'] = int(''.join(filter(str.isdigit, part)))
                    elif 'failed' in part:
                        summary['failed'] = int(''.join(filter(str.isdigit, part)))
                    elif 'skipped' in part:
                        summary['skipped'] = int(''.join(filter(str.isdigit, part)))
    
    summary['total_tests'] = summary['passed'] + summary['failed'] + summary['skipped']
    
    return summary

def _parse_lint_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """Parse linting output."""
    summary = {
        "total_issues": 0,
        "errors": 0,
        "warnings": 0,
        "fixable_issues": 0
    }
    
    output = stdout + stderr
    
    # Try to parse ESLint output
    if 'problems' in output:
        for line in output.splitlines():
            if 'error' in line and 'warning' in line:
                # Parse summary line
                import re
                errors = re.search(r'(\d+)\s+errors?', line)
                warnings = re.search(r'(\d+)\s+warnings?', line)
                
                if errors:
                    summary['errors'] = int(errors.group(1))
                if warnings:
                    summary['warnings'] = int(warnings.group(1))
    
    summary['total_issues'] = summary['errors'] + summary['warnings']
    
    return summary

async def _detect_linter(repo_path: Path, project_type: str) -> Optional[str]:
    """Detect available linter."""
    # Check for ESLint
    if (repo_path / '.eslintrc.js').exists() or (repo_path / '.eslintrc.json').exists():
        return 'npx eslint .'
    
    # Check for PHP linters
    if project_type in ['laravel', 'php']:
        if (repo_path / 'phpcs.xml').exists():
            return 'vendor/bin/phpcs'
    
    return None

async def _detect_formatter(repo_path: Path, project_type: str) -> Optional[str]:
    """Detect available code formatter."""
    # Check for Prettier
    if (repo_path / '.prettierrc').exists() or (repo_path / '.prettierrc.json').exists():
        return 'npx prettier --write .'
    
    # Check for PHP formatters
    if project_type in ['laravel', 'php']:
        if (repo_path / '.php-cs-fixer.php').exists():
            return 'vendor/bin/php-cs-fixer fix'
    
    return None

def _detect_formatter_name(cmd: str) -> str:
    """Detect formatter name from command."""
    if 'prettier' in cmd:
        return 'Prettier'
    elif 'php-cs-fixer' in cmd:
        return 'PHP CS Fixer'
    elif 'eslint' in cmd and '--fix' in cmd:
        return 'ESLint'
    return 'Unknown'

def _count_formatted_files(output: str) -> int:
    """Count number of files formatted."""
    # Simple heuristic - count lines that look like file paths
    count = 0
    for line in output.splitlines():
        if '.' in line and ('/' in line or '\\' in line):
            count += 1
    return count

async def _check_compiled_assets(repo_path: Path, project_type: str) -> List[str]:
    """Check what assets were compiled."""
    assets = []
    
    # Check common asset directories
    asset_dirs = [
        ('public/js', '*.js'),
        ('public/css', '*.css'),
        ('dist', '*'),
        ('build', '*')
    ]
    
    for dir_name, pattern in asset_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists():
            for file in dir_path.glob(pattern):
                if file.is_file():
                    assets.append(str(file.relative_to(repo_path)))
    
    return assets[:20]  # Limit to 20 files

# Test function
async def main():
    # Test build operations
    result = await project_builder(
        operation='build',
        repo_name='brand-guardian-ai',
        target='development'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())