"""
AI-Generated Tool: dependency_manager
Description: Manage npm, composer, and other package dependencies for repositories
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

__description__ = "Manage package dependencies for repositories - install, update, audit, add/remove packages"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["install", "update", "add", "remove", "audit", "list", "outdated", "clean"],
        "description": "Dependency operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "package_name": {
        "type": "string",
        "required": False,
        "description": "Package name for add/remove operations"
    },
    "version": {
        "type": "string",
        "required": False,
        "description": "Package version (e.g., '^1.0.0', 'latest')"
    },
    "dev": {
        "type": "boolean",
        "required": False,
        "default": False,
        "description": "Whether to add as dev dependency"
    },
    "manager": {
        "type": "string",
        "required": False,
        "enum": ["npm", "yarn", "composer", "auto"],
        "default": "auto",
        "description": "Package manager to use"
    }
}
__examples__ = [
    "dependency_manager(operation='install', repo_name='my-app')",
    "dependency_manager(operation='add', repo_name='my-app', package_name='axios', version='^1.0.0')",
    "dependency_manager(operation='update', repo_name='my-app')",
    "dependency_manager(operation='audit', repo_name='my-app')",
    "dependency_manager(operation='list', repo_name='my-app')"
]

# Base path
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"

async def dependency_manager(**kwargs) -> Dict[str, Any]:
    """
    Manage package dependencies for repositories.
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
    
    # Detect package manager
    manager = kwargs.get('manager', 'auto')
    if manager == 'auto':
        manager = await _detect_package_manager(repo_path)
        if not manager:
            return {
                "status": "error",
                "message": "No package manager detected (no package.json or composer.json found)"
            }
    
    # Route to appropriate operation
    operations = {
        'install': install_dependencies,
        'update': update_dependencies,
        'add': add_package,
        'remove': remove_package,
        'audit': audit_dependencies,
        'list': list_dependencies,
        'outdated': check_outdated,
        'clean': clean_dependencies
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        kwargs['repo_path'] = repo_path
        kwargs['manager'] = manager
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def install_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Install all dependencies."""
    repo_path = params['repo_path']
    manager = params['manager']
    
    commands = {
        'npm': ['npm', 'install'],
        'yarn': ['yarn', 'install'],
        'composer': ['composer', 'install']
    }
    
    if manager not in commands:
        return {
            "status": "error",
            "message": f"Unsupported package manager: {manager}"
        }
    
    try:
        start_time = datetime.now()
        
        result = subprocess.run(
            commands[manager],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to install dependencies: {result.stderr}",
                "command": ' '.join(commands[manager])
            }
        
        duration = (datetime.now() - start_time).seconds
        
        # Count installed packages
        package_count = await _count_packages(repo_path, manager)
        
        return {
            "status": "success",
            "message": "Dependencies installed successfully",
            "manager": manager,
            "packages_installed": package_count,
            "duration_seconds": duration,
            "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Installation failed: {e.stderr}"
        }

async def update_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Update all dependencies."""
    repo_path = params['repo_path']
    manager = params['manager']
    
    commands = {
        'npm': ['npm', 'update'],
        'yarn': ['yarn', 'upgrade'],
        'composer': ['composer', 'update']
    }
    
    if manager not in commands:
        return {
            "status": "error",
            "message": f"Unsupported package manager: {manager}"
        }
    
    try:
        # Check for outdated packages first
        outdated = await check_outdated(params)
        
        result = subprocess.run(
            commands[manager],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to update dependencies: {result.stderr}"
            }
        
        return {
            "status": "success",
            "message": "Dependencies updated successfully",
            "manager": manager,
            "outdated_before": outdated.get('total_outdated', 0),
            "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Update failed: {e.stderr}"
        }

async def add_package(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add a new package."""
    repo_path = params['repo_path']
    manager = params['manager']
    package_name = params.get('package_name')
    version = params.get('version', 'latest')
    is_dev = params.get('dev', False)
    
    if not package_name:
        return {
            "status": "error",
            "message": "package_name is required for add operation"
        }
    
    # Format package with version
    package_spec = f"{package_name}@{version}" if version != 'latest' else package_name
    
    # Build command
    if manager == 'npm':
        cmd = ['npm', 'install', package_spec]
        if is_dev:
            cmd.append('--save-dev')
    elif manager == 'yarn':
        cmd = ['yarn', 'add', package_spec]
        if is_dev:
            cmd.append('--dev')
    elif manager == 'composer':
        cmd = ['composer', 'require', package_spec]
        if is_dev:
            cmd.insert(2, '--dev')
    else:
        return {
            "status": "error",
            "message": f"Unsupported package manager: {manager}"
        }
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to add package: {result.stderr}",
                "command": ' '.join(cmd)
            }
        
        # Get installed version
        installed_version = await _get_package_version(repo_path, manager, package_name)
        
        return {
            "status": "success",
            "message": f"Package '{package_name}' added successfully",
            "package": package_name,
            "version": installed_version or version,
            "dev_dependency": is_dev,
            "manager": manager
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Failed to add package: {e.stderr}"
        }

async def remove_package(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a package."""
    repo_path = params['repo_path']
    manager = params['manager']
    package_name = params.get('package_name')
    
    if not package_name:
        return {
            "status": "error",
            "message": "package_name is required for remove operation"
        }
    
    commands = {
        'npm': ['npm', 'uninstall', package_name],
        'yarn': ['yarn', 'remove', package_name],
        'composer': ['composer', 'remove', package_name]
    }
    
    if manager not in commands:
        return {
            "status": "error",
            "message": f"Unsupported package manager: {manager}"
        }
    
    try:
        result = subprocess.run(
            commands[manager],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to remove package: {result.stderr}"
            }
        
        return {
            "status": "success",
            "message": f"Package '{package_name}' removed successfully",
            "package": package_name,
            "manager": manager
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Failed to remove package: {e.stderr}"
        }

async def audit_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Audit dependencies for security vulnerabilities."""
    repo_path = params['repo_path']
    manager = params['manager']
    
    if manager == 'npm':
        try:
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            audit_data = json.loads(result.stdout) if result.stdout else {}
            
            vulnerabilities = audit_data.get('metadata', {}).get('vulnerabilities', {})
            
            return {
                "status": "success",
                "manager": manager,
                "vulnerabilities": {
                    "critical": vulnerabilities.get('critical', 0),
                    "high": vulnerabilities.get('high', 0),
                    "moderate": vulnerabilities.get('moderate', 0),
                    "low": vulnerabilities.get('low', 0),
                    "total": vulnerabilities.get('total', 0)
                },
                "summary": audit_data.get('metadata', {}).get('summary', 'No vulnerabilities found')
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Audit failed: {str(e)}"
            }
            
    elif manager == 'yarn':
        try:
            result = subprocess.run(
                ['yarn', 'audit', '--json'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            # Parse yarn audit output (NDJSON format)
            vulnerabilities = {
                "critical": 0,
                "high": 0,
                "moderate": 0,
                "low": 0,
                "total": 0
            }
            
            for line in result.stdout.splitlines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'auditSummary':
                            vulns = data.get('data', {}).get('vulnerabilities', {})
                            vulnerabilities.update(vulns)
                            vulnerabilities['total'] = sum(vulns.values())
                    except:
                        pass
            
            return {
                "status": "success",
                "manager": manager,
                "vulnerabilities": vulnerabilities
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Audit failed: {str(e)}"
            }
    
    elif manager == 'composer':
        return {
            "status": "info",
            "message": "Composer doesn't have built-in audit. Use 'composer audit' if available or external tools.",
            "manager": manager
        }
    
    return {
        "status": "error",
        "message": f"Audit not supported for {manager}"
    }

async def list_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all dependencies."""
    repo_path = params['repo_path']
    manager = params['manager']
    
    dependencies = {
        "production": {},
        "development": {}
    }
    
    if manager in ['npm', 'yarn']:
        package_json_path = repo_path / 'package.json'
        if package_json_path.exists():
            package_data = json.loads(package_json_path.read_text())
            dependencies['production'] = package_data.get('dependencies', {})
            dependencies['development'] = package_data.get('devDependencies', {})
            
    elif manager == 'composer':
        composer_json_path = repo_path / 'composer.json'
        if composer_json_path.exists():
            composer_data = json.loads(composer_json_path.read_text())
            dependencies['production'] = composer_data.get('require', {})
            dependencies['development'] = composer_data.get('require-dev', {})
    
    total_deps = len(dependencies['production']) + len(dependencies['development'])
    
    return {
        "status": "success",
        "manager": manager,
        "total_dependencies": total_deps,
        "production_count": len(dependencies['production']),
        "development_count": len(dependencies['development']),
        "dependencies": dependencies
    }

async def check_outdated(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for outdated packages."""
    repo_path = params['repo_path']
    manager = params['manager']
    
    outdated_packages = []
    
    if manager == 'npm':
        try:
            result = subprocess.run(
                ['npm', 'outdated', '--json'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                outdated_data = json.loads(result.stdout)
                for pkg, info in outdated_data.items():
                    outdated_packages.append({
                        "package": pkg,
                        "current": info.get('current', 'unknown'),
                        "wanted": info.get('wanted', 'unknown'),
                        "latest": info.get('latest', 'unknown'),
                        "type": info.get('type', 'dependencies')
                    })
        except:
            pass
            
    elif manager == 'yarn':
        try:
            result = subprocess.run(
                ['yarn', 'outdated', '--json'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.splitlines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'table' and data.get('data', {}).get('body'):
                            for row in data['data']['body']:
                                outdated_packages.append({
                                    "package": row[0],
                                    "current": row[1],
                                    "wanted": row[2],
                                    "latest": row[3],
                                    "type": row[4]
                                })
                    except:
                        pass
        except:
            pass
            
    elif manager == 'composer':
        try:
            result = subprocess.run(
                ['composer', 'outdated', '--format=json'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                outdated_data = json.loads(result.stdout)
                for pkg in outdated_data.get('installed', []):
                    if pkg.get('latest-status') == 'outdated':
                        outdated_packages.append({
                            "package": pkg.get('name'),
                            "current": pkg.get('version'),
                            "latest": pkg.get('latest'),
                            "type": "require"
                        })
        except:
            pass
    
    return {
        "status": "success",
        "manager": manager,
        "total_outdated": len(outdated_packages),
        "outdated_packages": outdated_packages
    }

async def clean_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and reinstall dependencies."""
    repo_path = params['repo_path']
    manager = params['manager']
    
    # Delete lock files and node_modules/vendor
    if manager in ['npm', 'yarn']:
        node_modules = repo_path / 'node_modules'
        if node_modules.exists():
            import shutil
            shutil.rmtree(node_modules)
            
        # Remove lock files
        (repo_path / 'package-lock.json').unlink(missing_ok=True)
        (repo_path / 'yarn.lock').unlink(missing_ok=True)
        
    elif manager == 'composer':
        vendor = repo_path / 'vendor'
        if vendor.exists():
            import shutil
            shutil.rmtree(vendor)
            
        (repo_path / 'composer.lock').unlink(missing_ok=True)
    
    # Reinstall
    install_result = await install_dependencies(params)
    
    return {
        "status": "success" if install_result['status'] == 'success' else "error",
        "message": "Dependencies cleaned and reinstalled",
        "manager": manager,
        "install_result": install_result
    }

async def _detect_package_manager(repo_path: Path) -> Optional[str]:
    """Detect which package manager is used."""
    if (repo_path / 'package.json').exists():
        if (repo_path / 'yarn.lock').exists():
            return 'yarn'
        return 'npm'
    elif (repo_path / 'composer.json').exists():
        return 'composer'
    return None

async def _count_packages(repo_path: Path, manager: str) -> int:
    """Count total installed packages."""
    count = 0
    
    if manager in ['npm', 'yarn']:
        node_modules = repo_path / 'node_modules'
        if node_modules.exists():
            # Count direct subdirectories
            count = len([d for d in node_modules.iterdir() if d.is_dir() and not d.name.startswith('.')])
    elif manager == 'composer':
        vendor = repo_path / 'vendor'
        if vendor.exists():
            # Count vendor packages (rough estimate)
            for org_dir in vendor.iterdir():
                if org_dir.is_dir() and not org_dir.name.startswith('.'):
                    count += len([d for d in org_dir.iterdir() if d.is_dir()])
    
    return count

async def _get_package_version(repo_path: Path, manager: str, package_name: str) -> Optional[str]:
    """Get installed version of a package."""
    if manager in ['npm', 'yarn']:
        package_json_path = repo_path / 'package.json'
        if package_json_path.exists():
            package_data = json.loads(package_json_path.read_text())
            version = package_data.get('dependencies', {}).get(package_name)
            if not version:
                version = package_data.get('devDependencies', {}).get(package_name)
            return version
    elif manager == 'composer':
        composer_lock = repo_path / 'composer.lock'
        if composer_lock.exists():
            lock_data = json.loads(composer_lock.read_text())
            for pkg in lock_data.get('packages', []) + lock_data.get('packages-dev', []):
                if pkg.get('name') == package_name:
                    return pkg.get('version')
    return None

# Test function
async def main():
    # Test dependency operations
    result = await dependency_manager(
        operation='list',
        repo_name='brand-guardian-ai'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())