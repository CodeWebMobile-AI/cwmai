"""
AI-Generated Tool: manage_local_repository
Description: Full CRUD operations for managing repositories in local_repos
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

__description__ = "Create, read, update, delete and manage repositories in local_repos"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["create", "delete", "list", "rename", "archive", "unarchive", "info"],
        "description": "Operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": False,
        "description": "Repository name (required for most operations)"
    },
    "new_name": {
        "type": "string",
        "required": False,
        "description": "New name for rename operation"
    },
    "template": {
        "type": "string",
        "required": False,
        "default": "basic",
        "description": "Template to use for create operation (basic, laravel-react, nextjs)"
    },
    "description": {
        "type": "string",
        "required": False,
        "description": "Repository description for create operation"
    }
}
__examples__ = [
    "manage_local_repository(operation='list')",
    "manage_local_repository(operation='create', repo_name='my-new-app', template='laravel-react')",
    "manage_local_repository(operation='delete', repo_name='old-project')",
    "manage_local_repository(operation='rename', repo_name='old-name', new_name='new-name')",
    "manage_local_repository(operation='info', repo_name='my-app')"
]

# Base paths
LOCAL_REPOS_BASE = Path(__file__).parent.parent.parent.parent / "local_repos"
REPOSITORIES_PATH = LOCAL_REPOS_BASE / "repositories" / "CodeWebMobile-AI"
ARCHIVES_PATH = LOCAL_REPOS_BASE / "archives"
TEMPLATES_PATH = LOCAL_REPOS_BASE / "templates"

async def manage_local_repository(**kwargs) -> Dict[str, Any]:
    """
    Manage repositories in local_repos with full CRUD operations.
    """
    operation = kwargs.get('operation')
    repo_name = kwargs.get('repo_name')
    
    if not operation:
        return {
            "status": "error",
            "message": "operation parameter is required"
        }
    
    # Route to appropriate operation
    operations = {
        'create': create_repository,
        'delete': delete_repository,
        'list': list_repositories,
        'rename': rename_repository,
        'archive': archive_repository,
        'unarchive': unarchive_repository,
        'info': get_repository_info
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def create_repository(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new repository."""
    repo_name = params.get('repo_name')
    template = params.get('template', 'basic')
    description = params.get('description', '')
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name is required for create operation"
        }
    
    # Sanitize repo name
    repo_name = repo_name.lower().replace(' ', '-')
    repo_path = REPOSITORIES_PATH / repo_name
    
    if repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' already exists"
        }
    
    # Create repository directory
    repo_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize based on template
    files_created = []
    
    if template == 'basic':
        files_created = await _create_basic_template(repo_path, repo_name, description)
    elif template == 'laravel-react':
        files_created = await _create_laravel_react_template(repo_path, repo_name, description)
    elif template == 'nextjs':
        files_created = await _create_nextjs_template(repo_path, repo_name, description)
    else:
        # Use basic template as fallback
        files_created = await _create_basic_template(repo_path, repo_name, description)
    
    # Initialize git repository
    try:
        subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'add', '.'], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', f'Initial commit: {repo_name}'],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        git_initialized = True
    except:
        git_initialized = False
    
    # Create metadata
    metadata = {
        "name": repo_name,
        "description": description,
        "template": template,
        "created_at": datetime.now().isoformat(),
        "path": str(repo_path),
        "files": files_created
    }
    
    metadata_path = repo_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    
    return {
        "status": "success",
        "message": f"Repository '{repo_name}' created successfully",
        "repository": {
            "name": repo_name,
            "path": str(repo_path),
            "template": template,
            "files_created": len(files_created),
            "git_initialized": git_initialized
        },
        "next_steps": [
            f"cd {repo_path}",
            "npm install (if Node.js project)",
            "composer install (if PHP project)",
            "Create monetization strategy with generate_monetization_strategy tool"
        ]
    }

async def delete_repository(params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a repository (moves to archives)."""
    repo_name = params.get('repo_name')
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name is required for delete operation"
        }
    
    repo_path = REPOSITORIES_PATH / repo_name
    
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    # Archive instead of permanent deletion
    ARCHIVES_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{repo_name}_{timestamp}"
    archive_path = ARCHIVES_PATH / archive_name
    
    # Move to archives
    shutil.move(str(repo_path), str(archive_path))
    
    return {
        "status": "success",
        "message": f"Repository '{repo_name}' archived successfully",
        "archived_to": str(archive_path),
        "restore_command": f"manage_local_repository(operation='unarchive', repo_name='{archive_name}')"
    }

async def list_repositories(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all repositories."""
    repositories = []
    
    if REPOSITORIES_PATH.exists():
        for repo_dir in REPOSITORIES_PATH.iterdir():
            if repo_dir.is_dir():
                repo_info = await _get_repo_summary(repo_dir)
                repositories.append(repo_info)
    
    # Sort by name
    repositories.sort(key=lambda x: x['name'])
    
    return {
        "status": "success",
        "total_repositories": len(repositories),
        "repositories": repositories,
        "storage_path": str(REPOSITORIES_PATH)
    }

async def rename_repository(params: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a repository."""
    repo_name = params.get('repo_name')
    new_name = params.get('new_name')
    
    if not repo_name or not new_name:
        return {
            "status": "error",
            "message": "Both repo_name and new_name are required for rename operation"
        }
    
    # Sanitize new name
    new_name = new_name.lower().replace(' ', '-')
    
    old_path = REPOSITORIES_PATH / repo_name
    new_path = REPOSITORIES_PATH / new_name
    
    if not old_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    if new_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{new_name}' already exists"
        }
    
    # Rename directory
    old_path.rename(new_path)
    
    # Update metadata if exists
    metadata_path = new_path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata['name'] = new_name
        metadata['renamed_from'] = repo_name
        metadata['renamed_at'] = datetime.now().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2))
    
    return {
        "status": "success",
        "message": f"Repository renamed from '{repo_name}' to '{new_name}'",
        "old_name": repo_name,
        "new_name": new_name,
        "new_path": str(new_path)
    }

async def archive_repository(params: Dict[str, Any]) -> Dict[str, Any]:
    """Archive a repository (same as delete but explicit)."""
    return await delete_repository(params)

async def unarchive_repository(params: Dict[str, Any]) -> Dict[str, Any]:
    """Restore a repository from archives."""
    archive_name = params.get('repo_name')
    
    if not archive_name:
        return {
            "status": "error",
            "message": "repo_name (archive name) is required for unarchive operation"
        }
    
    archive_path = ARCHIVES_PATH / archive_name
    
    if not archive_path.exists():
        return {
            "status": "error",
            "message": f"Archive '{archive_name}' not found"
        }
    
    # Extract original repo name (remove timestamp)
    repo_name = '_'.join(archive_name.split('_')[:-2]) if '_' in archive_name else archive_name
    restore_path = REPOSITORIES_PATH / repo_name
    
    # Check if already exists
    if restore_path.exists():
        # Add suffix to avoid conflict
        suffix = 1
        while (REPOSITORIES_PATH / f"{repo_name}_{suffix}").exists():
            suffix += 1
        repo_name = f"{repo_name}_{suffix}"
        restore_path = REPOSITORIES_PATH / repo_name
    
    # Move back to repositories
    shutil.move(str(archive_path), str(restore_path))
    
    return {
        "status": "success",
        "message": f"Repository restored from archive",
        "repository_name": repo_name,
        "restored_to": str(restore_path)
    }

async def get_repository_info(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed information about a repository."""
    repo_name = params.get('repo_name')
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name is required for info operation"
        }
    
    repo_path = REPOSITORIES_PATH / repo_name
    
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    info = {
        "name": repo_name,
        "path": str(repo_path),
        "exists": True
    }
    
    # Get metadata
    metadata_path = repo_path / "metadata.json"
    if metadata_path.exists():
        info['metadata'] = json.loads(metadata_path.read_text())
    
    # Check for key files
    key_files = {
        'readme': 'README.md',
        'package_json': 'package.json',
        'composer_json': 'composer.json',
        'architecture': 'architecture.md',
        'monetization': 'monetization_strategy.md',
        'market_analysis': 'market_analysis.md'
    }
    
    info['files'] = {}
    for key, filename in key_files.items():
        file_path = repo_path / filename
        info['files'][key] = file_path.exists()
    
    # Get git status
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        info['git'] = {
            'initialized': True,
            'has_changes': bool(result.stdout.strip())
        }
    except:
        info['git'] = {'initialized': False}
    
    # Count files and calculate size
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        total_files += len(files)
        for file in files:
            if not file.startswith('.'):
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
    
    info['statistics'] = {
        'total_files': total_files,
        'total_size_mb': round(total_size / 1024 / 1024, 2)
    }
    
    return {
        "status": "success",
        "repository": info
    }

# Helper functions

async def _create_basic_template(repo_path: Path, repo_name: str, description: str) -> List[str]:
    """Create basic project template."""
    files = []
    
    # README.md
    readme_content = f"""# {repo_name}

{description if description else 'A new project'}

## Getting Started

1. Install dependencies
2. Configure environment
3. Run the application

## License

MIT
"""
    (repo_path / "README.md").write_text(readme_content)
    files.append("README.md")
    
    # .gitignore
    gitignore_content = """# Dependencies
node_modules/
vendor/

# Environment
.env
.env.local

# IDE
.idea/
.vscode/

# Logs
*.log

# OS
.DS_Store
"""
    (repo_path / ".gitignore").write_text(gitignore_content)
    files.append(".gitignore")
    
    return files

async def _create_laravel_react_template(repo_path: Path, repo_name: str, description: str) -> List[str]:
    """Create Laravel + React template."""
    files = await _create_basic_template(repo_path, repo_name, description)
    
    # package.json
    package_json = {
        "name": repo_name,
        "version": "1.0.0",
        "description": description,
        "scripts": {
            "dev": "vite",
            "build": "vite build"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0"
        },
        "devDependencies": {
            "@vitejs/plugin-react": "^4.0.0",
            "vite": "^4.4.0"
        }
    }
    (repo_path / "package.json").write_text(json.dumps(package_json, indent=2))
    files.append("package.json")
    
    # composer.json
    composer_json = {
        "name": f"codeweb/{repo_name}",
        "description": description,
        "require": {
            "php": "^8.1",
            "laravel/framework": "^10.0"
        }
    }
    (repo_path / "composer.json").write_text(json.dumps(composer_json, indent=2))
    files.append("composer.json")
    
    # Create directory structure
    (repo_path / "app").mkdir(exist_ok=True)
    (repo_path / "resources" / "js").mkdir(parents=True, exist_ok=True)
    (repo_path / "public").mkdir(exist_ok=True)
    
    return files

async def _create_nextjs_template(repo_path: Path, repo_name: str, description: str) -> List[str]:
    """Create Next.js template."""
    files = await _create_basic_template(repo_path, repo_name, description)
    
    # package.json
    package_json = {
        "name": repo_name,
        "version": "0.1.0",
        "description": description,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start"
        },
        "dependencies": {
            "next": "14.0.0",
            "react": "^18.2.0",
            "react-dom": "^18.2.0"
        },
        "devDependencies": {
            "@types/react": "^18.2.0",
            "typescript": "^5.0.0"
        }
    }
    (repo_path / "package.json").write_text(json.dumps(package_json, indent=2))
    files.append("package.json")
    
    # tsconfig.json
    tsconfig = {
        "compilerOptions": {
            "target": "es5",
            "lib": ["dom", "dom.iterable", "esnext"],
            "allowJs": True,
            "skipLibCheck": True,
            "strict": True,
            "forceConsistentCasingInFileNames": True,
            "noEmit": True,
            "esModuleInterop": True,
            "module": "esnext",
            "moduleResolution": "node",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "jsx": "preserve",
            "incremental": True
        },
        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
        "exclude": ["node_modules"]
    }
    (repo_path / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2))
    files.append("tsconfig.json")
    
    # Create directory structure
    (repo_path / "pages").mkdir(exist_ok=True)
    (repo_path / "styles").mkdir(exist_ok=True)
    (repo_path / "public").mkdir(exist_ok=True)
    
    return files

async def _get_repo_summary(repo_dir: Path) -> Dict[str, Any]:
    """Get summary info for a repository."""
    summary = {
        "name": repo_dir.name,
        "path": str(repo_dir)
    }
    
    # Check for metadata
    metadata_path = repo_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            summary['created_at'] = metadata.get('created_at', 'Unknown')
            summary['template'] = metadata.get('template', 'Unknown')
        except:
            pass
    
    # Check for key files
    summary['has_monetization'] = (repo_dir / "monetization_strategy.md").exists()
    summary['has_architecture'] = (repo_dir / "architecture.md").exists()
    
    return summary

# Test function
async def main():
    # Test listing repositories
    result = await manage_local_repository(operation='list')
    print(json.dumps(result, indent=2))
    
    # Test creating a repository
    create_result = await manage_local_repository(
        operation='create',
        repo_name='test-project',
        template='laravel-react',
        description='Test project for repository manager'
    )
    print(json.dumps(create_result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())