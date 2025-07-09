"""
AI-Generated Tool: environment_config_manager
Description: Manage environment configuration files (.env) for repositories
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

__description__ = "Manage .env files - create, update, validate, and sync environment variables"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["get", "set", "list", "create", "validate", "sync", "backup", "restore"],
        "description": "Environment config operation"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "key": {
        "type": "string",
        "required": False,
        "description": "Environment variable key"
    },
    "value": {
        "type": "string",
        "required": False,
        "description": "Environment variable value"
    },
    "template": {
        "type": "string",
        "required": False,
        "enum": ["laravel", "node", "react", "nextjs", "basic"],
        "description": "Template to use for creating .env"
    },
    "source": {
        "type": "string",
        "required": False,
        "default": ".env.example",
        "description": "Source file for sync operation"
    }
}
__examples__ = [
    "environment_config_manager(operation='list', repo_name='my-app')",
    "environment_config_manager(operation='set', repo_name='my-app', key='API_KEY', value='secret123')",
    "environment_config_manager(operation='get', repo_name='my-app', key='DATABASE_URL')",
    "environment_config_manager(operation='create', repo_name='my-app', template='laravel')",
    "environment_config_manager(operation='validate', repo_name='my-app')"
]

# Base path
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"

# Environment templates
ENV_TEMPLATES = {
    'laravel': """APP_NAME="{app_name}"
APP_ENV=local
APP_KEY=
APP_DEBUG=true
APP_URL=http://localhost

LOG_CHANNEL=stack
LOG_DEPRECATIONS_CHANNEL=null
LOG_LEVEL=debug

DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE={db_name}
DB_USERNAME=root
DB_PASSWORD=

BROADCAST_DRIVER=log
CACHE_DRIVER=file
FILESYSTEM_DISK=local
QUEUE_CONNECTION=sync
SESSION_DRIVER=file
SESSION_LIFETIME=120

REDIS_HOST=127.0.0.1
REDIS_PASSWORD=null
REDIS_PORT=6379

MAIL_MAILER=smtp
MAIL_HOST=mailhog
MAIL_PORT=1025
MAIL_USERNAME=null
MAIL_PASSWORD=null
MAIL_ENCRYPTION=null
MAIL_FROM_ADDRESS="hello@example.com"
MAIL_FROM_NAME="${APP_NAME}"
""",
    
    'node': """NODE_ENV=development
PORT=3000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# API Keys
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here

# External Services
REDIS_URL=redis://localhost:6379
""",
    
    'react': """REACT_APP_API_URL=http://localhost:3001/api
REACT_APP_ENV=development

# Feature Flags
REACT_APP_ENABLE_AUTH=true
REACT_APP_ENABLE_ANALYTICS=false

# Third Party Keys
REACT_APP_GOOGLE_ANALYTICS_ID=
REACT_APP_SENTRY_DSN=
""",
    
    'nextjs': """# Environment
NODE_ENV=development

# URLs
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_APP_URL=http://localhost:3000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Authentication
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key

# API Keys
API_SECRET_KEY=
STRIPE_SECRET_KEY=
STRIPE_PUBLISHABLE_KEY=
""",
    
    'basic': """# Application
APP_ENV=development
DEBUG=true

# Database
DATABASE_URL=

# API Keys
API_KEY=
SECRET_KEY=

# Services
CACHE_ENABLED=false
"""
}

async def environment_config_manager(**kwargs) -> Dict[str, Any]:
    """
    Manage environment configuration files.
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
    
    # Route to appropriate operation
    operations = {
        'get': get_env_value,
        'set': set_env_value,
        'list': list_env_vars,
        'create': create_env_file,
        'validate': validate_env_file,
        'sync': sync_env_file,
        'backup': backup_env_file,
        'restore': restore_env_file
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        kwargs['repo_path'] = repo_path
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def get_env_value(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get an environment variable value."""
    repo_path = params['repo_path']
    key = params.get('key')
    
    if not key:
        return {
            "status": "error",
            "message": "key parameter is required for get operation"
        }
    
    env_file = repo_path / '.env'
    if not env_file.exists():
        return {
            "status": "error",
            "message": ".env file not found",
            "suggestion": "Create .env file first using create operation"
        }
    
    # Read and parse .env file
    env_vars = _parse_env_file(env_file)
    
    if key in env_vars:
        # Mask sensitive values
        masked_value = _mask_sensitive_value(key, env_vars[key])
        
        return {
            "status": "success",
            "key": key,
            "value": env_vars[key],
            "masked_value": masked_value,
            "exists": True
        }
    else:
        return {
            "status": "info",
            "key": key,
            "exists": False,
            "message": f"Key '{key}' not found in .env file"
        }

async def set_env_value(params: Dict[str, Any]) -> Dict[str, Any]:
    """Set an environment variable value."""
    repo_path = params['repo_path']
    key = params.get('key')
    value = params.get('value', '')
    
    if not key:
        return {
            "status": "error",
            "message": "key parameter is required for set operation"
        }
    
    env_file = repo_path / '.env'
    
    # Create backup before modifying
    if env_file.exists():
        backup_result = await backup_env_file(params)
        if backup_result['status'] != 'success':
            return backup_result
    
    # Read existing content or create new
    if env_file.exists():
        content = env_file.read_text()
        env_vars = _parse_env_file(env_file)
    else:
        content = ""
        env_vars = {}
    
    # Update or add the key
    if key in env_vars:
        # Replace existing value
        pattern = re.compile(f'^{re.escape(key)}=.*$', re.MULTILINE)
        new_line = f'{key}={value}'
        content = pattern.sub(new_line, content)
        action = "updated"
    else:
        # Add new key
        if content and not content.endswith('\n'):
            content += '\n'
        content += f'{key}={value}\n'
        action = "added"
    
    # Write back
    env_file.write_text(content)
    
    return {
        "status": "success",
        "message": f"Environment variable {action}",
        "key": key,
        "masked_value": _mask_sensitive_value(key, value),
        "action": action,
        "backup_created": True
    }

async def list_env_vars(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all environment variables."""
    repo_path = params['repo_path']
    
    env_file = repo_path / '.env'
    if not env_file.exists():
        return {
            "status": "error",
            "message": ".env file not found"
        }
    
    # Parse .env file
    env_vars = _parse_env_file(env_file)
    
    # Categorize variables
    categorized = _categorize_env_vars(env_vars)
    
    # Mask sensitive values
    masked_vars = {}
    for key, value in env_vars.items():
        masked_vars[key] = _mask_sensitive_value(key, value)
    
    return {
        "status": "success",
        "total_variables": len(env_vars),
        "variables": masked_vars,
        "categories": categorized,
        "file_size": env_file.stat().st_size,
        "last_modified": datetime.fromtimestamp(env_file.stat().st_mtime).isoformat()
    }

async def create_env_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new .env file from template."""
    repo_path = params['repo_path']
    template = params.get('template', 'basic')
    
    env_file = repo_path / '.env'
    
    # Check if already exists
    if env_file.exists():
        return {
            "status": "error",
            "message": ".env file already exists",
            "suggestion": "Use backup operation first or delete existing file"
        }
    
    # Check for .env.example
    env_example = repo_path / '.env.example'
    if env_example.exists():
        # Copy from example
        shutil.copy2(env_example, env_file)
        source = ".env.example"
    else:
        # Use template
        if template not in ENV_TEMPLATES:
            template = 'basic'
        
        # Prepare template variables
        template_vars = {
            'app_name': params.get('repo_name', 'MyApp').replace('-', ' ').title(),
            'db_name': params.get('repo_name', 'myapp').replace('-', '_')
        }
        
        content = ENV_TEMPLATES[template].format(**template_vars)
        env_file.write_text(content)
        source = f"{template} template"
    
    # Create .env.example if it doesn't exist
    if not env_example.exists():
        env_example.write_text(env_file.read_text())
    
    return {
        "status": "success",
        "message": f".env file created from {source}",
        "file_path": str(env_file),
        "template_used": template if source.endswith('template') else None,
        "next_steps": [
            "Update environment variables with actual values",
            "Ensure .env is in .gitignore",
            "Generate APP_KEY if using Laravel: php artisan key:generate"
        ]
    }

async def validate_env_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate .env file for common issues."""
    repo_path = params['repo_path']
    
    env_file = repo_path / '.env'
    if not env_file.exists():
        return {
            "status": "error",
            "message": ".env file not found"
        }
    
    # Parse file
    env_vars = _parse_env_file(env_file)
    issues = []
    warnings = []
    
    # Check for empty required values
    required_patterns = ['KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'DATABASE', 'URL']
    for key, value in env_vars.items():
        if any(pattern in key.upper() for pattern in required_patterns):
            if not value or value in ['your_key_here', 'your_secret_here', 'changeme', 'password']:
                issues.append(f"{key}: Empty or default value")
    
    # Check for common mistakes
    for key, value in env_vars.items():
        # Spaces around equals
        if ' = ' in f"{key}={value}":
            warnings.append(f"{key}: Spaces around = sign")
        
        # Quotes in values (sometimes needed, sometimes not)
        if value.startswith('"') and value.endswith('"'):
            warnings.append(f"{key}: Value is quoted (may not be needed)")
        
        # Trailing spaces
        if value != value.strip():
            issues.append(f"{key}: Has trailing whitespace")
    
    # Check if .env is in .gitignore
    gitignore = repo_path / '.gitignore'
    env_ignored = False
    if gitignore.exists():
        gitignore_content = gitignore.read_text()
        if '.env' in gitignore_content or '*.env' in gitignore_content:
            env_ignored = True
    
    if not env_ignored:
        issues.append(".env file is not in .gitignore - SECURITY RISK")
    
    # Detect project type and check required vars
    project_type = await _detect_project_type(repo_path)
    missing_required = _check_required_vars(env_vars, project_type)
    issues.extend(missing_required)
    
    return {
        "status": "success" if not issues else "warning",
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "total_variables": len(env_vars),
        "env_in_gitignore": env_ignored,
        "project_type": project_type
    }

async def sync_env_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sync .env with .env.example or another source."""
    repo_path = params['repo_path']
    source = params.get('source', '.env.example')
    
    env_file = repo_path / '.env'
    source_file = repo_path / source
    
    if not source_file.exists():
        return {
            "status": "error",
            "message": f"Source file '{source}' not found"
        }
    
    # Parse both files
    source_vars = _parse_env_file(source_file)
    
    if env_file.exists():
        current_vars = _parse_env_file(env_file)
        
        # Backup current .env
        await backup_env_file(params)
    else:
        current_vars = {}
    
    # Find differences
    missing_keys = set(source_vars.keys()) - set(current_vars.keys())
    extra_keys = set(current_vars.keys()) - set(source_vars.keys())
    
    # Add missing keys
    if missing_keys:
        content = env_file.read_text() if env_file.exists() else ""
        
        if content and not content.endswith('\n'):
            content += '\n'
        
        content += f"\n# Synced from {source} on {datetime.now().isoformat()}\n"
        
        for key in sorted(missing_keys):
            value = source_vars[key]
            content += f"{key}={value}\n"
        
        env_file.write_text(content)
    
    return {
        "status": "success",
        "message": f"Synced with {source}",
        "added_keys": list(missing_keys),
        "extra_keys": list(extra_keys),
        "total_added": len(missing_keys),
        "suggestion": "Update the newly added keys with actual values"
    }

async def backup_env_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a backup of .env file."""
    repo_path = params['repo_path']
    
    env_file = repo_path / '.env'
    if not env_file.exists():
        return {
            "status": "error",
            "message": ".env file not found"
        }
    
    # Create backup filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = repo_path / f'.env.backup_{timestamp}'
    
    # Copy file
    shutil.copy2(env_file, backup_file)
    
    # Clean old backups (keep last 5)
    backups = sorted(repo_path.glob('.env.backup_*'))
    if len(backups) > 5:
        for old_backup in backups[:-5]:
            old_backup.unlink()
    
    return {
        "status": "success",
        "message": "Backup created",
        "backup_file": backup_file.name,
        "backup_path": str(backup_file),
        "total_backups": len(list(repo_path.glob('.env.backup_*')))
    }

async def restore_env_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Restore .env from backup."""
    repo_path = params['repo_path']
    backup_name = params.get('backup_name')
    
    if not backup_name:
        # List available backups
        backups = sorted(repo_path.glob('.env.backup_*'), reverse=True)
        
        if not backups:
            return {
                "status": "error",
                "message": "No backups found"
            }
        
        return {
            "status": "info",
            "message": "Available backups",
            "backups": [b.name for b in backups[:10]],
            "suggestion": "Specify backup_name parameter with one of the backup files"
        }
    
    # Find backup file
    backup_file = repo_path / backup_name
    if not backup_file.exists():
        # Try with .env.backup_ prefix
        backup_file = repo_path / f'.env.backup_{backup_name}'
        if not backup_file.exists():
            return {
                "status": "error",
                "message": f"Backup file '{backup_name}' not found"
            }
    
    # Backup current .env before restoring
    env_file = repo_path / '.env'
    if env_file.exists():
        current_backup = await backup_env_file(params)
        if current_backup['status'] != 'success':
            return current_backup
    
    # Restore
    shutil.copy2(backup_file, env_file)
    
    return {
        "status": "success",
        "message": "Environment file restored from backup",
        "restored_from": backup_file.name,
        "current_backed_up": env_file.exists()
    }

def _parse_env_file(env_file: Path) -> Dict[str, str]:
    """Parse .env file into dictionary."""
    env_vars = {}
    
    content = env_file.read_text()
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        # Parse key=value
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            env_vars[key] = value
    
    return env_vars

def _mask_sensitive_value(key: str, value: str) -> str:
    """Mask sensitive environment values."""
    sensitive_patterns = ['KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'PRIVATE', 'CREDENTIAL']
    
    if any(pattern in key.upper() for pattern in sensitive_patterns):
        if len(value) > 4:
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
        else:
            return '*' * len(value)
    
    return value

def _categorize_env_vars(env_vars: Dict[str, str]) -> Dict[str, List[str]]:
    """Categorize environment variables by type."""
    categories = {
        'app': [],
        'database': [],
        'auth': [],
        'api': [],
        'cache': [],
        'mail': [],
        'other': []
    }
    
    for key in env_vars:
        key_upper = key.upper()
        
        if any(word in key_upper for word in ['APP_', 'NODE_ENV', 'PORT']):
            categories['app'].append(key)
        elif any(word in key_upper for word in ['DB_', 'DATABASE']):
            categories['database'].append(key)
        elif any(word in key_upper for word in ['AUTH', 'JWT', 'SESSION']):
            categories['auth'].append(key)
        elif any(word in key_upper for word in ['API', 'ENDPOINT', 'URL']):
            categories['api'].append(key)
        elif any(word in key_upper for word in ['CACHE', 'REDIS']):
            categories['cache'].append(key)
        elif any(word in key_upper for word in ['MAIL', 'SMTP', 'EMAIL']):
            categories['mail'].append(key)
        else:
            categories['other'].append(key)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

async def _detect_project_type(repo_path: Path) -> str:
    """Detect project type from files."""
    if (repo_path / 'artisan').exists():
        return 'laravel'
    elif (repo_path / 'next.config.js').exists():
        return 'nextjs'
    elif (repo_path / 'package.json').exists():
        package_data = (repo_path / 'package.json').read_text()
        if 'react-scripts' in package_data:
            return 'react'
        return 'node'
    return 'unknown'

def _check_required_vars(env_vars: Dict[str, str], project_type: str) -> List[str]:
    """Check for required environment variables by project type."""
    missing = []
    
    required_by_type = {
        'laravel': ['APP_NAME', 'APP_ENV', 'APP_KEY', 'APP_URL', 'DB_CONNECTION'],
        'nextjs': ['NODE_ENV'],
        'react': ['REACT_APP_API_URL'],
        'node': ['NODE_ENV', 'PORT']
    }
    
    required = required_by_type.get(project_type, [])
    
    for var in required:
        if var not in env_vars or not env_vars[var]:
            missing.append(f"Missing required variable: {var}")
    
    return missing

# Test function
async def main():
    # Test environment operations
    result = await environment_config_manager(
        operation='list',
        repo_name='brand-guardian-ai'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())