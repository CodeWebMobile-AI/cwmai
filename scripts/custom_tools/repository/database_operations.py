"""
AI-Generated Tool: database_operations
Description: Manage database operations for repositories - migrations, seeding, backups
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import subprocess
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil

__description__ = "Manage database operations - run migrations, seed data, backup/restore"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["migrate", "rollback", "seed", "fresh", "backup", "restore", "status", "create-migration"],
        "description": "Database operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "migration_name": {
        "type": "string",
        "required": False,
        "description": "Name for new migration (for create-migration)"
    },
    "steps": {
        "type": "integer",
        "required": False,
        "default": 1,
        "description": "Number of migrations to rollback"
    },
    "backup_name": {
        "type": "string",
        "required": False,
        "description": "Name for backup file (auto-generated if not provided)"
    }
}
__examples__ = [
    "database_operations(operation='migrate', repo_name='my-app')",
    "database_operations(operation='seed', repo_name='my-app')",
    "database_operations(operation='rollback', repo_name='my-app', steps=2)",
    "database_operations(operation='backup', repo_name='my-app')",
    "database_operations(operation='create-migration', repo_name='my-app', migration_name='add_users_table')"
]

# Base paths
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"
BACKUPS_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "db_backups"

async def database_operations(**kwargs) -> Dict[str, Any]:
    """
    Manage database operations for repositories.
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
    
    # Detect project type and database system
    project_info = await _detect_project_database(repo_path)
    if not project_info['has_database']:
        return {
            "status": "error",
            "message": "No database configuration found in this project"
        }
    
    # Route to appropriate operation
    operations = {
        'migrate': run_migrations,
        'rollback': rollback_migrations,
        'seed': seed_database,
        'fresh': fresh_database,
        'backup': backup_database,
        'restore': restore_database,
        'status': migration_status,
        'create-migration': create_migration
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        kwargs['repo_path'] = repo_path
        kwargs['project_info'] = project_info
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def run_migrations(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run pending database migrations."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    
    if project_info['type'] == 'laravel':
        cmd = ['php', 'artisan', 'migrate']
        
        # Check if in production
        env_file = repo_path / '.env'
        if env_file.exists():
            env_content = env_file.read_text()
            if 'APP_ENV=production' in env_content:
                cmd.append('--force')
    
    elif project_info['type'] == 'django':
        cmd = ['python', 'manage.py', 'migrate']
    
    elif project_info['type'] == 'rails':
        cmd = ['rails', 'db:migrate']
    
    else:
        # Check for custom migration tools
        migration_cmd = await _detect_migration_command(repo_path)
        if not migration_cmd:
            return {
                "status": "error",
                "message": "No migration command found for this project type"
            }
        cmd = migration_cmd
    
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
                "message": "Migration failed",
                "error": result.stderr,
                "command": ' '.join(cmd)
            }
        
        # Parse migration output
        migrations_run = _parse_migration_output(result.stdout, project_info['type'])
        
        return {
            "status": "success",
            "message": "Migrations completed successfully",
            "migrations_run": migrations_run,
            "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Migration command failed: {str(e)}"
        }

async def rollback_migrations(params: Dict[str, Any]) -> Dict[str, Any]:
    """Rollback database migrations."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    steps = params.get('steps', 1)
    
    if project_info['type'] == 'laravel':
        cmd = ['php', 'artisan', 'migrate:rollback', f'--step={steps}']
    
    elif project_info['type'] == 'django':
        # Django requires specific app and migration name
        return {
            "status": "error",
            "message": "Django rollback requires specific app and migration name",
            "suggestion": "Use Django admin to identify migration to rollback to"
        }
    
    elif project_info['type'] == 'rails':
        cmd = ['rails', 'db:rollback', f'STEP={steps}']
    
    else:
        return {
            "status": "error",
            "message": "Rollback not supported for this project type"
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
                "message": "Rollback failed",
                "error": result.stderr
            }
        
        return {
            "status": "success",
            "message": f"Rolled back {steps} migration(s)",
            "output": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Rollback failed: {str(e)}"
        }

async def seed_database(params: Dict[str, Any]) -> Dict[str, Any]:
    """Seed the database with sample data."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    
    if project_info['type'] == 'laravel':
        cmd = ['php', 'artisan', 'db:seed']
    
    elif project_info['type'] == 'django':
        # Check for fixtures
        fixtures_dir = repo_path / 'fixtures'
        if fixtures_dir.exists():
            cmd = ['python', 'manage.py', 'loaddata', 'fixtures/*.json']
        else:
            return {
                "status": "error",
                "message": "No fixtures found for Django project"
            }
    
    elif project_info['type'] == 'rails':
        cmd = ['rails', 'db:seed']
    
    else:
        # Check for seed script
        if (repo_path / 'seed.js').exists():
            cmd = ['node', 'seed.js']
        elif (repo_path / 'scripts' / 'seed.js').exists():
            cmd = ['node', 'scripts/seed.js']
        else:
            return {
                "status": "error",
                "message": "No seed script found"
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
                "message": "Seeding failed",
                "error": result.stderr
            }
        
        return {
            "status": "success",
            "message": "Database seeded successfully",
            "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Seeding failed: {str(e)}"
        }

async def fresh_database(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop all tables and re-run migrations (DESTRUCTIVE)."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    
    if project_info['type'] == 'laravel':
        cmd = ['php', 'artisan', 'migrate:fresh', '--seed']
    
    elif project_info['type'] == 'rails':
        # Rails equivalent
        commands = [
            ['rails', 'db:drop'],
            ['rails', 'db:create'],
            ['rails', 'db:migrate'],
            ['rails', 'db:seed']
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            if result.returncode != 0:
                return {
                    "status": "error",
                    "message": f"Fresh database failed at: {' '.join(cmd)}",
                    "error": result.stderr
                }
        
        return {
            "status": "success",
            "message": "Database refreshed successfully"
        }
    
    else:
        return {
            "status": "error",
            "message": "Fresh database not supported for this project type",
            "suggestion": "Manually drop and recreate database, then run migrations"
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
                "message": "Fresh migration failed",
                "error": result.stderr
            }
        
        return {
            "status": "success",
            "message": "Database refreshed with migrations and seeds",
            "warning": "All existing data has been deleted"
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Fresh database failed: {str(e)}"
        }

async def backup_database(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a database backup."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    backup_name = params.get('backup_name')
    
    # Generate backup name if not provided
    if not backup_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{params['repo_name']}_{timestamp}"
    
    # Ensure backups directory exists
    BACKUPS_PATH.mkdir(parents=True, exist_ok=True)
    backup_file = BACKUPS_PATH / f"{backup_name}.sql"
    
    # Get database config
    db_config = await _get_database_config(repo_path, project_info['type'])
    if not db_config:
        return {
            "status": "error",
            "message": "Could not read database configuration"
        }
    
    # Build backup command based on database type
    if db_config['driver'] == 'mysql':
        cmd = [
            'mysqldump',
            '-h', db_config.get('host', 'localhost'),
            '-P', str(db_config.get('port', 3306)),
            '-u', db_config.get('username', 'root'),
            f"-p{db_config.get('password', '')}",
            db_config['database']
        ]
    
    elif db_config['driver'] == 'postgresql':
        env = os.environ.copy()
        env['PGPASSWORD'] = db_config.get('password', '')
        
        cmd = [
            'pg_dump',
            '-h', db_config.get('host', 'localhost'),
            '-p', str(db_config.get('port', 5432)),
            '-U', db_config.get('username', 'postgres'),
            '-d', db_config['database'],
            '-f', str(backup_file)
        ]
    
    elif db_config['driver'] == 'sqlite':
        # Simple file copy for SQLite
        sqlite_file = repo_path / db_config['database']
        if sqlite_file.exists():
            shutil.copy2(sqlite_file, backup_file)
            return {
                "status": "success",
                "message": "SQLite database backed up",
                "backup_file": str(backup_file),
                "size_mb": round(backup_file.stat().st_size / 1024 / 1024, 2)
            }
        else:
            return {
                "status": "error",
                "message": f"SQLite database file not found: {db_config['database']}"
            }
    
    else:
        return {
            "status": "error",
            "message": f"Backup not supported for database driver: {db_config['driver']}"
        }
    
    try:
        # Run backup command
        if db_config['driver'] == 'mysql':
            # MySQL outputs to stdout
            with open(backup_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True
                )
        else:
            # PostgreSQL uses -f flag
            result = subprocess.run(
                cmd,
                env=env if db_config['driver'] == 'postgresql' else None,
                capture_output=True,
                text=True
            )
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": "Database backup failed",
                "error": result.stderr
            }
        
        # Get backup file size
        size_mb = round(backup_file.stat().st_size / 1024 / 1024, 2)
        
        return {
            "status": "success",
            "message": "Database backed up successfully",
            "backup_file": str(backup_file),
            "size_mb": size_mb,
            "database": db_config['database'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Backup failed: {str(e)}"
        }

async def restore_database(params: Dict[str, Any]) -> Dict[str, Any]:
    """Restore database from backup."""
    backup_name = params.get('backup_name')
    
    if not backup_name:
        return {
            "status": "error",
            "message": "backup_name is required for restore operation"
        }
    
    # Find backup file
    backup_file = BACKUPS_PATH / f"{backup_name}.sql"
    if not backup_file.exists():
        # Try without extension
        backup_file = BACKUPS_PATH / backup_name
        if not backup_file.exists():
            return {
                "status": "error",
                "message": f"Backup file not found: {backup_name}",
                "available_backups": [f.stem for f in BACKUPS_PATH.glob('*.sql')][:10]
            }
    
    return {
        "status": "info",
        "message": "Database restore requires careful handling",
        "backup_file": str(backup_file),
        "warning": "Manual restore recommended to avoid data loss",
        "instructions": [
            "1. Create a backup of current database first",
            "2. Drop existing database or tables",
            "3. Restore from backup file",
            "4. Run any new migrations"
        ]
    }

async def migration_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check migration status."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    
    if project_info['type'] == 'laravel':
        cmd = ['php', 'artisan', 'migrate:status']
    
    elif project_info['type'] == 'django':
        cmd = ['python', 'manage.py', 'showmigrations']
    
    elif project_info['type'] == 'rails':
        cmd = ['rails', 'db:migrate:status']
    
    else:
        return {
            "status": "error",
            "message": "Migration status not supported for this project type"
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
                "message": "Failed to get migration status",
                "error": result.stderr
            }
        
        # Parse status output
        status_info = _parse_migration_status(result.stdout, project_info['type'])
        
        return {
            "status": "success",
            "migration_status": status_info,
            "raw_output": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Status check failed: {str(e)}"
        }

async def create_migration(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new migration file."""
    repo_path = params['repo_path']
    project_info = params['project_info']
    migration_name = params.get('migration_name')
    
    if not migration_name:
        return {
            "status": "error",
            "message": "migration_name is required for create-migration"
        }
    
    if project_info['type'] == 'laravel':
        cmd = ['php', 'artisan', 'make:migration', migration_name]
    
    elif project_info['type'] == 'django':
        # Django requires app name
        return {
            "status": "error",
            "message": "Django migration creation requires app name",
            "example": "python manage.py makemigrations app_name"
        }
    
    elif project_info['type'] == 'rails':
        cmd = ['rails', 'generate', 'migration', migration_name]
    
    else:
        return {
            "status": "error",
            "message": "Migration creation not supported for this project type"
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
                "message": "Failed to create migration",
                "error": result.stderr
            }
        
        # Extract created file path
        created_file = _extract_migration_file(result.stdout, project_info['type'])
        
        return {
            "status": "success",
            "message": f"Migration '{migration_name}' created",
            "migration_file": created_file,
            "next_steps": [
                "Edit the migration file to define schema changes",
                "Run 'database_operations(operation=\"migrate\")' to apply"
            ]
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Migration creation failed: {str(e)}"
        }

async def _detect_project_database(repo_path: Path) -> Dict[str, Any]:
    """Detect project type and database configuration."""
    # Laravel
    if (repo_path / 'artisan').exists():
        return {
            'type': 'laravel',
            'has_database': True,
            'config_file': '.env'
        }
    
    # Django
    if (repo_path / 'manage.py').exists():
        return {
            'type': 'django',
            'has_database': True,
            'config_file': 'settings.py'
        }
    
    # Rails
    if (repo_path / 'Gemfile').exists() and 'rails' in (repo_path / 'Gemfile').read_text():
        return {
            'type': 'rails',
            'has_database': True,
            'config_file': 'config/database.yml'
        }
    
    # Node.js with common ORMs
    if (repo_path / 'package.json').exists():
        package_data = json.loads((repo_path / 'package.json').read_text())
        deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
        
        if 'sequelize' in deps or 'typeorm' in deps or 'prisma' in deps:
            return {
                'type': 'node',
                'has_database': True,
                'config_file': '.env'
            }
    
    return {
        'type': 'unknown',
        'has_database': False
    }

async def _get_database_config(repo_path: Path, project_type: str) -> Optional[Dict[str, Any]]:
    """Extract database configuration."""
    config = {}
    
    if project_type == 'laravel':
        env_file = repo_path / '.env'
        if env_file.exists():
            env_content = env_file.read_text()
            
            # Parse .env file
            for line in env_content.splitlines():
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    if key == 'DB_CONNECTION':
                        config['driver'] = value
                    elif key == 'DB_HOST':
                        config['host'] = value
                    elif key == 'DB_PORT':
                        config['port'] = int(value) if value.isdigit() else value
                    elif key == 'DB_DATABASE':
                        config['database'] = value
                    elif key == 'DB_USERNAME':
                        config['username'] = value
                    elif key == 'DB_PASSWORD':
                        config['password'] = value
    
    return config if config else None

async def _detect_migration_command(repo_path: Path) -> Optional[List[str]]:
    """Detect migration command for non-standard projects."""
    # Check package.json scripts
    if (repo_path / 'package.json').exists():
        package_data = json.loads((repo_path / 'package.json').read_text())
        scripts = package_data.get('scripts', {})
        
        if 'migrate' in scripts:
            return ['npm', 'run', 'migrate']
        elif 'db:migrate' in scripts:
            return ['npm', 'run', 'db:migrate']
    
    return None

def _parse_migration_output(output: str, project_type: str) -> int:
    """Parse migration output to count migrations run."""
    count = 0
    
    if project_type == 'laravel':
        for line in output.splitlines():
            if 'Migrating:' in line:
                count += 1
    
    return count

def _parse_migration_status(output: str, project_type: str) -> Dict[str, Any]:
    """Parse migration status output."""
    status = {
        'total': 0,
        'run': 0,
        'pending': 0
    }
    
    if project_type == 'laravel':
        for line in output.splitlines():
            if 'Run' in line:
                status['run'] += 1
                status['total'] += 1
            elif 'Pending' in line:
                status['pending'] += 1
                status['total'] += 1
    
    return status

def _extract_migration_file(output: str, project_type: str) -> Optional[str]:
    """Extract created migration file path from output."""
    if project_type == 'laravel':
        for line in output.splitlines():
            if 'Created Migration:' in line:
                return line.split(':', 1)[1].strip()
    
    return None

# Test function
async def main():
    # Test database operations
    result = await database_operations(
        operation='status',
        repo_name='brand-guardian-ai'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())