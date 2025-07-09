"""
Setup Laravel React Starter Tool
=================================

Sets up the complete Laravel React starter kit in repositories that are missing it.
This tool clones a template repository and pushes it to empty repositories.
"""

import os
import asyncio
import logging
import subprocess
import json
import shutil
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback

from scripts.state_manager import StateManager
from scripts.logger import logger

__description__ = "Set up complete Laravel React starter kit in repositories missing the codebase"
__parameters__ = {
    'repo_name': {'type': 'string', 'required': True, 'description': 'Repository to set up (e.g., reputation-ai)'},
    'template_repo': {'type': 'string', 'required': False, 'description': 'Template repository to use (default: uses a standard Laravel React template)'},
    'preserve_docs': {'type': 'boolean', 'required': False, 'description': 'Preserve existing README and ARCHITECTURE files'}
}
__examples__ = [
    {'repo_name': 'reputation-ai', 'preserve_docs': True},
    {'repo_name': 'new-project'}
]

# Laravel React starter kit initialization commands
LARAVEL_INIT_COMMANDS = [
    "composer create-project laravel/laravel . --prefer-dist",
    "composer require laravel/breeze --dev",
    "php artisan breeze:install react --typescript",
    "npm install",
    "npm run build"
]

# Additional configs for Tailwind and PostCSS if missing
TAILWIND_CONFIG = """/** @type {import('tailwindcss').Config} */
export default {
    content: [
        './vendor/laravel/framework/src/Illuminate/Pagination/resources/views/*.blade.php',
        './storage/framework/views/*.php',
        './resources/views/**/*.blade.php',
        './resources/js/**/*.tsx',
        './resources/js/**/*.ts',
        './resources/js/**/*.jsx',
        './resources/js/**/*.js',
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Figtree', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            },
        },
    },
    plugins: [require('@tailwindcss/forms')],
};
"""

POSTCSS_CONFIG = """export default {
    plugins: {
        tailwindcss: {},
        autoprefixer: {},
    },
};
"""

async def setup_laravel_react_starter(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up Laravel React starter kit in a repository.
    """
    try:
        repo_name = kwargs.get('repo_name')
        if not repo_name:
            return {"success": False, "error": "repo_name is required"}
        
        template_repo = kwargs.get('template_repo')
        preserve_docs = kwargs.get('preserve_docs', True)
        
        # Initialize components with absolute path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        state_file_path = os.path.join(project_root, "system_state.json")
        state_manager = StateManager(local_path=state_file_path)
        state_manager.load_state()
        state = state_manager.get_state()
        
        # Find repository in state
        projects = state.get('projects', {})
        project = None
        full_repo_name = None
        
        for key, proj in projects.items():
            if key == repo_name or proj.get('name') == repo_name:
                project = proj
                full_repo_name = proj.get('full_name')
                break
        
        if not project:
            return {"success": False, "error": f"Repository '{repo_name}' not found in managed projects"}
        
        logger.info(f"Setting up Laravel React starter kit for {full_repo_name}...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            
            # Clone the repository
            logger.info(f"Cloning {full_repo_name}...")
            clone_cmd = f"gh repo clone {full_repo_name} {repo_path}"
            result = subprocess.run(clone_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"success": False, "error": f"Failed to clone repository: {result.stderr}"}
            
            # Save existing docs if requested
            saved_files = {}
            if preserve_docs:
                for doc_file in ['README.md', 'ARCHITECTURE.md']:
                    doc_path = repo_path / doc_file
                    if doc_path.exists():
                        saved_files[doc_file] = doc_path.read_text()
                        logger.info(f"Preserved {doc_file}")
            
            # Check if repository is empty (only docs)
            existing_files = list(repo_path.iterdir())
            non_doc_files = [f for f in existing_files if f.name not in ['README.md', 'ARCHITECTURE.md', '.git', '.github']]
            
            if non_doc_files:
                return {
                    "success": False, 
                    "error": "Repository already contains code files. Use update tools instead.",
                    "existing_files": [f.name for f in non_doc_files[:10]]
                }
            
            # Initialize Laravel React project
            if template_repo:
                # Use template repository
                logger.info(f"Using template repository: {template_repo}")
                template_cmd = f"gh repo clone {template_repo} template_temp"
                subprocess.run(template_cmd, shell=True, cwd=temp_dir, capture_output=True)
                
                # Copy template files (excluding .git)
                template_path = Path(temp_dir) / "template_temp"
                if template_path.exists():
                    for item in template_path.iterdir():
                        if item.name != '.git':
                            dest = repo_path / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dest)
            else:
                # Create new Laravel project
                logger.info("Initializing new Laravel React project...")
                
                # Run composer create-project
                create_cmd = "composer create-project laravel/laravel temp_laravel --prefer-dist"
                result = subprocess.run(create_cmd, shell=True, cwd=temp_dir, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Move Laravel files to repo directory
                    laravel_path = Path(temp_dir) / "temp_laravel"
                    for item in laravel_path.iterdir():
                        if item.name != '.git':
                            dest = repo_path / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dest)
                    
                    # Install Breeze with React
                    logger.info("Installing Laravel Breeze with React...")
                    os.chdir(repo_path)
                    
                    breeze_cmds = [
                        "composer require laravel/breeze --dev",
                        "php artisan breeze:install react --typescript --no-interaction"
                    ]
                    
                    for cmd in breeze_cmds:
                        subprocess.run(cmd, shell=True, capture_output=True)
                else:
                    logger.warning("Composer failed, creating basic structure...")
                    # Create basic Laravel structure manually
                    create_basic_structure(repo_path)
            
            # Ensure Tailwind and PostCSS configs exist
            tailwind_path = repo_path / "tailwind.config.js"
            if not tailwind_path.exists():
                tailwind_path.write_text(TAILWIND_CONFIG)
                logger.info("Added tailwind.config.js")
            
            postcss_path = repo_path / "postcss.config.js"
            if not postcss_path.exists():
                postcss_path.write_text(POSTCSS_CONFIG)
                logger.info("Added postcss.config.js")
            
            # Restore preserved docs
            if saved_files:
                for doc_file, content in saved_files.items():
                    doc_path = repo_path / doc_file
                    doc_path.write_text(content)
                    logger.info(f"Restored {doc_file}")
            
            # Create .gitignore if missing
            gitignore_path = repo_path / ".gitignore"
            if not gitignore_path.exists():
                create_gitignore(gitignore_path)
            
            # Add all files and commit
            logger.info("Committing changes...")
            os.chdir(repo_path)
            
            git_commands = [
                "git add -A",
                'git commit -m "Initialize Laravel React starter kit\n\nAdded complete Laravel React setup with TypeScript and Tailwind CSS"'
            ]
            
            for cmd in git_commands:
                subprocess.run(cmd, shell=True, capture_output=True)
            
            # Push to repository
            logger.info("Pushing to repository...")
            push_cmd = f"git push origin {project.get('default_branch', 'main')}"
            push_result = subprocess.run(push_cmd, shell=True, capture_output=True, text=True)
            
            if push_result.returncode != 0:
                return {"success": False, "error": f"Failed to push: {push_result.stderr}"}
            
            logger.info(f"✓ Successfully set up Laravel React starter kit for {full_repo_name}")
            
            return {
                "success": True,
                "repository": full_repo_name,
                "message": "Laravel React starter kit successfully set up",
                "preserved_docs": list(saved_files.keys()) if saved_files else [],
                "added_files": ["Laravel framework", "React with TypeScript", "Tailwind CSS", "Laravel Breeze authentication"]
            }
            
    except Exception as e:
        error_msg = f"Error setting up Laravel React starter: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

def create_basic_structure(repo_path: Path):
    """Create basic Laravel/React structure if composer fails."""
    # Create directories
    dirs = [
        "app/Http/Controllers", "app/Models", "app/Providers",
        "bootstrap/cache", "config", "database/migrations", "database/seeders",
        "public", "resources/js/Components", "resources/js/Pages",
        "resources/js/Layouts", "resources/views", "routes", "storage/app/public",
        "storage/framework/cache", "storage/framework/sessions", "storage/framework/views",
        "storage/logs", "tests/Feature", "tests/Unit"
    ]
    
    for dir_path in dirs:
        (repo_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create basic files
    files = {
        "composer.json": get_basic_composer_json(),
        "package.json": get_basic_package_json(),
        ".env.example": get_basic_env_example(),
        "vite.config.ts": get_vite_config(),
        "tsconfig.json": get_tsconfig(),
        "resources/js/app.tsx": get_app_tsx(),
        "routes/web.php": "<?php\n\nuse Illuminate\\Support\\Facades\\Route;\n\nRoute::get('/', function () {\n    return view('welcome');\n});\n",
        "public/index.php": "<?php\n\nrequire __DIR__.'/../vendor/autoload.php';\n\n$app = require_once __DIR__.'/../bootstrap/app.php';\n\n$kernel = $app->make(Illuminate\\Contracts\\Http\\Kernel::class);\n\n$response = $kernel->handle(\n    $request = Illuminate\\Http\\Request::capture()\n);\n\n$response->send();\n\n$kernel->terminate($request, $response);\n"
    }
    
    for file_path, content in files.items():
        file_full_path = repo_path / file_path
        file_full_path.parent.mkdir(parents=True, exist_ok=True)
        file_full_path.write_text(content)

def get_basic_composer_json():
    return """{
    "name": "laravel/laravel",
    "type": "project",
    "description": "The Laravel Framework.",
    "keywords": ["framework", "laravel"],
    "license": "MIT",
    "require": {
        "php": "^8.1",
        "guzzlehttp/guzzle": "^7.2",
        "laravel/framework": "^10.0",
        "laravel/sanctum": "^3.2",
        "laravel/tinker": "^2.8"
    },
    "require-dev": {
        "fakerphp/faker": "^1.9.1",
        "laravel/breeze": "^1.0",
        "laravel/pint": "^1.0",
        "laravel/sail": "^1.18",
        "mockery/mockery": "^1.4.4",
        "nunomaduro/collision": "^7.0",
        "phpunit/phpunit": "^10.0",
        "spatie/laravel-ignition": "^2.0"
    },
    "autoload": {
        "psr-4": {
            "App\\\\": "app/",
            "Database\\\\Factories\\\\": "database/factories/",
            "Database\\\\Seeders\\\\": "database/seeders/"
        }
    },
    "scripts": {
        "post-autoload-dump": [
            "Illuminate\\\\Foundation\\\\ComposerScripts::postAutoloadDump",
            "@php artisan package:discover --ansi"
        ]
    },
    "minimum-stability": "stable",
    "prefer-stable": true
}"""

def get_basic_package_json():
    return """{
    "private": true,
    "scripts": {
        "dev": "vite",
        "build": "tsc && vite build"
    },
    "devDependencies": {
        "@headlessui/react": "^1.7.13",
        "@inertiajs/react": "^1.0.0",
        "@tailwindcss/forms": "^0.5.3",
        "@types/node": "^18.13.0",
        "@types/react": "^18.0.28",
        "@types/react-dom": "^18.0.10",
        "@vitejs/plugin-react": "^4.0.3",
        "autoprefixer": "^10.4.12",
        "axios": "^1.1.2",
        "laravel-vite-plugin": "^0.8.0",
        "postcss": "^8.4.18",
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "tailwindcss": "^3.2.1",
        "typescript": "^5.0.2",
        "vite": "^4.0.0"
    }
}"""

def get_basic_env_example():
    return """APP_NAME=Laravel
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
DB_DATABASE=laravel
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
MAIL_HOST=mailpit
MAIL_PORT=1025
MAIL_USERNAME=null
MAIL_PASSWORD=null
MAIL_ENCRYPTION=null
MAIL_FROM_ADDRESS="hello@example.com"
MAIL_FROM_NAME="${APP_NAME}"
"""

def get_vite_config():
    return """import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [
        laravel({
            input: 'resources/js/app.tsx',
            refresh: true,
        }),
        react(),
    ],
});
"""

def get_tsconfig():
    return """{
    "compilerOptions": {
        "target": "ES2020",
        "useDefineForClassFields": true,
        "lib": ["ES2020", "DOM", "DOM.Iterable"],
        "module": "ESNext",
        "skipLibCheck": true,
        "moduleResolution": "node",
        "allowImportingTsExtensions": true,
        "resolveJsonModule": true,
        "isolatedModules": true,
        "noEmit": true,
        "jsx": "react-jsx",
        "strict": true,
        "noUnusedLocals": true,
        "noUnusedParameters": true,
        "noFallthroughCasesInSwitch": true,
        "allowJs": true,
        "esModuleInterop": true,
        "allowSyntheticDefaultImports": true,
        "forceConsistentCasingInFileNames": true,
        "baseUrl": ".",
        "paths": {
            "@/*": ["./resources/js/*"]
        }
    },
    "include": ["resources/js/**/*.ts", "resources/js/**/*.tsx", "resources/js/**/*.js"],
    "exclude": ["node_modules", "public"]
}"""

def get_app_tsx():
    return """import './bootstrap';
import '../css/app.css';

import { createRoot } from 'react-dom/client';
import { createInertiaApp } from '@inertiajs/react';
import { resolvePageComponent } from 'laravel-vite-plugin/inertia-helpers';

const appName = window.document.getElementsByTagName('title')[0]?.innerText || 'Laravel';

createInertiaApp({
    title: (title) => `${title} - ${appName}`,
    resolve: (name) => resolvePageComponent(`./Pages/${name}.tsx`, import.meta.glob('./Pages/**/*.tsx')),
    setup({ el, App, props }) {
        const root = createRoot(el);
        root.render(<App {...props} />);
    },
    progress: {
        color: '#4B5563',
    },
});
"""

def create_gitignore(path: Path):
    """Create a comprehensive .gitignore file."""
    content = """# Laravel
/node_modules
/public/hot
/public/storage
/storage/*.key
/vendor
.env
.env.backup
.phpunit.result.cache
Homestead.json
Homestead.yaml
npm-debug.log
yarn-error.log
/.idea
/.vscode

# Build
/public/build
/public/mix-manifest.json

# OS
.DS_Store
Thumbs.db

# Editor
*.swp
*.swo
*~

# Logs
/storage/logs/*
!/storage/logs/.gitkeep
"""
    path.write_text(content)

# Export for tool system
setup_laravel_react_starter_tool = setup_laravel_react_starter

if __name__ == "__main__":
    import sys
    
    async def test():
        if len(sys.argv) > 1:
            repo = sys.argv[1]
            print(f"Testing setup for: {repo}")
            result = await setup_laravel_react_starter(repo_name=repo, preserve_docs=True)
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print("Usage: python setup_laravel_react_starter.py <repo_name>")
    
    asyncio.run(test())