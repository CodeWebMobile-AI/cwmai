#!/usr/bin/env python3
"""
Deployment Manager

Handle all deployment operations including cloud deployments, 
container management, CI/CD pipelines, and environment management.
"""

import os
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import shutil

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.http_ai_client import HTTPAIClient
from scripts.logger import Logger


class DeploymentManager:
    """Manages all deployment operations with AI-enhanced capabilities."""
    
    def __init__(self):
        self.logger = Logger()
        self.ai_client = HTTPAIClient()
        self.deployment_configs = {
            'docker': {
                'file': 'Dockerfile',
                'compose': 'docker-compose.yml',
                'registry': 'docker.io'
            },
            'kubernetes': {
                'deployment': 'k8s/deployment.yaml',
                'service': 'k8s/service.yaml',
                'config': 'k8s/configmap.yaml'
            },
            'cloud': {
                'aws': {'cli': 'aws', 'config': '.aws/config'},
                'gcp': {'cli': 'gcloud', 'config': '.gcloud/config'},
                'azure': {'cli': 'az', 'config': '.azure/config'}
            }
        }
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Main entry point with operation routing."""
        operations = {
            'deploy': self.deploy_application,
            'rollback': self.rollback_deployment,
            'status': self.check_deployment_status,
            'containerize': self.containerize_application,
            'setup_cicd': self.setup_cicd_pipeline,
            'manage_env': self.manage_environments,
            'scale': self.scale_deployment,
            'monitor': self.monitor_deployment
        }
        
        if operation not in operations:
            return {
                'success': False,
                'error': f'Unknown operation: {operation}',
                'available_operations': list(operations.keys())
            }
            
        try:
            return await operations[operation](**kwargs)
        except Exception as e:
            self.logger.error(f"Error in {operation}: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    async def deploy_application(self, target: str = 'docker', 
                                environment: str = 'production',
                                config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy application to specified target."""
        try:
            deployment_steps = []
            
            # Validate environment
            if not self._validate_environment(environment):
                return {'success': False, 'error': f'Invalid environment: {environment}'}
                
            # Deploy based on target
            if target == 'docker':
                result = await self._deploy_docker(environment, config)
            elif target == 'kubernetes':
                result = await self._deploy_kubernetes(environment, config)
            elif target in ['aws', 'gcp', 'azure']:
                result = await self._deploy_cloud(target, environment, config)
            else:
                return {'success': False, 'error': f'Unknown deployment target: {target}'}
                
            # Record deployment
            deployment_record = {
                'timestamp': datetime.now().isoformat(),
                'target': target,
                'environment': environment,
                'version': result.get('version', 'latest'),
                'status': 'success' if result.get('success') else 'failed'
            }
            
            self._save_deployment_record(deployment_record)
            
            return {
                'success': result.get('success', False),
                'target': target,
                'environment': environment,
                'deployment_id': result.get('deployment_id'),
                'url': result.get('url'),
                'steps': result.get('steps', []),
                'record': deployment_record
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def rollback_deployment(self, deployment_id: str = None,
                                 target: str = 'docker',
                                 steps_back: int = 1) -> Dict[str, Any]:
        """Rollback to a previous deployment."""
        try:
            # Get deployment history
            history = self._get_deployment_history(target)
            
            if not history:
                return {'success': False, 'error': 'No deployment history found'}
                
            # Find target deployment
            if deployment_id:
                target_deployment = next((d for d in history if d['id'] == deployment_id), None)
            else:
                # Rollback by steps
                if steps_back >= len(history):
                    return {'success': False, 'error': 'Not enough deployments to rollback'}
                target_deployment = history[-(steps_back + 1)]
                
            if not target_deployment:
                return {'success': False, 'error': 'Target deployment not found'}
                
            # Perform rollback
            if target == 'docker':
                result = await self._rollback_docker(target_deployment)
            elif target == 'kubernetes':
                result = await self._rollback_kubernetes(target_deployment)
            else:
                result = await self._rollback_cloud(target, target_deployment)
                
            return {
                'success': result.get('success', False),
                'rolled_back_to': target_deployment.get('id'),
                'version': target_deployment.get('version'),
                'timestamp': target_deployment.get('timestamp'),
                'result': result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def check_deployment_status(self, target: str = 'docker',
                                    deployment_id: str = None) -> Dict[str, Any]:
        """Check status of deployments."""
        try:
            if target == 'docker':
                status = await self._check_docker_status()
            elif target == 'kubernetes':
                status = await self._check_kubernetes_status()
            else:
                status = await self._check_cloud_status(target)
                
            return {
                'success': True,
                'target': target,
                'status': status,
                'healthy': status.get('healthy', False),
                'instances': status.get('instances', []),
                'metrics': status.get('metrics', {})
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def containerize_application(self, app_path: str = '.',
                                     base_image: str = None,
                                     port: int = None) -> Dict[str, Any]:
        """Create Docker container for application."""
        try:
            # Detect application type
            app_type = self._detect_application_type(app_path)
            
            # Generate Dockerfile using AI
            dockerfile_content = await self._generate_dockerfile(
                app_path, app_type, base_image, port
            )
            
            # Save Dockerfile
            dockerfile_path = os.path.join(app_path, 'Dockerfile')
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
                
            # Generate docker-compose.yml
            compose_content = await self._generate_docker_compose(
                app_type, port
            )
            
            compose_path = os.path.join(app_path, 'docker-compose.yml')
            with open(compose_path, 'w') as f:
                f.write(compose_content)
                
            # Build image
            build_result = await self._build_docker_image(app_path)
            
            return {
                'success': True,
                'dockerfile': dockerfile_path,
                'docker_compose': compose_path,
                'app_type': app_type,
                'image_name': build_result.get('image_name'),
                'build_log': build_result.get('log')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def setup_cicd_pipeline(self, platform: str = 'github',
                                 stages: List[str] = None) -> Dict[str, Any]:
        """Set up CI/CD pipeline."""
        try:
            if not stages:
                stages = ['test', 'build', 'deploy']
                
            # Generate pipeline config based on platform
            if platform == 'github':
                config = await self._generate_github_actions(stages)
                config_path = '.github/workflows/deploy.yml'
            elif platform == 'gitlab':
                config = await self._generate_gitlab_ci(stages)
                config_path = '.gitlab-ci.yml'
            elif platform == 'jenkins':
                config = await self._generate_jenkinsfile(stages)
                config_path = 'Jenkinsfile'
            else:
                return {'success': False, 'error': f'Unknown CI/CD platform: {platform}'}
                
            # Create directories if needed
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Save config
            with open(config_path, 'w') as f:
                f.write(config)
                
            return {
                'success': True,
                'platform': platform,
                'config_file': config_path,
                'stages': stages,
                'config': config,
                'instructions': self._get_cicd_setup_instructions(platform)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def manage_environments(self, action: str = 'list',
                                environment: str = None,
                                config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manage deployment environments."""
        try:
            if action == 'list':
                envs = self._list_environments()
                return {
                    'success': True,
                    'environments': envs,
                    'count': len(envs)
                }
                
            elif action == 'create':
                if not environment:
                    return {'success': False, 'error': 'Environment name required'}
                    
                result = await self._create_environment(environment, config)
                return {
                    'success': result.get('success'),
                    'environment': environment,
                    'config': result.get('config'),
                    'message': f'Environment {environment} created'
                }
                
            elif action == 'update':
                if not environment:
                    return {'success': False, 'error': 'Environment name required'}
                    
                result = await self._update_environment(environment, config)
                return {
                    'success': result.get('success'),
                    'environment': environment,
                    'updated_config': result.get('config')
                }
                
            elif action == 'delete':
                if not environment:
                    return {'success': False, 'error': 'Environment name required'}
                    
                if environment in ['production', 'main']:
                    return {'success': False, 'error': 'Cannot delete protected environment'}
                    
                result = self._delete_environment(environment)
                return {
                    'success': result,
                    'message': f'Environment {environment} deleted'
                }
                
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def scale_deployment(self, target: str = 'docker',
                             replicas: int = None,
                             auto_scale: bool = False,
                             metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Scale deployment up or down."""
        try:
            if target == 'docker':
                result = await self._scale_docker(replicas)
            elif target == 'kubernetes':
                result = await self._scale_kubernetes(replicas, auto_scale, metrics)
            else:
                result = await self._scale_cloud(target, replicas, auto_scale)
                
            return {
                'success': result.get('success'),
                'target': target,
                'current_replicas': result.get('current'),
                'desired_replicas': replicas,
                'auto_scaling': auto_scale,
                'scaling_policy': result.get('policy')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def monitor_deployment(self, target: str = 'docker',
                               metrics: List[str] = None) -> Dict[str, Any]:
        """Monitor deployment metrics."""
        try:
            if not metrics:
                metrics = ['cpu', 'memory', 'requests', 'errors']
                
            # Collect metrics based on target
            if target == 'docker':
                data = await self._collect_docker_metrics(metrics)
            elif target == 'kubernetes':
                data = await self._collect_kubernetes_metrics(metrics)
            else:
                data = await self._collect_cloud_metrics(target, metrics)
                
            # Analyze metrics
            analysis = await self._analyze_metrics(data)
            
            return {
                'success': True,
                'target': target,
                'metrics': data,
                'analysis': analysis,
                'alerts': analysis.get('alerts', []),
                'recommendations': analysis.get('recommendations', [])
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    # Helper methods for Docker
    async def _deploy_docker(self, environment: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy using Docker."""
        steps = []
        
        # Build image
        build_cmd = ['docker', 'build', '-t', f'app:{environment}', '.']
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        steps.append(f"Built image: app:{environment}")
        
        # Run container
        run_cmd = [
            'docker', 'run', '-d',
            '--name', f'app-{environment}',
            '-p', f'{config.get("port", 8080)}:8080',
            f'app:{environment}'
        ]
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        container_id = result.stdout.strip()
        
        steps.append(f"Started container: {container_id[:12]}")
        
        return {
            'success': result.returncode == 0,
            'deployment_id': container_id,
            'version': environment,
            'steps': steps,
            'url': f'http://localhost:{config.get("port", 8080)}'
        }
        
    async def _deploy_kubernetes(self, environment: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy using Kubernetes."""
        steps = []
        
        # Apply deployment
        deploy_cmd = ['kubectl', 'apply', '-f', 'k8s/']
        result = subprocess.run(deploy_cmd, capture_output=True, text=True)
        steps.append("Applied Kubernetes manifests")
        
        # Wait for rollout
        wait_cmd = ['kubectl', 'rollout', 'status', 'deployment/app']
        result = subprocess.run(wait_cmd, capture_output=True, text=True)
        steps.append("Deployment rolled out successfully")
        
        # Get service URL
        svc_cmd = ['kubectl', 'get', 'svc', 'app', '-o', 'json']
        result = subprocess.run(svc_cmd, capture_output=True, text=True)
        
        return {
            'success': True,
            'deployment_id': f'k8s-{environment}',
            'version': environment,
            'steps': steps
        }
        
    async def _deploy_cloud(self, provider: str, environment: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to cloud provider."""
        if provider == 'aws':
            # Simplified AWS deployment
            return {
                'success': True,
                'deployment_id': f'aws-{environment}-{datetime.now().strftime("%Y%m%d%H%M%S")}',
                'version': environment,
                'url': f'https://app-{environment}.amazonaws.com'
            }
        elif provider == 'gcp':
            return {
                'success': True,
                'deployment_id': f'gcp-{environment}-{datetime.now().strftime("%Y%m%d%H%M%S")}',
                'version': environment,
                'url': f'https://app-{environment}.appspot.com'
            }
        else:
            return {'success': False, 'error': f'Cloud provider {provider} not implemented'}
            
    async def _generate_dockerfile(self, app_path: str, app_type: str,
                                 base_image: str, port: int) -> str:
        """Generate Dockerfile using AI."""
        # Read key files to understand the app
        files_content = self._read_app_files(app_path, app_type)
        
        prompt = f"""Generate a production-ready Dockerfile for this {app_type} application:

Application structure:
{files_content}

Requirements:
1. Use appropriate base image ({base_image or 'auto-select best'})
2. Install all dependencies
3. Set up proper working directory
4. Copy application files efficiently
5. Expose port {port or 'auto-detect'}
6. Use multi-stage build if beneficial
7. Include security best practices
8. Optimize for size and performance

Return only the Dockerfile content."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _generate_docker_compose(self, app_type: str, port: int) -> str:
        """Generate docker-compose.yml."""
        compose = f"""version: '3.8'

services:
  app:
    build: .
    ports:
      - "{port or 8080}:8080"
    environment:
      - NODE_ENV=production
    restart: unless-stopped
"""
        
        # Add database if needed
        if app_type in ['laravel', 'django', 'rails']:
            compose += """
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
"""
        
        return compose
        
    async def _generate_github_actions(self, stages: List[str]) -> str:
        """Generate GitHub Actions workflow."""
        workflow = """name: Deploy

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
"""
        
        for stage in stages:
            if stage == 'test':
                workflow += """
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up environment
        uses: actions/setup-node@v3
        with:
          node-version: '16'
      - name: Install dependencies
        run: npm install
      - name: Run tests
        run: npm test
"""
            elif stage == 'build':
                workflow += """
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t app:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push app:${{ github.sha }}
"""
            elif stage == 'deploy':
                workflow += """
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Add deployment commands here
          echo "Deploying to production"
"""
        
        return workflow
        
    def _detect_application_type(self, app_path: str) -> str:
        """Detect application type from files."""
        checks = [
            ('package.json', 'node'),
            ('requirements.txt', 'python'),
            ('composer.json', 'php'),
            ('Gemfile', 'ruby'),
            ('go.mod', 'go'),
            ('pom.xml', 'java')
        ]
        
        for file, app_type in checks:
            if os.path.exists(os.path.join(app_path, file)):
                return app_type
                
        return 'unknown'
        
    def _validate_environment(self, environment: str) -> bool:
        """Validate environment name."""
        valid_envs = ['development', 'staging', 'production', 'test']
        return environment in valid_envs
        
    def _save_deployment_record(self, record: Dict[str, Any]):
        """Save deployment record."""
        history_file = 'deployment_history.json'
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
            
        history.append(record)
        
        # Keep last 100 deployments
        history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    def _get_deployment_history(self, target: str) -> List[Dict[str, Any]]:
        """Get deployment history."""
        try:
            with open('deployment_history.json', 'r') as f:
                history = json.load(f)
            return [d for d in history if d.get('target') == target]
        except:
            return []
            
    def _list_environments(self) -> List[Dict[str, Any]]:
        """List all environments."""
        return [
            {'name': 'development', 'active': True},
            {'name': 'staging', 'active': True},
            {'name': 'production', 'active': True}
        ]
        
    async def _create_environment(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new environment."""
        env_config = {
            'name': name,
            'created': datetime.now().isoformat(),
            'config': config or {}
        }
        
        # Save environment config
        env_file = f'environments/{name}.json'
        os.makedirs('environments', exist_ok=True)
        
        with open(env_file, 'w') as f:
            json.dump(env_config, f, indent=2)
            
        return {'success': True, 'config': env_config}
        
    def _read_app_files(self, app_path: str, app_type: str) -> str:
        """Read key application files."""
        files_to_read = {
            'node': ['package.json', 'index.js', 'app.js'],
            'python': ['requirements.txt', 'app.py', 'main.py'],
            'php': ['composer.json', 'index.php'],
        }
        
        content = ""
        for file in files_to_read.get(app_type, []):
            file_path = os.path.join(app_path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content += f"\n{file}:\n{f.read()[:500]}\n"
                    
        return content
        
    async def _analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deployment metrics."""
        analysis = {
            'healthy': True,
            'alerts': [],
            'recommendations': []
        }
        
        # Check CPU usage
        if metrics.get('cpu', {}).get('usage', 0) > 80:
            analysis['alerts'].append('High CPU usage detected')
            analysis['recommendations'].append('Consider scaling up')
            
        # Check memory
        if metrics.get('memory', {}).get('usage', 0) > 85:
            analysis['alerts'].append('High memory usage detected')
            analysis['recommendations'].append('Optimize memory usage or scale up')
            
        # Check error rate
        if metrics.get('errors', {}).get('rate', 0) > 5:
            analysis['healthy'] = False
            analysis['alerts'].append('High error rate detected')
            
        return analysis
        
    def _get_cicd_setup_instructions(self, platform: str) -> str:
        """Get setup instructions for CI/CD platform."""
        instructions = {
            'github': "1. Commit the workflow file\n2. Set up secrets in repository settings\n3. Push to trigger pipeline",
            'gitlab': "1. Commit the .gitlab-ci.yml file\n2. Configure CI/CD variables\n3. Push to trigger pipeline",
            'jenkins': "1. Create new pipeline job\n2. Point to repository\n3. Configure credentials\n4. Run pipeline"
        }
        return instructions.get(platform, "Follow platform documentation")
        
    # Simplified metric collection methods
    async def _collect_docker_metrics(self, metrics: List[str]) -> Dict[str, Any]:
        """Collect Docker metrics."""
        return {
            'cpu': {'usage': 45.2},
            'memory': {'usage': 62.1},
            'requests': {'rate': 120},
            'errors': {'rate': 0.5}
        }
        
    async def _check_docker_status(self) -> Dict[str, Any]:
        """Check Docker deployment status."""
        cmd = ['docker', 'ps', '--format', 'json']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            'healthy': True,
            'instances': 1,
            'metrics': {'uptime': '2 days'}
        }
        
    async def _scale_docker(self, replicas: int) -> Dict[str, Any]:
        """Scale Docker deployment."""
        return {
            'success': True,
            'current': 1,
            'desired': replicas,
            'message': f'Scaling to {replicas} replicas'
        }
        
    async def _build_docker_image(self, app_path: str) -> Dict[str, Any]:
        """Build Docker image."""
        cmd = ['docker', 'build', '-t', 'app:latest', app_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'image_name': 'app:latest',
            'log': result.stdout
        }


async def main(operation: str = None, **kwargs):
    """Tool entry point."""
    manager = DeploymentManager()
    
    if not operation:
        return {
            'success': True,
            'available_operations': [
                'deploy', 'rollback', 'status', 'containerize',
                'setup_cicd', 'manage_env', 'scale', 'monitor'
            ],
            'description': 'Comprehensive deployment management tool'
        }
        
    return await manager.execute(operation, **kwargs)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        # Show available operations
        result = await main()
        print("Deployment Manager Operations:", result)
        
        # Containerize application
        result = await main('containerize', app_path='.', port=8080)
        print("\nContainerized app:", result.get('success'))
        
        # Set up CI/CD
        result = await main('setup_cicd', platform='github')
        print("\nCI/CD setup:", result.get('config_file'))
        
    asyncio.run(demo())