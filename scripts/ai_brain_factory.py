"""
AI Brain Factory

Factory for creating AIBrain instances with environment-specific configurations.
Provides proper initialization for different deployment environments with 
appropriate state loading, context gathering, and error handling.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ai_brain import IntelligentAIBrain


class AIBrainFactory:
    """Factory for creating AIBrain instances with environment-specific configurations."""
    
    @staticmethod
    def create_for_workflow() -> IntelligentAIBrain:
        """Create AIBrain optimized for GitHub Actions workflow.
        
        Loads minimal state and context for fast initialization in CI environment.
        Includes workflow-specific metadata and fallback handling.
        
        Returns:
            AIBrain instance configured for GitHub Actions
        """
        logger = logging.getLogger('AIBrainFactory')
        
        try:
            # Validate GitHub Actions environment
            if not os.getenv('GITHUB_ACTIONS'):
                logger.warning("Not in GitHub Actions environment - using workflow defaults")
            
            # Load minimal state for workflow with repository discovery
            from state_manager import StateManager
            state_manager = StateManager()
            
            try:
                # Use repository discovery for workflows to get real project data
                logger.info("Loading state with repository discovery for workflow")
                state = state_manager.load_state_with_repository_discovery()
                logger.info(f"Workflow loaded {len(state.get('projects', {}))} repositories")
            except Exception as e:
                logger.warning(f"Repository discovery failed, using fallback: {e}")
                try:
                    state = state_manager.load_workflow_state()
                except (AttributeError, FileNotFoundError):
                    # Fallback to regular state if workflow-specific method doesn't exist
                    logger.info("Using standard state loading for workflow")
                    state = state_manager.load_state()
            
            # Create brain with loaded data first
            brain = IntelligentAIBrain(state, {})
            
            # Gather CI-specific context using integrated methods
            try:
                import asyncio
                context = asyncio.run(brain.gather_workflow_context())
            except Exception as e:
                logger.warning(f"Workflow context gathering failed: {e}, using minimal context")
                # Fallback to minimal context
                context = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "charter_goals": ["workflow_execution", "ci_optimization"],
                    "environment": "github_actions",
                    "market_trends": [],
                    "technology_updates": [],
                    "github_trending": [],
                    "programming_news": []
                }
            
            # Add workflow-specific metadata
            context.update({
                'environment': 'github_actions',
                'run_id': os.getenv('GITHUB_RUN_ID'),
                'ref': os.getenv('GITHUB_REF'),
                'repository': os.getenv('GITHUB_REPOSITORY'),
                'actor': os.getenv('GITHUB_ACTOR'),
                'workflow': os.getenv('GITHUB_WORKFLOW'),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'optimized_for': 'ci_performance'
            })
            
            # Update brain with gathered context
            brain.context.update(context)
            
            # Validate brain health
            if AIBrainFactory._validate_brain_health(brain):
                logger.info("AIBrain created successfully for GitHub Actions workflow")
                return brain
            else:
                logger.warning("Brain health check failed, using fallback")
                return AIBrainFactory.create_minimal_fallback()
            
        except Exception as e:
            logger.error(f"Failed to create workflow AIBrain: {e}")
            logger.info("Using minimal fallback AIBrain")
            return AIBrainFactory.create_minimal_fallback()
    
    @staticmethod
    def create_for_production() -> IntelligentAIBrain:
        """Create AIBrain for live production system.
        
        Loads complete state and context for full functionality.
        Includes monitoring, audit logging, and production optimizations.
        
        Returns:
            AIBrain instance configured for production
        """
        logger = logging.getLogger('AIBrainFactory')
        
        try:
            # Load complete production state with repository discovery
            from state_manager import StateManager
            state_manager = StateManager()
            
            try:
                # Always use repository discovery for production to get latest repo data
                logger.info("Loading production state with repository discovery")
                state = state_manager.load_state_with_repository_discovery()
                logger.info(f"Production loaded {len(state.get('projects', {}))} repositories")
            except Exception as e:
                logger.warning(f"Repository discovery failed in production, using fallback: {e}")
                try:
                    state = state_manager.load_production_state()
                except (AttributeError, FileNotFoundError):
                    logger.info("Using standard state loading for production")
                    state = state_manager.load_state()
            
            # Create brain with loaded data first
            brain = IntelligentAIBrain(state, {})
            
            # Gather full production context using integrated methods
            try:
                import asyncio
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, create a task
                    task = loop.create_task(brain.gather_production_context(state.get('charter', {})))
                    context = loop.run_until_complete(task)
                    logger.info("Production context gathered successfully via event loop task")
                except RuntimeError:
                    # No running event loop, safe to use asyncio.run()
                    context = asyncio.run(brain.gather_production_context(state.get('charter', {})))
                    logger.info("Production context gathered successfully via asyncio.run()")
            except Exception as e:
                logger.warning(f"Production context gathering failed: {e}, using standard context")
                # Fallback to standard context gathering - fix async issue
                try:
                    import asyncio
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # If we're in a running loop, create a task
                        task = loop.create_task(brain.gather_context(state.get('charter', {})))
                        context = loop.run_until_complete(task)
                        logger.info("Context gathered successfully via event loop task")
                    except RuntimeError:
                        # No running event loop, safe to use asyncio.run()
                        context = asyncio.run(brain.gather_context(state.get('charter', {})))
                        logger.info("Context gathered successfully via asyncio.run()")
                except Exception as e2:
                    logger.error(f"All context gathering failed: {e2}, using minimal context")
                    context = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "charter_goals": [state.get('charter', {}).get("primary_goal", ""), state.get('charter', {}).get("secondary_goal", "")],
                        "environment": "production",
                        "market_trends": [],
                        "security_alerts": [],
                        "technology_updates": [],
                        "github_trending": [],
                        "programming_news": []
                    }
            
            # Add production-specific configuration
            context.update({
                'environment': 'production',
                'monitoring_enabled': True,
                'audit_logging': True,
                'full_functionality': True,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'optimized_for': 'full_capabilities'
            })
            
            # Update brain with gathered context
            brain.context.update(context)
            
            if AIBrainFactory._validate_brain_health(brain):
                logger.info("AIBrain created successfully for production")
                return brain
            else:
                logger.error("Production brain health check failed")
                raise RuntimeError("Failed to create healthy production AIBrain")
            
        except Exception as e:
            logger.error(f"Failed to create production AIBrain: {e}")
            raise
    
    @staticmethod
    def create_for_testing() -> IntelligentAIBrain:
        """Create AIBrain with controlled test data.
        
        Uses predictable test data for consistent unit testing.
        Disables external API calls and provides mock data.
        
        Returns:
            AIBrain instance configured for testing
        """
        logger = logging.getLogger('AIBrainFactory')
        
        # Predictable test data
        test_state = {
            'charter': {
                'purpose': 'test_system',
                'version': 'test_v1',
                'created_at': '2025-01-01T00:00:00Z'
            },
            'projects': {
                'test_project_1': {
                    'name': 'Test Project Alpha',
                    'status': 'active',
                    'health_score': 0.85,
                    'type': 'web_application'
                },
                'test_project_2': {
                    'name': 'Test Project Beta', 
                    'status': 'development',
                    'health_score': 0.70,
                    'type': 'api_service'
                }
            },
            'system_performance': {
                'success_rate': 0.85,
                'avg_task_completion_time': 45,
                'total_tasks_completed': 150,
                'uptime_percentage': 99.5
            }
        }
        
        test_context = {
            'environment': 'test',
            'mock_data': True,
            'api_calls_disabled': True,
            'predictable_responses': True,
            'test_mode': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'market_trends': ['test_trend_1', 'test_trend_2'],
            'capabilities': ['GitHub API', 'AI Models', 'Task Generation']
        }
        
        brain = IntelligentAIBrain(test_state, test_context)
        logger.info("AIBrain created successfully for testing")
        return brain
    
    @staticmethod
    def create_for_development() -> IntelligentAIBrain:
        """Create AIBrain for local development.
        
        Optimized for development workflow with debug features enabled.
        Uses development-specific configurations and logging.
        
        Returns:
            AIBrain instance configured for development
        """
        logger = logging.getLogger('AIBrainFactory')
        
        try:
            # Load state with development settings
            from state_manager import StateManager
            state_manager = StateManager()
            state = state_manager.load_state()
            
            # Create brain with loaded data first
            brain = IntelligentAIBrain(state, {})
            
            # Gather context with development optimizations using integrated methods
            try:
                import asyncio
                context = asyncio.run(brain.gather_context(state.get('charter', {})))
            except Exception as e:
                logger.warning(f"Development context gathering failed: {e}, using minimal context")
                context = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "charter_goals": [state.get('charter', {}).get("primary_goal", ""), state.get('charter', {}).get("secondary_goal", "")],
                    "environment": "development",
                    "market_trends": [],
                    "technology_updates": [],
                    "github_trending": [],
                    "programming_news": []
                }
            
            # Add development-specific configuration
            context.update({
                'environment': 'development',
                'debug_mode': True,
                'verbose_logging': True,
                'development_features': True,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'optimized_for': 'development_speed'
            })
            
            # Update brain with gathered context
            brain.context.update(context)
            logger.info("AIBrain created successfully for development")
            return brain
            
        except Exception as e:
            logger.error(f"Failed to create development AIBrain: {e}")
            # For development, fallback to test data so developers can continue working
            logger.info("Using test configuration for development fallback")
            return AIBrainFactory.create_for_testing()
    
    @staticmethod
    def create_for_research() -> IntelligentAIBrain:
        """Create AIBrain optimized for research tasks.
        
        Configured for research operations with focus on information gathering,
        analysis, and insight extraction.
        
        Returns:
            AIBrain instance configured for research
        """
        logger = logging.getLogger('AIBrainFactory')
        
        try:
            # Load state for research operations
            from state_manager import StateManager
            state_manager = StateManager()
            
            # Load state with research focus
            state = state_manager.load_state()
            
            # Create brain with loaded data
            brain = IntelligentAIBrain(state, {})
            
            # Gather research-focused context
            try:
                import asyncio
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, create a task
                    task = loop.create_task(brain.gather_context(state.get('charter', {})))
                    context = loop.run_until_complete(task)
                except RuntimeError:
                    # No running event loop, safe to use asyncio.run()
                    context = asyncio.run(brain.gather_context(state.get('charter', {})))
            except Exception as e:
                logger.warning(f"Research context gathering failed: {e}, using minimal context")
                context = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "charter_goals": ["research", "continuous_improvement"],
                    "environment": "research",
                    "market_trends": [],
                    "technology_updates": [],
                    "github_trending": [],
                    "programming_news": []
                }
            
            # Add research-specific configuration
            context.update({
                'environment': 'research',
                'research_mode': True,
                'analysis_enabled': True,
                'learning_enabled': True,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'optimized_for': 'research_and_learning',
                'research_capabilities': {
                    'web_search': True,
                    'paper_analysis': True,
                    'trend_identification': True,
                    'pattern_recognition': True,
                    'insight_extraction': True
                }
            })
            
            # Update brain with research context
            brain.context.update(context)
            
            if AIBrainFactory._validate_brain_health(brain):
                logger.info("AIBrain created successfully for research")
                return brain
            else:
                logger.warning("Research brain health check failed, using fallback")
                return AIBrainFactory.create_for_development()
            
        except Exception as e:
            logger.error(f"Failed to create research AIBrain: {e}")
            logger.info("Using development AIBrain as fallback for research")
            return AIBrainFactory.create_for_development()
    
    @staticmethod
    def create_minimal_fallback() -> IntelligentAIBrain:
        """Create emergency fallback AIBrain when normal creation fails.
        
        Provides basic functionality with minimal dependencies.
        Used as last resort when other factory methods fail.
        
        Returns:
            Minimal but functional AIBrain instance
        """
        logger = logging.getLogger('AIBrainFactory')
        
        # Minimal state for emergency operation
        minimal_state = {
            'charter': {
                'purpose': 'emergency_mode',
                'version': 'fallback_v1',
                'created_at': datetime.now(timezone.utc).isoformat()
            },
            'projects': {},
            'system_performance': {
                'success_rate': 0.0,
                'total_tasks_completed': 0
            }
        }
        
        minimal_context = {
            'environment': 'fallback',
            'limited_functionality': True,
            'emergency_mode': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'capabilities': ['basic_operations']
        }
        
        brain = IntelligentAIBrain(minimal_state, minimal_context)
        logger.warning("Created minimal fallback AIBrain - limited functionality")
        return brain
    
    @staticmethod
    def create_with_config(config: Dict[str, Any]) -> IntelligentAIBrain:
        """Create AIBrain with custom configuration.
        
        Args:
            config: Custom configuration dictionary
            
        Returns:
            AIBrain instance with custom configuration
        """
        logger = logging.getLogger('AIBrainFactory')
        
        try:
            # Load state based on config
            from state_manager import StateManager
            state_manager = StateManager()
            
            if hasattr(state_manager, 'load_state_with_config'):
                state = state_manager.load_state_with_config(config)
            else:
                state = state_manager.load_state()
            
            # Create brain with loaded data first
            brain = IntelligentAIBrain(state, {})
            
            # Gather context based on config using integrated methods
            try:
                import asyncio
                context = asyncio.run(brain.gather_context(state.get('charter', {})))
            except Exception as e:
                logger.warning(f"Context gathering failed: {e}, using minimal context")
                context = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "charter_goals": [state.get('charter', {}).get("primary_goal", ""), state.get('charter', {}).get("secondary_goal", "")],
                    "market_trends": [],
                    "technology_updates": [],
                    "github_trending": [],
                    "programming_news": []
                }
            
            # Add config metadata
            context.update({
                'environment': 'custom_config',
                'config_applied': True,
                'created_at': datetime.now(timezone.utc).isoformat()
            })
            
            # Update brain with gathered context
            brain.context.update(context)
            logger.info("AIBrain created successfully with custom configuration")
            return brain
            
        except Exception as e:
            logger.error(f"Failed to create AIBrain with config: {e}")
            raise
    
    @staticmethod
    def _validate_brain_health(brain: IntelligentAIBrain) -> bool:
        """Validate that AIBrain is functional.
        
        Args:
            brain: AIBrain instance to validate
            
        Returns:
            True if brain passes health checks, False otherwise
        """
        try:
            # Verify essential attributes exist
            if not hasattr(brain, 'state') or not hasattr(brain, 'context'):
                return False
            
            # Verify state has required structure
            if not isinstance(brain.state, dict):
                return False
                
            # Verify context has required structure  
            if not isinstance(brain.context, dict):
                return False
            
            # Test that AI clients are accessible (if available)
            if hasattr(brain, 'anthropic_client') and hasattr(brain, 'openai_client'):
                # Basic structure check passed
                pass
            
            # Test research capabilities method
            if hasattr(brain, 'get_research_capabilities'):
                try:
                    capabilities = brain.get_research_capabilities()
                    if not isinstance(capabilities, dict):
                        return False
                    # Verify required keys exist
                    required_keys = ['available_providers', 'research_functions', 'analysis_types', 'research_ready']
                    if not all(key in capabilities for key in required_keys):
                        return False
                except Exception:
                    return False
            
            # Test basic attribute access
            _ = brain.charter  # Should not raise exception
            _ = brain.projects  # Should not raise exception
            
            return True
            
        except Exception as e:
            logger = logging.getLogger('AIBrainFactory')
            logger.error(f"AIBrain health check failed: {e}")
            return False


# Convenience aliases for backward compatibility
AIBrain = IntelligentAIBrain