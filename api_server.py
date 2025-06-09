#!/usr/bin/env python3
"""
CWMAI API Server with Redis Rate Limiting

FastAPI-based API server for the CWMAI autonomous AI task management system.
Provides endpoints for system interaction with sophisticated rate limiting.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.ai_brain_factory import AIBrainFactory
from scripts.task_manager import TaskManager
from scripts.task_analyzer import TaskAnalyzer
from scripts.http_ai_client import HTTPAIClient


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=60, description="Requests per minute per IP")
    requests_per_hour: int = Field(default=1000, description="Requests per hour per IP")
    ai_requests_per_minute: int = Field(default=10, description="AI requests per minute per IP")
    ai_requests_per_hour: int = Field(default=100, description="AI requests per hour per IP")
    burst_requests: int = Field(default=10, description="Burst requests allowed")


class APIMetrics(BaseModel):
    """API metrics and monitoring data."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    average_response_time: float = 0.0
    active_connections: int = 0
    redis_status: str = "unknown"
    ai_provider_status: Dict[str, bool] = {}
    uptime_seconds: float = 0.0


class SystemStatusRequest(BaseModel):
    """Request model for system status."""
    include_metrics: bool = Field(default=True, description="Include performance metrics")
    include_tasks: bool = Field(default=True, description="Include task information")


class AIRequestModel(BaseModel):
    """Request model for AI interactions."""
    prompt: str = Field(..., description="The prompt to send to AI")
    model: Optional[str] = Field(default=None, description="Preferred AI model")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens in response")
    temperature: Optional[float] = Field(default=0.7, description="AI temperature setting")


class TaskCreateRequest(BaseModel):
    """Request model for creating tasks."""
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    priority: str = Field(default="medium", description="Task priority")
    task_type: str = Field(default="feature", description="Type of task")
    estimated_hours: float = Field(default=4.0, description="Estimated hours")


class CWMAIAPIServer:
    """CWMAI API Server with sophisticated rate limiting."""
    
    def __init__(self):
        """Initialize the API server."""
        self.app = FastAPI(
            title="CWMAI API",
            description="Autonomous AI Task Management System API with Redis Rate Limiting",
            version="1.0.0"
        )
        
        # Initialize components
        self.redis_client = None
        self.ai_brain = None
        self.task_manager = None
        self.task_analyzer = None
        self.ai_client = HTTPAIClient()
        self.start_time = time.time()
        self.metrics = APIMetrics()
        self.active_connections: List[WebSocket] = []
        
        # Rate limiting configuration
        self.rate_config = RateLimitConfig()
        
        # Initialize Redis connection
        self._init_redis()
        
        # Initialize rate limiter
        self.limiter = Limiter(
            key_func=get_remote_address,
            storage_uri=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
        )
        
        # Configure FastAPI app
        self._configure_app()
        
        # Initialize AI components
        self._init_ai_components()
        
        # Setup logging
        self._setup_logging()
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_password = os.getenv('REDIS_PASSWORD')
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            self.metrics.redis_status = "connected"
            print(f"✓ Redis connected at {redis_host}:{redis_port}")
            
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            print("Rate limiting will use in-memory storage")
            self.metrics.redis_status = "disconnected"
            # Fallback to in-memory storage
            self.redis_client = None
    
    def _configure_app(self):
        """Configure FastAPI application."""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add rate limiting
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Add custom middleware for metrics
        @self.app.middleware("http")
        async def metrics_middleware(request, call_next):
            start_time = time.time()
            self.metrics.total_requests += 1
            
            try:
                response = await call_next(request)
                self.metrics.successful_requests += 1
                return response
            except Exception as e:
                self.metrics.failed_requests += 1
                raise e
            finally:
                # Update average response time
                response_time = time.time() - start_time
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time) 
                    / self.metrics.total_requests
                )
        
        # Register routes
        self._register_routes()
    
    def _init_ai_components(self):
        """Initialize AI components."""
        try:
            self.ai_brain = AIBrainFactory.create_for_production()
            self.task_manager = TaskManager()
            self.task_analyzer = TaskAnalyzer()
            self.metrics.ai_provider_status = self.ai_client.get_research_ai_status()
            print("✓ AI components initialized")
        except Exception as e:
            print(f"⚠️ AI components initialization failed: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("CWMAIAPIServer")
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "CWMAI API",
                "version": "1.0.0",
                "description": "Autonomous AI Task Management System API",
                "features": [
                    "Redis-based rate limiting",
                    "AI provider integration",
                    "Task management",
                    "Real-time monitoring",
                    "WebSocket updates"
                ],
                "endpoints": {
                    "status": "/status",
                    "metrics": "/metrics",
                    "ai": "/ai/generate",
                    "tasks": "/tasks",
                    "websocket": "/ws"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "redis": self.metrics.redis_status,
                "ai_providers": self.metrics.ai_provider_status,
                "uptime": time.time() - self.start_time
            }
        
        @self.app.get("/status")
        @self.limiter.limit("30/minute")
        async def get_system_status(request, params: SystemStatusRequest = Depends()):
            """Get comprehensive system status."""
            try:
                status = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "system": "CWMAI",
                    "api_version": "1.0.0",
                    "status": "operational"
                }
                
                if params.include_metrics:
                    self.metrics.uptime_seconds = time.time() - self.start_time
                    status["metrics"] = self.metrics.dict()
                
                if params.include_tasks and self.task_analyzer:
                    # Get task information (mock for now)
                    status["tasks"] = {
                        "active_tasks": 0,
                        "completed_today": 0,
                        "success_rate": 0.95
                    }
                
                return status
                
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        @self.limiter.limit("60/minute")
        async def get_metrics(request):
            """Get detailed API metrics."""
            self.metrics.uptime_seconds = time.time() - self.start_time
            self.metrics.active_connections = len(self.active_connections)
            
            # Add Redis metrics if available
            redis_info = {}
            if self.redis_client:
                try:
                    redis_info = {
                        "connected_clients": self.redis_client.info().get('connected_clients', 0),
                        "used_memory": self.redis_client.info().get('used_memory_human', '0B'),
                        "keyspace_hits": self.redis_client.info().get('keyspace_hits', 0),
                        "keyspace_misses": self.redis_client.info().get('keyspace_misses', 0)
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to get Redis info: {e}")
            
            return {
                "api_metrics": self.metrics.dict(),
                "redis_metrics": redis_info,
                "rate_limit_config": self.rate_config.dict()
            }
        
        @self.app.post("/ai/generate")
        @self.limiter.limit("10/minute")
        async def generate_ai_response(request, ai_request: AIRequestModel):
            """Generate AI response with rate limiting."""
            try:
                # Track AI request metrics
                if self.redis_client:
                    key = f"ai_requests:{get_remote_address(request)}"
                    current = self.redis_client.incr(key)
                    if current == 1:
                        self.redis_client.expire(key, 3600)  # 1 hour expiry
                    
                    if current > self.rate_config.ai_requests_per_hour:
                        self.metrics.rate_limited_requests += 1
                        raise HTTPException(
                            status_code=429, 
                            detail="AI request rate limit exceeded"
                        )
                
                # Generate AI response
                response = await self.ai_client.generate_enhanced_response(
                    ai_request.prompt,
                    ai_request.model
                )
                
                # Broadcast to WebSocket connections
                await self._broadcast_update({
                    "type": "ai_response",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": response.get("provider"),
                    "success": True
                })
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error generating AI response: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tasks")
        @self.limiter.limit("20/minute")
        async def create_task(request, task_request: TaskCreateRequest):
            """Create a new task."""
            try:
                # Create task (mock implementation for now)
                task_data = {
                    "id": f"TASK-{int(time.time())}",
                    "title": task_request.title,
                    "description": task_request.description,
                    "priority": task_request.priority,
                    "type": task_request.task_type,
                    "estimated_hours": task_request.estimated_hours,
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.hset(
                        f"task:{task_data['id']}", 
                        mapping=task_data
                    )
                
                # Broadcast to WebSocket connections
                await self._broadcast_update({
                    "type": "task_created",
                    "task": task_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return task_data
                
            except Exception as e:
                self.logger.error(f"Error creating task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks")
        @self.limiter.limit("60/minute")
        async def list_tasks(request, limit: int = 10, offset: int = 0):
            """List tasks with pagination."""
            try:
                tasks = []
                
                # Get tasks from Redis if available
                if self.redis_client:
                    task_keys = self.redis_client.keys("task:*")
                    for key in task_keys[offset:offset+limit]:
                        task_data = self.redis_client.hgetall(key)
                        if task_data:
                            tasks.append(task_data)
                
                return {
                    "tasks": tasks,
                    "total": len(self.redis_client.keys("task:*")) if self.redis_client else 0,
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                self.logger.error(f"Error listing tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                # Send initial status
                await websocket.send_text(json.dumps({
                    "type": "connection_established",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "client_count": len(self.active_connections)
                }))
                
                # Keep connection alive and handle incoming messages
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    async def _broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all WebSocket connections."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the API server."""
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    CWMAI API SERVER                              ║
║                                                                  ║
║  Redis Rate Limiting: {'✓ Enabled' if self.redis_client else '✗ Disabled (In-Memory)'}                         ║
║  AI Providers: {len([p for p in self.metrics.ai_provider_status.values() if p])} Available                                   ║
║  WebSocket Support: ✓ Enabled                                   ║
║                                                                  ║
║  Access: http://{host}:{port}                                   ║
║  Docs: http://{host}:{port}/docs                               ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CWMAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    server = CWMAIAPIServer()
    server.run(host=args.host, port=args.port, reload=args.reload)