"""
API Client - Stub module for custom tools
"""

import aiohttp
import requests
from typing import Dict, Any, Optional
import json


class APIClient:
    """Simple API client for custom tools"""
    
    def __init__(self, base_url: str = "", headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.headers = headers or {}
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request"""
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request"""
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    async def async_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an async GET request"""
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
    
    async def async_post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an async POST request"""
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()


# Default instance
api_client = APIClient()