"""
AI-Generated Tool: analyze_file_types
Description: Analyze data for file types
Generated: 2025-06-15T11:05:48.161889+00:00
Requirements: 
        Tool Name: analyze_file_types
        Intent: Analyze data for file types
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Perform comprehensive analysis of items
2. Generate insights and recommendations
3. Return detailed report
        
"""

"""
File Type Analysis Module

This module provides functionality to analyze file types within a system or dataset.
It performs comprehensive analysis, generates insights, and returns detailed reports
about file type distributions, potential issues, and optimization recommendations.
"""

from scripts.state_manager import StateManager
from typing import Dict, Any
import logging

__description__ = "Analyze data for file types and generate insights"
__parameters__ = {}
__examples__ = [
    {"description": "Basic file type analysis", "parameters": {}},
]

async def analyze_file_types(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of file types in the system.
    
    Returns:
        Dict[str, Any]: Analysis report containing statistics, insights,
                        and recommendations about file types.
        
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If analysis cannot be completed
    """
    try:
        # Initialize state manager and load current state
        state_manager = StateManager()
        state = state_manager.load_state()
        
        # Validate no unexpected parameters were provided
        if kwargs:
            raise ValueError("This tool doesn't accept any parameters")
        
        # Get file data from state (example implementation)
        files = state.get('files', [])
        
        if not files:
            return {
                "total_files": 0,
                "summary": "No files found for analysis",
                "recommendations": []
            }
        
        # Perform analysis
        file_types = {}
        suspicious_files = []
        large_files = []
        
        for file in files:
            # Count file types
            file_type = file.get('type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Identify potentially suspicious files
            if file_type in ('exe', 'dll', 'bat', 'sh'):
                suspicious_files.append(file.get('path'))
                
            # Identify large files (>100MB)
            if file.get('size', 0) > 100 * 1024 * 1024:  # 100MB
                large_files.append({
                    'path': file.get('path'),
                    'size': file.get('size'),
                    'type': file_type
                })
        
        # Generate insights and recommendations
        total_files = len(files)
        most_common_type = max(file_types.items(), key=lambda x: x[1])[0]
        unique_types = len(file_types)
        
        recommendations = []
        if suspicious_files:
            recommendations.append({
                'issue': 'Potentially suspicious files detected',
                'count': len(suspicious_files),
                'action': 'Review these files for security risks',
                'items': suspicious_files[:10]  # Show first 10 as examples
            })
        
        if large_files:
            recommendations.append({
                'issue': 'Large files detected',
                'count': len(large_files),
                'action': 'Consider archiving or compressing these files',
                'items': large_files[:5]  # Show first 5 as examples
            })
        
        # Prepare final report
        report = {
            "total_files": total_files,
            "file_type_distribution": file_types,
            "most_common_type": most_common_type,
            "unique_file_types": unique_types,
            "suspicious_files_count": len(suspicious_files),
            "large_files_count": len(large_files),
            "summary": f"Analyzed {total_files} files with {unique_types} different types",
            "recommendations": recommendations
        }
        
        return report
        
    except ValueError as ve:
        logging.error(f"Parameter validation error: {str(ve)}")
        raise
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise RuntimeError(f"File type analysis failed: {str(e)}")
