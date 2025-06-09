"""
Data Export Service Module

Provides comprehensive data export functionality supporting CSV, JSON, and PDF formats.
Exports task data, performance metrics, repository analytics, and system state information.
"""

import json
import csv
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import pandas as pd
from io import StringIO

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state_manager import StateManager
from task_manager import TaskManager, TaskStatus, TaskPriority, TaskType

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class ExportFormat(Enum):
    """Export format enumeration."""
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"


class DataType(Enum):
    """Data type enumeration for export."""
    TASKS = "tasks"
    PERFORMANCE = "performance"
    REPOSITORIES = "repositories"
    ANALYTICS = "analytics"
    SYSTEM_STATE = "system_state"
    ALL = "all"


class DataExportService:
    """Provides comprehensive data export functionality."""
    
    def __init__(self, output_dir: str = "exports"):
        """Initialize the data export service.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir
        self.state_manager = StateManager()
        self.task_manager = TaskManager()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_data(self, 
                   data_type: DataType, 
                   export_format: ExportFormat,
                   filename: Optional[str] = None,
                   filters: Optional[Dict[str, Any]] = None) -> str:
        """Export data in the specified format.
        
        Args:
            data_type: Type of data to export
            export_format: Format for export (CSV, JSON, PDF)
            filename: Custom filename (optional)
            filters: Optional filters to apply to the data
            
        Returns:
            Path to the exported file
            
        Raises:
            ValueError: If PDF export is requested but reportlab is not available
            NotImplementedError: If unsupported combination is requested
        """
        if export_format == ExportFormat.PDF and not PDF_AVAILABLE:
            raise ValueError("PDF export requires reportlab library. Please install: pip install reportlab")
            
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type.value}_{timestamp}.{export_format.value}"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Get data based on type
        data = self._get_data_by_type(data_type, filters)
        
        # Export based on format
        if export_format == ExportFormat.JSON:
            self._export_json(data, filepath)
        elif export_format == ExportFormat.CSV:
            self._export_csv(data, data_type, filepath)
        elif export_format == ExportFormat.PDF:
            self._export_pdf(data, data_type, filepath)
        else:
            raise NotImplementedError(f"Export format {export_format.value} not implemented")
            
        return filepath
    
    def _get_data_by_type(self, data_type: DataType, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get data based on the specified type.
        
        Args:
            data_type: Type of data to retrieve
            filters: Optional filters to apply
            
        Returns:
            Dictionary containing the requested data
        """
        if data_type == DataType.TASKS:
            return self._get_task_data(filters)
        elif data_type == DataType.PERFORMANCE:
            return self._get_performance_data(filters)
        elif data_type == DataType.REPOSITORIES:
            return self._get_repository_data(filters)
        elif data_type == DataType.ANALYTICS:
            return self._get_analytics_data(filters)
        elif data_type == DataType.SYSTEM_STATE:
            return self._get_system_state_data(filters)
        elif data_type == DataType.ALL:
            return self._get_all_data(filters)
        else:
            raise ValueError(f"Unknown data type: {data_type.value}")
    
    def _get_task_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get task-related data."""
        task_state = self.task_manager.state
        tasks = task_state.get("tasks", {})
        
        # Apply filters if provided
        if filters:
            filtered_tasks = {}
            for task_id, task in tasks.items():
                include_task = True
                
                # Filter by status
                if "status" in filters and task.get("status") != filters["status"]:
                    include_task = False
                    
                # Filter by priority
                if "priority" in filters and task.get("priority") != filters["priority"]:
                    include_task = False
                    
                # Filter by date range
                if "start_date" in filters or "end_date" in filters:
                    task_date = task.get("created_at")
                    if task_date:
                        task_datetime = datetime.fromisoformat(task_date.replace('Z', '+00:00'))
                        if "start_date" in filters:
                            start_date = datetime.fromisoformat(filters["start_date"])
                            if task_datetime < start_date:
                                include_task = False
                        if "end_date" in filters:
                            end_date = datetime.fromisoformat(filters["end_date"])
                            if task_datetime > end_date:
                                include_task = False
                
                if include_task:
                    filtered_tasks[task_id] = task
            tasks = filtered_tasks
        
        return {
            "tasks": tasks,
            "summary": {
                "total_tasks": len(tasks),
                "active_tasks": task_state.get("active_tasks", 0),
                "completed_today": task_state.get("completed_today", 0),
                "success_rate": task_state.get("success_rate", 0.0),
                "last_updated": task_state.get("last_updated"),
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def _get_performance_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get performance metrics data."""
        system_state = self.state_manager.load_state()
        task_state = self.task_manager.state
        
        performance_data = {
            "system_performance": system_state.get("system_performance", {}),
            "task_metrics": {
                "active_tasks": task_state.get("active_tasks", 0),
                "completed_today": task_state.get("completed_today", 0),
                "success_rate": task_state.get("success_rate", 0.0),
                "task_counter": task_state.get("task_counter", 0)
            },
            "repository_health": {},
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add repository health scores
        projects = system_state.get("projects", {})
        for project_name, project_data in projects.items():
            performance_data["repository_health"][project_name] = {
                "health_score": project_data.get("health_score", 0),
                "metrics": project_data.get("metrics", {}),
                "recent_activity": project_data.get("recent_activity", {})
            }
        
        return performance_data
    
    def _get_repository_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get repository-related data."""
        system_state = self.state_manager.load_state()
        projects = system_state.get("projects", {})
        
        # Apply filters if provided
        if filters:
            filtered_projects = {}
            for project_name, project_data in projects.items():
                include_project = True
                
                # Filter by health score
                if "min_health_score" in filters:
                    health_score = project_data.get("health_score", 0)
                    if health_score < filters["min_health_score"]:
                        include_project = False
                
                # Filter by language
                if "language" in filters:
                    project_language = project_data.get("language")
                    if project_language != filters["language"]:
                        include_project = False
                
                if include_project:
                    filtered_projects[project_name] = project_data
            projects = filtered_projects
        
        return {
            "projects": projects,
            "discovery_info": system_state.get("repository_discovery", {}),
            "summary": {
                "total_repositories": len(projects),
                "active_repositories": len([p for p in projects.values() if p.get("status") == "active"]),
                "average_health_score": sum(p.get("health_score", 0) for p in projects.values()) / len(projects) if projects else 0
            },
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_analytics_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get analytics and learning metrics data."""
        system_state = self.state_manager.load_state()
        
        return {
            "learning_metrics": system_state.get("system_performance", {}).get("learning_metrics", {}),
            "charter": system_state.get("charter", {}),
            "external_context": system_state.get("external_context", {}),
            "task_queue": system_state.get("task_queue", []),
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_system_state_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get complete system state data."""
        system_state = self.state_manager.load_state()
        task_state = self.task_manager.state
        
        return {
            "system_state": system_state,
            "task_state": task_state,
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_all_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get all data types combined."""
        return {
            "tasks": self._get_task_data(filters),
            "performance": self._get_performance_data(filters),
            "repositories": self._get_repository_data(filters),
            "analytics": self._get_analytics_data(filters),
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _export_json(self, data: Dict[str, Any], filepath: str) -> None:
        """Export data as JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_csv(self, data: Dict[str, Any], data_type: DataType, filepath: str) -> None:
        """Export data as CSV."""
        if data_type == DataType.TASKS:
            self._export_tasks_csv(data, filepath)
        elif data_type == DataType.PERFORMANCE:
            self._export_performance_csv(data, filepath)
        elif data_type == DataType.REPOSITORIES:
            self._export_repositories_csv(data, filepath)
        elif data_type == DataType.ANALYTICS:
            self._export_analytics_csv(data, filepath)
        else:
            # For complex data types, flatten and export as CSV
            self._export_flattened_csv(data, filepath)
    
    def _export_tasks_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """Export task data as CSV."""
        tasks = data.get("tasks", {})
        
        # Flatten task data for CSV
        csv_data = []
        for task_id, task in tasks.items():
            row = {
                "task_id": task_id,
                "title": task.get("title", ""),
                "status": task.get("status", ""),
                "priority": task.get("priority", ""),
                "type": task.get("type", ""),
                "created_at": task.get("created_at", ""),
                "updated_at": task.get("updated_at", ""),
                "estimated_hours": task.get("estimated_hours", 0),
                "dependencies": ", ".join(task.get("dependencies", [])),
                "labels": ", ".join(task.get("labels", []))
            }
            csv_data.append(row)
        
        # Write CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
        else:
            # Create empty CSV with headers
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["task_id", "title", "status", "priority", "type", 
                               "created_at", "updated_at", "estimated_hours", "dependencies", "labels"])
    
    def _export_performance_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """Export performance data as CSV."""
        csv_data = []
        
        # System performance metrics
        system_perf = data.get("system_performance", {})
        task_metrics = data.get("task_metrics", {})
        
        row = {
            "metric_type": "system_overview",
            "total_cycles": system_perf.get("total_cycles", 0),
            "successful_actions": system_perf.get("successful_actions", 0),
            "failed_actions": system_perf.get("failed_actions", 0),
            "active_tasks": task_metrics.get("active_tasks", 0),
            "completed_today": task_metrics.get("completed_today", 0),
            "success_rate": task_metrics.get("success_rate", 0.0),
            "export_timestamp": data.get("export_timestamp", "")
        }
        csv_data.append(row)
        
        # Repository health data
        repo_health = data.get("repository_health", {})
        for repo_name, health_data in repo_health.items():
            row = {
                "metric_type": "repository_health",
                "repository": repo_name,
                "health_score": health_data.get("health_score", 0),
                "stars": health_data.get("metrics", {}).get("stars", 0),
                "forks": health_data.get("metrics", {}).get("forks", 0),
                "issues_open": health_data.get("metrics", {}).get("issues_open", 0),
                "recent_commits": health_data.get("recent_activity", {}).get("recent_commits", 0)
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _export_repositories_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """Export repository data as CSV."""
        projects = data.get("projects", {})
        csv_data = []
        
        for project_name, project_data in projects.items():
            metrics = project_data.get("metrics", {})
            recent_activity = project_data.get("recent_activity", {})
            
            row = {
                "name": project_name,
                "full_name": project_data.get("full_name", ""),
                "description": project_data.get("description", ""),
                "language": project_data.get("language", ""),
                "health_score": project_data.get("health_score", 0),
                "status": project_data.get("status", ""),
                "stars": metrics.get("stars", 0),
                "forks": metrics.get("forks", 0),
                "issues_open": metrics.get("issues_open", 0),
                "watchers": metrics.get("watchers", 0),
                "recent_commits": recent_activity.get("recent_commits", 0),
                "contributors_count": recent_activity.get("contributors_count", 0),
                "last_commit_date": recent_activity.get("last_commit_date", ""),
                "last_checked": project_data.get("last_checked", "")
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _export_analytics_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """Export analytics data as CSV."""
        learning_metrics = data.get("learning_metrics", {})
        
        csv_data = [{
            "decision_accuracy": learning_metrics.get("decision_accuracy", 0.0),
            "goal_achievement": learning_metrics.get("goal_achievement", 0.0),
            "resource_efficiency": learning_metrics.get("resource_efficiency", 0.0),
            "export_timestamp": data.get("export_timestamp", "")
        }]
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _export_flattened_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """Export complex data as flattened CSV."""
        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    items.append((new_key, ', '.join(map(str, v))))
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flattened = flatten_dict(data)
        df = pd.DataFrame([flattened])
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _export_pdf(self, data: Dict[str, Any], data_type: DataType, filepath: str) -> None:
        """Export data as PDF."""
        if not PDF_AVAILABLE:
            raise ValueError("PDF export requires reportlab library")
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB')
        )
        
        title = f"CWMAI Data Export - {data_type.value.title()}"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Export info
        export_time = data.get("export_timestamp", datetime.now(timezone.utc).isoformat())
        story.append(Paragraph(f"<b>Export Date:</b> {export_time}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        if data_type == DataType.TASKS:
            self._add_tasks_to_pdf(story, data, styles)
        elif data_type == DataType.PERFORMANCE:
            self._add_performance_to_pdf(story, data, styles)
        elif data_type == DataType.REPOSITORIES:
            self._add_repositories_to_pdf(story, data, styles)
        elif data_type == DataType.ANALYTICS:
            self._add_analytics_to_pdf(story, data, styles)
        else:
            self._add_general_data_to_pdf(story, data, styles)
        
        doc.build(story)
    
    def _add_tasks_to_pdf(self, story: List, data: Dict[str, Any], styles) -> None:
        """Add task data to PDF."""
        summary = data.get("summary", {})
        tasks = data.get("tasks", {})
        
        # Summary section
        story.append(Paragraph("<b>Task Summary</b>", styles['Heading2']))
        story.append(Paragraph(f"Total Tasks: {summary.get('total_tasks', 0)}", styles['Normal']))
        story.append(Paragraph(f"Active Tasks: {summary.get('active_tasks', 0)}", styles['Normal']))
        story.append(Paragraph(f"Completed Today: {summary.get('completed_today', 0)}", styles['Normal']))
        story.append(Paragraph(f"Success Rate: {summary.get('success_rate', 0):.1%}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Tasks table
        if tasks:
            story.append(Paragraph("<b>Task Details</b>", styles['Heading2']))
            
            table_data = [["Task ID", "Title", "Status", "Priority", "Type", "Created"]]
            for task_id, task in list(tasks.items())[:20]:  # Limit to first 20 tasks
                table_data.append([
                    task_id,
                    task.get("title", "")[:30] + "..." if len(task.get("title", "")) > 30 else task.get("title", ""),
                    task.get("status", ""),
                    task.get("priority", ""),
                    task.get("type", ""),
                    task.get("created_at", "")[:10] if task.get("created_at") else ""
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
    def _add_performance_to_pdf(self, story: List, data: Dict[str, Any], styles) -> None:
        """Add performance data to PDF."""
        system_perf = data.get("system_performance", {})
        task_metrics = data.get("task_metrics", {})
        
        story.append(Paragraph("<b>System Performance</b>", styles['Heading2']))
        story.append(Paragraph(f"Total Cycles: {system_perf.get('total_cycles', 0)}", styles['Normal']))
        story.append(Paragraph(f"Successful Actions: {system_perf.get('successful_actions', 0)}", styles['Normal']))
        story.append(Paragraph(f"Failed Actions: {system_perf.get('failed_actions', 0)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>Task Metrics</b>", styles['Heading2']))
        story.append(Paragraph(f"Active Tasks: {task_metrics.get('active_tasks', 0)}", styles['Normal']))
        story.append(Paragraph(f"Completed Today: {task_metrics.get('completed_today', 0)}", styles['Normal']))
        story.append(Paragraph(f"Success Rate: {task_metrics.get('success_rate', 0):.1%}", styles['Normal']))
        
    def _add_repositories_to_pdf(self, story: List, data: Dict[str, Any], styles) -> None:
        """Add repository data to PDF."""
        summary = data.get("summary", {})
        projects = data.get("projects", {})
        
        story.append(Paragraph("<b>Repository Summary</b>", styles['Heading2']))
        story.append(Paragraph(f"Total Repositories: {summary.get('total_repositories', 0)}", styles['Normal']))
        story.append(Paragraph(f"Active Repositories: {summary.get('active_repositories', 0)}", styles['Normal']))
        story.append(Paragraph(f"Average Health Score: {summary.get('average_health_score', 0):.1f}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        if projects:
            story.append(Paragraph("<b>Repository Details</b>", styles['Heading2']))
            
            table_data = [["Name", "Language", "Health Score", "Stars", "Forks", "Issues"]]
            for project_name, project_data in list(projects.items())[:15]:  # Limit to first 15 repos
                metrics = project_data.get("metrics", {})
                table_data.append([
                    project_name[:20] + "..." if len(project_name) > 20 else project_name,
                    project_data.get("language", "N/A"),
                    str(project_data.get("health_score", 0)),
                    str(metrics.get("stars", 0)),
                    str(metrics.get("forks", 0)),
                    str(metrics.get("issues_open", 0))
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
    
    def _add_analytics_to_pdf(self, story: List, data: Dict[str, Any], styles) -> None:
        """Add analytics data to PDF."""
        learning_metrics = data.get("learning_metrics", {})
        charter = data.get("charter", {})
        
        story.append(Paragraph("<b>Learning Metrics</b>", styles['Heading2']))
        story.append(Paragraph(f"Decision Accuracy: {learning_metrics.get('decision_accuracy', 0):.1%}", styles['Normal']))
        story.append(Paragraph(f"Goal Achievement: {learning_metrics.get('goal_achievement', 0):.1%}", styles['Normal']))
        story.append(Paragraph(f"Resource Efficiency: {learning_metrics.get('resource_efficiency', 0):.1%}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>System Charter</b>", styles['Heading2']))
        story.append(Paragraph(f"Primary Goal: {charter.get('primary_goal', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Secondary Goal: {charter.get('secondary_goal', 'N/A')}", styles['Normal']))
        constraints = charter.get("constraints", [])
        if constraints:
            story.append(Paragraph(f"Constraints: {', '.join(constraints)}", styles['Normal']))
    
    def _add_general_data_to_pdf(self, story: List, data: Dict[str, Any], styles) -> None:
        """Add general data to PDF."""
        story.append(Paragraph("<b>Data Summary</b>", styles['Heading2']))
        
        def add_dict_to_story(d, level=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    story.append(Paragraph(f"{'  ' * level}<b>{key}:</b>", styles['Normal']))
                    add_dict_to_story(value, level + 1)
                elif isinstance(value, list):
                    story.append(Paragraph(f"{'  ' * level}<b>{key}:</b> {len(value)} items", styles['Normal']))
                else:
                    story.append(Paragraph(f"{'  ' * level}<b>{key}:</b> {str(value)[:100]}...", styles['Normal']))
        
        add_dict_to_story(data)
    
    def get_export_performance_benchmark(self, data_type: DataType, export_format: ExportFormat) -> Dict[str, Any]:
        """Benchmark export performance for the specified data type and format.
        
        Args:
            data_type: Type of data to benchmark
            export_format: Format to benchmark
            
        Returns:
            Dictionary containing performance metrics
        """
        import time
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Perform export
        try:
            filepath = self.export_data(data_type, export_format)
            file_size = os.path.getsize(filepath)
            success = True
        except Exception as e:
            file_size = 0
            success = False
            filepath = str(e)
        
        end_time = time.time()
        memory_after = self._get_memory_usage()
        
        return {
            "data_type": data_type.value,
            "export_format": export_format.value,
            "execution_time_seconds": round(end_time - start_time, 3),
            "memory_used_mb": round((memory_after - memory_before) / 1024 / 1024, 2),
            "file_size_bytes": file_size,
            "success": success,
            "output_file": filepath,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            return 0


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CWMAI Data Export Service")
    parser.add_argument("--data-type", choices=[dt.value for dt in DataType], 
                       default="all", help="Type of data to export")
    parser.add_argument("--format", choices=[ef.value for ef in ExportFormat], 
                       default="json", help="Export format")
    parser.add_argument("--output-dir", default="exports", 
                       help="Output directory for exported files")
    parser.add_argument("--filename", help="Custom filename for export")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Create export service
    export_service = DataExportService(output_dir=args.output_dir)
    
    data_type = DataType(args.data_type)
    export_format = ExportFormat(args.format)
    
    if args.benchmark:
        print("Running performance benchmark...")
        benchmark = export_service.get_export_performance_benchmark(data_type, export_format)
        print(f"Benchmark Results:")
        for key, value in benchmark.items():
            print(f"  {key}: {value}")
    else:
        print(f"Exporting {data_type.value} data as {export_format.value}...")
        try:
            filepath = export_service.export_data(data_type, export_format, args.filename)
            print(f"Export completed successfully: {filepath}")
        except Exception as e:
            print(f"Export failed: {e}")


if __name__ == "__main__":
    main()