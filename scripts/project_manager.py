"""
Project Manager - Stub module for custom tools
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class ProjectManager:
    """Simple project manager for custom tools"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or ".")
    
    def find_files(self, pattern: str = "*.py", recursive: bool = True) -> List[Path]:
        """Find files matching pattern"""
        if recursive:
            return list(self.project_root.rglob(pattern))
        else:
            return list(self.project_root.glob(pattern))
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get basic project information"""
        return {
            "root": str(self.project_root),
            "name": self.project_root.name,
            "python_files": len(self.find_files("*.py")),
            "directories": len([d for d in self.project_root.rglob("*") if d.is_dir()])
        }
    
    def read_file(self, file_path: str) -> str:
        """Read file contents"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path.read_text()
    
    def write_file(self, file_path: str, content: str):
        """Write content to file"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


# Default instance
project_manager = ProjectManager()