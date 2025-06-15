"""
Formatter - Stub module for custom tools
"""

import json
from typing import Any, Dict, List
import textwrap


class Formatter:
    """Simple formatter for custom tools"""
    
    @staticmethod
    def format_json(data: Any, indent: int = 2) -> str:
        """Format data as pretty JSON"""
        return json.dumps(data, indent=indent, sort_keys=True)
    
    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]]) -> str:
        """Format data as a simple text table"""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Build table
        lines = []
        
        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            lines.append(row_line)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_markdown(title: str, sections: Dict[str, str]) -> str:
        """Format content as markdown"""
        lines = [f"# {title}", ""]
        
        for section_title, content in sections.items():
            lines.extend([f"## {section_title}", "", content, ""])
        
        return "\n".join(lines)
    
    @staticmethod
    def wrap_text(text: str, width: int = 80) -> str:
        """Wrap text to specified width"""
        return textwrap.fill(text, width=width)


# Default instance
formatter = Formatter()