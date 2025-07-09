"""
Code Documentation Generator
Automatically generates documentation for Python code files.
"""

import ast
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import textwrap
import os
import hashlib
import logging
import asyncio  # Required for asynchronous operation
import concurrent.futures
import re
import shutil  # For file operations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DocGenerator(ast.NodeVisitor):
    """AST visitor to extract documentation info."""
    
    def __init__(self):
        self.module_doc = None
        self.classes = []
        self.functions = []
        self.current_class = None
        self.errors = []  # Collect parsing errors
        
    def visit_Module(self, node):
        """Extract module docstring."""
        try:
            if ast.get_docstring(node):
                self.module_doc = ast.get_docstring(node)
        except Exception as e:
            self.errors.append(f"Error extracting module docstring: {e}")
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Extract class information."""
        try:
            class_info = {
                'name': node.name,
                'docstring': ast.get_docstring(node),
                'methods': [],
                'base_classes': [base.id for base in node.bases if hasattr(base, 'id')],
                'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')], # Include class decorators
                'lineno': node.lineno,
                'col_offset': node.col_offset
            }
            
            # Store current class context
            old_class = self.current_class
            self.current_class = class_info
            
            # Visit class body
            self.generic_visit(node)
            
            # Restore context
            self.current_class = old_class
            self.classes.append(class_info)
        except Exception as e:
            self.errors.append(f"Error extracting class info for {node.name} at line {node.lineno}, column {node.col_offset}: {e}")
        
    def visit_FunctionDef(self, node):
        """Extract function information."""
        try:
            func_info = {
                'name': node.name,
                'docstring': ast.get_docstring(node),
                'args': [arg.arg for arg in node.args.args],
                'returns': None,
                'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')],
                'lineno': node.lineno,  # Add line number for better error reporting
                'col_offset': node.col_offset
            }
            
            # Extract return type annotation
            if node.returns:
                if hasattr(node.returns, 'id'):
                    func_info['returns'] = node.returns.id
                elif hasattr(node.returns, 'value'):
                    func_info['returns'] = ast.unparse(node.returns)
            
            if self.current_class:
                self.current_class['methods'].append(func_info)
            else:
                self.functions.append(func_info)
        except Exception as e:
            self.errors.append(f"Error extracting function info for {node.name} at line {node.lineno}, column {node.col_offset}: {e}")
            
    visit_AsyncFunctionDef = visit_FunctionDef


def is_valid_file_path(file_path: str) -> bool:
    """Validate a file path."""
    path = Path(file_path)
    return path.exists() and path.is_file() and path.suffix == '.py'


def calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculate the MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except PermissionError:
        logging.error(f"Permission error accessing {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error calculating file hash for {file_path}: {e}")
        return None


def load_cache(cache_file: str) -> Dict[str, Any]:
    """Load cached file hashes and documentation from a JSON file."""
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Cache file {cache_file} not found. Creating a new one.")
        return {}
    except json.JSONDecodeError:
        logging.warning(f"Invalid JSON in cache file {cache_file}, resetting cache.")
        return {}  # Reset cache if corrupted
    except Exception as e:
        logging.error(f"Error loading cache file {cache_file}: {e}")
        return {}


def save_cache(cache: Dict[str, Any], cache_file: str) -> None:
    """Save file hashes and documentation to a JSON file."""
    try:
        # Ensure cache directory exists
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except PermissionError:
        logging.error(f"Permission error saving cache file {cache_file}.")
    except Exception as e:
        logging.error(f"Error saving cache file {cache_file}: {e}")


def validate_directory(directory: str) -> bool:
    """Validate a directory path."""
    path = Path(directory)
    return path.exists() and path.is_dir()


def cleanup_docstring(docstring: str) -> str:
    """Remove leading/trailing whitespace and normalize indentation."""
    if not docstring:
        return ""
    return textwrap.dedent(docstring).strip()


def extract_parameter_details(docstring: str) -> Dict[str, str]:
    """Extract parameter details (type, description) from a docstring (e.g., Google, NumPy style)."""
    params = {}
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Google style
        if line.startswith(":param"):
            parts = line.split(":", 2)
            if len(parts) > 2:
                param_name_type = parts[1].strip().split(" ", 1)
                if len(param_name_type) == 2:
                    param_name, param_type = param_name_type
                else:
                    param_name = param_name_type[0]
                    param_type = "unknown"

                description = parts[2].strip()
                i += 1
                while i < len(lines) and lines[i].strip().startswith((":", " ")):  # handles multi-line description
                    if not lines[i].strip().startswith(":"):
                        description += " " + lines[i].strip()
                    i += 1

                params[param_name] = {"type": param_type, "description": description}
                continue  # Skip increment since we did it inside the loop

        # NumPy style
        elif line.lower().startswith("parameters") or line.lower().startswith("arguments"):
            i += 1
            while i < len(lines):
                param_line = lines[i].strip()
                if not param_line:
                    i += 1
                    continue

                match = re.match(r"(\w+)\s*:\s*(.+)", param_line)
                if match:
                    param_name = match.group(1).strip()
                    param_details = match.group(2).strip()

                    # Try to split into type and description
                    type_desc = param_details.split(",", 1)
                    if len(type_desc) == 2:
                        param_type = type_desc[0].strip()
                        description = type_desc[1].strip()
                    else:
                        param_type = "unknown"
                        description = type_desc[0].strip()

                    params[param_name] = {"type": param_type, "description": description}

                i += 1
                if i < len(lines) and not lines[i].strip():
                    i += 1  # Skip empty lines after param

                if i >= len(lines) or lines[i].strip().startswith(":"):
                    break # Stop parsing params

                continue

        elif line.startswith(":return:") or line.startswith(":returns:"): # Corrected key
            i += 1 # Skip line starting with :return: or :returns:
            continue

        i += 1 # Increment loop for non-relevant lines
    return params


def generate_table_of_contents(results: Dict[str, Any], format: str) -> str:
    """Generate a table of contents based on the processed files."""
    toc_lines = ["# Table of Contents\n" if format == "markdown" else "Table of Contents\n=================\n"]
    generated = results.get('generated', [])
    for file in generated:
        if format == "markdown":
            toc_lines.append(f"- [{file}](./{file.replace('.py', f'.{format_to_extension(format)}')})\n") # corrected suffix replacement and relative path
        else:  # rst
            toc_lines.append(f"* :doc:`{file.replace('.py', '')}`\n")

    return "".join(toc_lines)

async def generate_documentation(file_path: str, format: str = "markdown", use_cache: bool = True, cache_file: str = ".doc_cache.json") -> str:
    """
    Generate documentation for a Python file.
    
    Args:
        file_path: Path to Python file
        format: Output format ("markdown", "rst", "json")
        use_cache: Whether to use caching for performance
        cache_file: File to store cache data
        
    Returns:
        Generated documentation
    """
    # Input validation
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")
    if format not in ["markdown", "rst", "json", "html"]:  # Added HTML format
        raise ValueError("Unsupported format.")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be a boolean.")
    if not isinstance(cache_file, str):
        raise TypeError("cache_file must be a string.")

    if not is_valid_file_path(file_path):
        raise FileNotFoundError(f"Error: {file_path} is not a valid Python file")
    
    cache = load_cache(cache_file) if use_cache else {}
    file_hash = calculate_file_hash(file_path)
    cache_key = f"{file_path}_{format}" # Unique cache key per format
    
    if use_cache and file_hash and cache_key in cache and cache[cache_key].get('hash') == file_hash:
        logging.info(f"Using cached documentation for {file_path} in {format} format.")
        return cache[cache_key].get('doc', "")

    try:
        path = Path(file_path)
        content = path.read_text()
        tree = ast.parse(content)
        
        # Extract documentation
        visitor = DocGenerator()
        visitor.visit(tree)
        
        if visitor.errors:
            logging.warning(f"Errors found while parsing {file_path}: {visitor.errors}")

        # Generate output based on format
        if format == "markdown":
            doc = _generate_markdown(path.name, visitor)
        elif format == "rst":
            doc = _generate_rst(path.name, visitor)
        elif format == "json":
            doc = _generate_json(visitor)
        elif format == "html":
            doc = _generate_html(path.name, visitor) # Added HTML generation
        else:
            raise ValueError("Error: Unsupported format")

        # Save to cache
        if use_cache and file_hash:
            cache[cache_key] = {'hash': file_hash, 'doc': doc}
            save_cache(cache, cache_file)
            logging.info(f"Documentation for {file_path} in {format} format saved to cache.")
            
        return doc
            
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}: {e}")
        return f"Error: File not found: {file_path}"
    except SyntaxError as e:
        logging.error(f"Syntax error in {file_path}: {e}")
        return f"Error: Syntax error in {file_path}: {e}"
    except Exception as e:
        logging.exception(f"Error generating documentation for {file_path}: {str(e)}")  # Log the full traceback
        return f"Error generating documentation: {str(e)}"


def _generate_markdown(filename: str, visitor: DocGenerator) -> str:
    """Generate Markdown documentation."""
    lines = [f"# {filename}\n"]
    
    if visitor.module_doc:
        lines.append(f"{cleanup_docstring(visitor.module_doc)}\n")
        
    # Functions
    if visitor.functions:
        lines.append("## Functions\n")
        for func in visitor.functions:
            lines.append(f"### {func['name']}\n")
            
            # Signature
            args = ', '.join(func['args'])
            returns = f" -> {func['returns']}" if func['returns'] else ""
            lines.append(f"```python")
            lines.append(f"def {func['name']}({args}){returns}")
            lines.append(f"```\n")

            # Extract parameter details from docstring
            param_details = extract_parameter_details(func['docstring'] or "")
            if param_details:
                lines.append("#### Parameters\n")
                lines.append("| Name | Type | Description |\n")
                lines.append("|---|---|---|\n")
                for param_name, details in param_details.items():
                    lines.append(f"| {param_name} | {details['type']} | {details['description']} |\n")

            if func['docstring']:
                lines.append(f"{cleanup_docstring(func['docstring'])}\n")

            if func['decorators']:
                lines.append(f"**Decorators:** {', '.join(func['decorators'])}\n")
                
    # Classes
    if visitor.classes:
        lines.append("## Classes\n")
        for cls in visitor.classes:
            lines.append(f"### {cls['name']}\n")
            
            if cls['base_classes']:
                lines.append(f"**Inherits from:** {', '.join(cls['base_classes'])}\n")

            if cls['decorators']:
                lines.append(f"**Decorators:** {', '.join(cls['decorators'])}\n")
                
            if cls['docstring']:
                lines.append(f"{cleanup_docstring(cls['docstring'])}\n")
                
            # Methods
            if cls['methods']:
                lines.append("#### Methods\n")
                for method in cls['methods']:
                    args = ', '.join(method['args'])
                    returns = f" -> {method['returns']}" if method['returns'] else ""
                    
                    lines.append(f"**{method['name']}**({args}){returns}\n")

                    # Extract parameter details from docstring
                    param_details = extract_parameter_details(method['docstring'] or "")
                    if param_details:
                        lines.append("##### Parameters\n")
                        lines.append("| Name | Type | Description |\n")
                        lines.append("|---|---|---|\n")
                        for param_name, details in param_details.items():
                            lines.append(f"| {param_name} | {details['type']} | {details['description']} |\n")
                    
                    if method['docstring']:
                        lines.append(f"{cleanup_docstring(method['docstring'])}\n")
                        
    return '\n'.join(lines)


def _generate_rst(filename: str, visitor: DocGenerator) -> str:
    """Generate reStructuredText documentation."""
    lines = []
    
    # Title
    title = filename
    lines.append(title)
    lines.append('=' * len(title))
    lines.append('')
    
    if visitor.module_doc:
        lines.append(cleanup_docstring(visitor.module_doc))
        lines.append('')
        
    # Functions
    if visitor.functions:
        lines.append("Functions")
        lines.append("---------")
        lines.append('')
        
        for func in visitor.functions:
            lines.append(f".. function:: {func['name']}({', '.join(func['args'])})")
            lines.append('')

            # Extract parameter details from docstring
            param_details = extract_parameter_details(func['docstring'] or "") # Added parameter details to RST format
            if param_details:
                lines.append("   :Parameters:")
                lines.append("")
                for param_name, details in param_details.items():
                    lines.append(f"      {param_name} ({details['type']}): {details['description']}")
                lines.append("")
            
            if func['docstring']:
                docstring = cleanup_docstring(func['docstring'])
                for line in docstring.split('\n'):
                    lines.append(f"   {line}")
                lines.append('')
                
    # Classes
    if visitor.classes:
        lines.append("Classes")
        lines.append("-------")
        lines.append('')
        
        for cls in visitor.classes:
            lines.append(f".. class:: {cls['name']}")
            lines.append('')
            
            if cls['docstring']:
                docstring = cleanup_docstring(cls['docstring'])
                for line in docstring.split('\n'):
                    lines.append(f"   {line}")
                lines.append('')
                
    return '\n'.join(lines)


def _generate_json(visitor: DocGenerator) -> str:
    """Generate JSON documentation."""
    doc = {
        'module_docstring': cleanup_docstring(visitor.module_doc) if visitor.module_doc else None,
        'functions': [
            {k: (cleanup_docstring(v) if k == 'docstring' and v else v) for k, v in func.items()}
            for func in visitor.functions
        ],
        'classes': [
            {
                **cls,
                'docstring': cleanup_docstring(cls['docstring']) if cls['docstring'] else None,
                'methods': [
                    {k: (cleanup_docstring(v) if k == 'docstring' and v else v) for k, v in method.items()}
                    for method in cls['methods']
                ]
            }
            for cls in visitor.classes
        ]
    }
    return json.dumps(doc, indent=2)

def _generate_html(filename: str, visitor: DocGenerator) -> str:
    """Generate HTML documentation."""
    html_lines = ["<!DOCTYPE html>",
                  "<html lang='en'>",
                  "<head>",
                  f"<title>Documentation for {filename}</title>",
                  "<style>",
                  """
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    h2 { color: #444; margin-top: 30px; }
                    h3 { color: #555; margin-top: 20px; }
                    h4 { color: #666; margin-top: 15px; }
                    p { line-height: 1.6; }
                    pre { background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; overflow: auto; }
                    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f0f0f0; }
                    .decorator { font-style: italic; color: #777; }
                  """,
                  "</style>",
                  "</head>",
                  "<body>"]

    html_lines.append(f"<h1>{filename}</h1>")

    if visitor.module_doc:
        html_lines.append(f"<p>{cleanup_docstring(visitor.module_doc)}</p>")

    # Functions
    if visitor.functions:
        html_lines.append("<h2>Functions</h2>")
        for func in visitor.functions:
            html_lines.append(f"<h3>{func['name']}</h3>")
            args = ', '.join(func['args'])
            returns = f" &rarr; {func['returns']}" if func['returns'] else ""
            html_lines.append(f"<pre><code>def {func['name']}({args}){returns}</code></pre>")

            # Parameter Details
            param_details = extract_parameter_details(func['docstring'] or "")
            if param_details:
                html_lines.append("<h4>Parameters</h4>")
                html_lines.append("<table>")
                html_lines.append("<thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>")
                html_lines.append("<tbody>")
                for param_name, details in param_details.items():
                    html_lines.append(f"<tr><td>{param_name}</td><td>{details['type']}</td><td>{details['description']}</td></tr>")
                html_lines.append("</tbody></table>")

            if func['docstring']:
                html_lines.append(f"<p>{cleanup_docstring(func['docstring'])}</p>")

            if func['decorators']:
                decorators_html = ', '.join([f"<span class='decorator'>@{d}</span>" for d in func['decorators']])
                html_lines.append(f"<p><strong>Decorators:</strong> {decorators_html}</p>")

    # Classes
    if visitor.classes:
        html_lines.append("<h2>Classes</h2>")
        for cls in visitor.classes:
            html_lines.append(f"<h3>{cls['name']}</h3>")

            if cls['base_classes']:
                html_lines.append(f"<p><strong>Inherits from:</strong> {', '.join(cls['base_classes'])}</p>")

            if cls['decorators']:
                decorators_html = ', '.join([f"<span class='decorator'>@{d}</span>" for d in cls['decorators']])
                html_lines.append(f"<p><strong>Decorators:</strong> {decorators_html}</p>")


            if cls['docstring']:
                html_lines.append(f"<p>{cleanup_docstring(cls['docstring'])}</p>")

            # Methods
            if cls['methods']:
                html_lines.append("<h4>Methods</h4>")
                for method in cls['methods']:
                    args = ', '.join(method['args'])
                    returns = f" &rarr; {method['returns']}" if method['returns'] else ""
                    html_lines.append(f"<h5>{method['name']}({args}){returns}</h5>")

                    # Parameter Details
                    param_details = extract_parameter_details(method['docstring'] or "")
                    if param_details:
                        html_lines.append("<h6>Parameters</h6>")
                        html_lines.append("<table>")
                        html_lines.append("<thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>")
                        html_lines.append("<tbody>")
                        for param_name, details in param_details.items():
                            html_lines.append(f"<tr><td>{param_name}</td><td>{details['type']}</td><td>{details['description']}</td></tr>")
                        html_lines.append("</tbody></table>")

                    if method['docstring']:
                        html_lines.append(f"<p>{cleanup_docstring(method['docstring'])}</p>")

    html_lines.append("</body></html>")
    return '\n'.join(html_lines)


async def generate_project_docs(directory: str = ".", output_dir: str = "docs", format: str = "markdown", use_cache: bool = True, cache_file: str = ".doc_cache.json", concurrency: int = 4, create_index: bool = True, clean_output_dir: bool = False) -> Dict[str, Any]:
    """
    Generate documentation for all Python files in a directory.

    Args:
        directory: Source directory
        output_dir: Output directory for docs
        format: Output format ("markdown", "rst", "json", "html")
        use_cache: Whether to use caching for performance
        cache_file: File to store cache data
        concurrency: Number of files to process concurrently
        create_index: Whether to generate index file
        clean_output_dir: Whether to clean the output directory before generation

    Returns:
        Summary of generated documentation
    """
    # Input validation
    if not isinstance(directory, str):
        raise TypeError("directory must be a string.")
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be a string.")
    if format not in ["markdown", "rst", "json", "html"]:
        raise ValueError("Unsupported format.")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be a boolean.")
    if not isinstance(cache_file, str):
        raise TypeError("cache_file must be a string.")
    if not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer.")
    if not isinstance(create_index, bool):
        raise TypeError("create_index must be a boolean.")
    if not isinstance(clean_output_dir, bool):
        raise TypeError("clean_output_dir must be a boolean.")

    results = {
        'generated': [],
        'failed': [],
        'total_files': 0
    }

    source_dir = Path(directory)
    out_dir = Path(output_dir)

    if not validate_directory(directory):
        results['error'] = f"Invalid source directory: {directory}"
        return results

    try:
        if clean_output_dir:
            if out_dir.exists():
                shutil.rmtree(out_dir)  # Delete existing directory
            out_dir.mkdir(parents=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        results['error'] = f"Failed to create output directory: {e}"
        return results

    # Find all Python files
    py_files = list(source_dir.rglob("*.py"))
    results['total_files'] = len(py_files)

    # Prepare tasks for concurrent execution
    async def process_file(py_file: Path) -> None:
        try:
            # Generate documentation
            doc = await generate_documentation(str(py_file), format=format, use_cache=use_cache, cache_file=cache_file)

            # Save to output directory
            relative_path = py_file.relative_to(source_dir)
            doc_path = out_dir / relative_path.with_suffix(f'.{format_to_extension(format)}')  # corrected the suffix
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(doc, encoding="utf-8")  # Ensure UTF-8 encoding

            results['generated'].append(str(relative_path))
            logging.info(f"Generated documentation for {py_file}")

        except Exception as e:
            logging.exception(f"Failed to process {py_file}: {e}")
            results['failed'].append({
                'file': str(py_file),
                'error': str(e)
            })
    # Create a list of tasks
    tasks = [process_file(py_file) for py_file in py_files]

    # Execute tasks concurrently using asyncio.gather with limited concurrency
    semaphore = asyncio.Semaphore(concurrency)
    async def limited_process_file(py_file: Path):
        async with semaphore:
            await process_file(py_file)
    limited_tasks = [limited_process_file(py_file) for py_file in py_files]


    if tasks:  # Ensure there are tasks before running asyncio.gather
        try:
            await asyncio.gather(*limited_tasks, return_exceptions=False) # Do not stop execution if errors occur
        except Exception as e:
            logging.error(f"An error occurred during concurrent execution: {e}")
            results['error'] = str(e)

    # Generate table of contents
    if create_index and not results.get('error'):
        try:
            toc = generate_table_of_contents(results, format)
            toc_path = Path(output_dir) / f"index.{format_to_extension(format)}" # corrected path
            toc_path.write_text(toc, encoding="utf-8") # Ensure UTF-8 encoding
            results['generated'].append(str(toc_path.relative_to(source_dir))) # Store relative path for TOC
        except Exception as e:
            logging.error(f"Failed to generate table of contents: {e}")
            results['error'] = f"Failed to generate table of contents: {e}"
    return results


def format_to_extension(format: str) -> str:
    """Convert documentation format to file extension."""
    if format == "markdown":
        return "md"
    elif format == "rst":
        return "rst"
    elif format == "json":
        return "json"
    elif format == "html":
        return "html"
    else:
        return "txt"  # Default extension

# Export the functions
__all__ = ['generate_documentation', 'generate_project_docs']

# Example usage (within an asyncio event loop)
if __name__ == "__main__":
    async def main():
        # Example 1: Generate documentation for a single file
        try:
            doc = await generate_documentation("example.py", format="markdown", use_cache=True)
            print(doc)
        except Exception as e:
            print(f"Error: {e}")

        # Example 2: Generate documentation for a project
        try:
            project_results = await generate_project_docs(directory=".", output_dir="output_docs", format="html", use_cache=True, concurrency=4, create_index=True, clean_output_dir=True)
            print(json.dumps(project_results, indent=2))
        except Exception as e:
            print(f"Error: {e}")

    # Create example.py file
    with open("example.py", "w") as f:
        f.write("""
\"\"\"
This is an example module.
\"\"\"

class MyClass:
    \"\"\"
    This is an example class.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int
        Description of param2.
    \"\"\"
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def my_method(self, arg1: int, arg2: str) -> bool:
        \"\"\"
        This is an example method.

        :param arg1: An integer argument.
        :type arg1: int
        :param arg2: A string argument.
        :type arg2: str
        :return: True if successful, False otherwise.
        :rtype: bool
        \"\"\"
        return True

async def my_async_function(input_value: str) -> str:
    \"\"\"
    An example of async function.

    :param input_value: Input string value.
    :type input_value: str
    :returns: Processed output string.
    \"\"\"
    return f"Processed: {input_value}"
""")


    asyncio.run(main())