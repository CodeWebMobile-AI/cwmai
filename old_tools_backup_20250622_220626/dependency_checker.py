"""
Dependency Checker Tool
Validates project dependencies and checks for updates or conflicts.
"""

import subprocess
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from packaging import version
import asyncio
import aiohttp
from functools import lru_cache

# Constants for API rate limiting and caching
MAX_CONCURRENT_REQUESTS = 5
UPDATE_CHECK_CACHE_SIZE = 50
SECURITY_CHECK_CACHE_SIZE = 50
INSTALLED_PACKAGES_CACHE_SIZE = 1 #Only store 1 result. We want an updated list whenever needed.

# Custom Exception for Dependency Issues
class DependencyError(Exception):
    """Base class for dependency-related exceptions."""
    pass

class RequirementsFileError(DependencyError):
    """Exception raised when there's an issue with the requirements file."""
    pass

class PipCommandError(DependencyError):
    """Exception raised when a pip command fails."""
    def __init__(self, message, cmd, returncode, output):
        super().__init__(message)
        self.cmd = cmd
        self.returncode = returncode
        self.output = output


async def check_dependencies(requirements_file: str = "requirements.txt") -> Dict[str, Any]:
    """
    Check project dependencies for issues and updates.

    Args:
        requirements_file: Path to requirements file

    Returns:
        Dependency analysis report
    """
    report = {
        'file': requirements_file,
        'total_dependencies': 0,
        'issues': [],
        'updates_available': [],
        'security_warnings': [],
        'recommendations': [],
        'summary': {}  # Add a summary field
    }

    # Input Validation: Check if requirements_file is a string
    if not isinstance(requirements_file, str):
        report['issues'].append({
            'type': 'invalid_input',
            'message': f'Invalid input: requirements_file must be a string, but got {type(requirements_file)}'
        })
        report['summary']['status'] = 'failed'
        report['summary']['message'] = 'Invalid input provided.'
        return report

    # Check if file exists
    req_path = Path(requirements_file)
    if not req_path.exists():
        report['issues'].append({
            'type': 'missing_file',
            'message': f'{requirements_file} not found'
        })
        report['summary']['status'] = 'failed'
        report['summary']['message'] = 'Requirements file not found.'
        return report

    # Initialize counters for the summary
    not_installed_count = 0
    version_mismatch_count = 0
    updates_available_count = 0
    security_warnings_count = 0

    # Read requirements
    try:
        try:
            requirements = req_path.read_text().splitlines()
        except FileNotFoundError as e:
            raise RequirementsFileError(f"Requirements file not found: {e}") from e
        except IOError as e:
            raise RequirementsFileError(f"Error reading requirements file: {e}") from e

        dependencies = {}

        for line in requirements:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse dependency
                match = re.match(r'^([a-zA-Z0-9-_]+)([><=!~^]+)?(.*)$', line)  # More permissive version specifier
                if match:
                    pkg_name = match.group(1)
                    operator = match.group(2) or ''
                    version_spec = match.group(3) or ''
                    dependencies[pkg_name] = {
                        'spec': line,
                        'operator': operator,
                        'version': version_spec
                    }

        report['total_dependencies'] = len(dependencies)

        # Check installed versions
        try:
            installed = await list_installed_packages() # Use Async version

            # Check each dependency
            for pkg_name, dep_info in dependencies.items():
                pkg_lower = pkg_name.lower()

                if pkg_lower not in installed:
                    report['issues'].append({
                        'type': 'not_installed',
                        'package': pkg_name,
                        'message': f'{pkg_name} is not installed'
                    })
                    not_installed_count += 1
                else:
                    # Check version compatibility
                    installed_ver = installed[pkg_lower]
                    if dep_info['version']:
                        if not _check_version_compatibility(
                            installed_ver, dep_info['operator'], dep_info['version']
                        ):
                            report['issues'].append({
                                'type': 'version_mismatch',
                                'package': pkg_name,
                                'installed': installed_ver,
                                'required': dep_info['spec']
                            })
                            version_mismatch_count += 1
        except PipCommandError as e: #Specific handling for PipCommandError
            report['issues'].append({
                'type': 'pip_error',
                'message': f'Error running pip list: {str(e)}',
                'command': e.cmd,
                'returncode': e.returncode,
                'output': e.output
            })
            report['summary']['status'] = 'warning'
            report['summary']['message'] = 'An error occurred while running pip list. Dependency checks may be incomplete.'
        except Exception as e:
            report['issues'].append({
                'type': 'error',
                'message': f'Error getting installed packages: {str(e)}'
            })
            report['summary']['status'] = 'error'
            report['summary']['message'] = 'An unexpected error occurred while listing installed packages.'

        # Check for updates (limited to prevent API rate limits)
        # Use asyncio.gather to run update checks concurrently, respecting MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def limited_check_update(pkg_name):
            async with semaphore:
                try:
                    update_info = await _check_for_update(pkg_name)
                    if update_info:
                        report['updates_available'].append(update_info)
                        nonlocal updates_available_count
                        updates_available_count += 1
                except Exception as e:
                    print(f"Error checking update for {pkg_name}: {e}")
                    report['issues'].append({
                        'type': 'update_check_error',
                        'package': pkg_name,
                        'message': f"Error checking update: {e}"
                    })


        update_tasks = [limited_check_update(pkg_name) for pkg_name in dependencies.keys()]
        await asyncio.gather(*update_tasks, return_exceptions=True) # Capture exceptions in gather

        # Check for security vulnerabilities
        try:
            security_vulnerabilities = await check_security_vulnerabilities(list(dependencies.keys()))
            if security_vulnerabilities:
                report['security_warnings'].extend(security_vulnerabilities)
                security_warnings_count = len(security_vulnerabilities)
        except Exception as e:
            print(f"Error checking security vulnerabilities: {e}")
            report['issues'].append({
                'type': 'security_check_error',
                'message': f"Error checking security vulnerabilities: {e}"
            })

        # Add recommendations
        if report['issues']:
            report['recommendations'].append({
                'type': 'fix_issues',
                'message': 'Run: pip install -r requirements.txt --upgrade'
            })

        if len(report['updates_available']) > 0:  # Adjust threshold
            report['recommendations'].append({
                'type': 'update_dependencies',
                'message': 'Consider updating dependencies to latest stable versions'
            })

    except RequirementsFileError as e: # More Specific error handling
        report['issues'].append({
            'type': 'file_error',
            'message': str(e)
        })
        report['summary']['status'] = 'failed'
        report['summary']['message'] = str(e)

    except Exception as e:
        report['issues'].append({
            'type': 'error',
            'message': str(e)
        })
        report['summary']['status'] = 'error'
        report['summary']['message'] = f'An unexpected error occurred: {str(e)}'
    finally:
        # Construct summary
        report['summary']['total_dependencies'] = report['total_dependencies']
        report['summary']['not_installed'] = not_installed_count
        report['summary']['version_mismatches'] = version_mismatch_count
        report['summary']['updates_available'] = updates_available_count
        report['summary']['security_warnings'] = security_warnings_count
        if 'status' not in report['summary']:
            report['summary']['status'] = 'success'
            report['summary']['message'] = 'All dependencies checked.'

    return report


def _check_version_compatibility(installed: str, operator: str, required: str) -> bool:
    """Check if installed version meets requirement."""
    try:
        installed_v = version.parse(installed)
        required_v = version.parse(required)

        if operator == '==':
            return installed_v == required_v
        elif operator == '>=':
            return installed_v >= required_v
        elif operator == '>':
            return installed_v > required_v
        elif operator == '<=':
            return installed_v <= required_v
        elif operator == '<':
            return installed_v < required_v
        elif operator == '!=':
            return installed_v != required_v
        elif operator == '~=': # Compatible release operator
            return installed_v >= required_v and str(installed_v).split('.')[0] == str(required_v).split('.')[0] and str(installed_v).split('.')[1] == str(required_v).split('.')[1]
        elif operator == '^=': # Compatible release operator
            return installed_v >= required_v and str(installed_v).split('.')[0] == str(required_v).split('.')[0]
        else:
            print(f"Warning: Unsupported operator '{operator}'. Assuming compatibility.")
            return True
    except version.InvalidVersion as e:
        print(f"Warning: Invalid version format encountered: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error during version comparison: {e}")
        return True


@lru_cache(maxsize=UPDATE_CHECK_CACHE_SIZE)
async def _check_for_update(package_name: str) -> Optional[Dict[str, str]]:
    """Check if package has updates available."""
    try:
        result = subprocess.run(
            ['pip', 'index', 'versions', package_name],
            capture_output=True, text=True, timeout=10 #Increased timeout
        )

        if result.returncode == 0:
            output = result.stdout
            # Parse available versions
            if 'Available versions:' in output:
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if 'Available versions:' in line and i + 1 < len(lines):
                        versions = lines[i + 1].strip().split(', ')
                        if versions:
                            latest = versions[0]
                            return {
                                'package': package_name,
                                'latest_version': latest,
                                'available_versions': versions[:5]
                            }
        else:
             print(f"Error checking for updates for {package_name}: {result.stderr}") #Capture stderr

    except subprocess.TimeoutExpired:
        print(f"Timeout checking for updates for {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error checking for updates for {package_name}: {e}")
    except Exception as e:
        print(f"Unexpected error checking for updates for {package_name}: {e}")
    return None

@lru_cache(maxsize=SECURITY_CHECK_CACHE_SIZE)
async def _check_security_vulnerability(package_name: str) -> Optional[List[Dict[str, str]]]:
    """Check for security vulnerabilities using an external API (e.g., PyPI's Warehouse API)."""
    # Replace with actual API endpoint and authentication if necessary
    api_url = f"https://pypi.org/pypi/{package_name}/json"  # Example, may require modification
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=10) as response: #Added Timeout
                if response.status == 200:
                    data = await response.json()
                    vulnerabilities = []
                    if 'vulnerabilities' in data.get('info', {}): # Safer access
                        for vulnerability in data['info']['vulnerabilities']:
                            vulnerabilities.append({
                                'package': package_name,
                                'details': vulnerability['details'],
                                'link': vulnerability['link'],
                                'id': vulnerability['id']
                            })
                    return vulnerabilities
                elif response.status == 404:
                    print(f"Package {package_name} not found in PyPI.")
                    return None
                else:
                    print(f"Error checking security vulnerabilities for {package_name}: {response.status}")
                    return None
    except aiohttp.ClientError as e:
        print(f"aiohttp exception for {package_name}: {e}")
        return None
    except asyncio.TimeoutError:
        print(f"Timeout during security vulnerability check for {package_name}")
        return None
    except Exception as e:
        print(f"Unexpected error checking security vulnerabilities for {package_name}: {e}")
        return None

async def check_security_vulnerabilities(package_names: List[str]) -> List[Dict[str, str]]:
    """Check multiple packages for security vulnerabilities concurrently."""
    vulnerabilities = []
    tasks = [asyncio.create_task(_check_security_vulnerability(name)) for name in package_names]
    results = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions

    for result in results:
        if isinstance(result, Exception):  # Handle exceptions from tasks
            print(f"Error checking security vulnerabilities: {result}")
        elif result:
            vulnerabilities.extend(result)
    return vulnerabilities


async def fix_dependency_issues(report: Dict[str, Any], auto_fix: bool = False) -> Dict[str, Any]:
    """
    Fix dependency issues found in the report.

    Args:
        report: Dependency check report
        auto_fix: Whether to automatically fix issues

    Returns:
        Fix results
    """
    results = {
        'fixed': [],
        'failed': [],
        'commands': [],
        'skipped': [] # List of packages that were skipped due to errors
    }

    if not report.get('issues'):
        return results

    # Generate fix commands
    not_installed = [i['package'] for i in report['issues']
                    if i['type'] == 'not_installed']

    if not_installed:
        cmd = f"pip install {' '.join(not_installed)}"
        results['commands'].append(cmd)

        if auto_fix:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    results['fixed'].extend(not_installed)
                else:
                    results['failed'].extend(not_installed)
                    print(f"Error installing {not_installed}: {result.stderr}")
            except subprocess.CalledProcessError as e:
                results['failed'].append(f"Error: {str(e)}")
                print(f"Error installing {not_installed}: {e.stderr}")
            except Exception as e:
                results['failed'].append(f"Error: {str(e)}")
                print(f"Unexpected error during installation: {e}")

    # Version mismatches
    mismatches = [i for i in report['issues'] if i['type'] == 'version_mismatch']
    if mismatches:
        cmd = "pip install -r requirements.txt --upgrade"
        results['commands'].append(cmd)

        if auto_fix:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    results['fixed'].extend([m['package'] for m in mismatches])  # Mark as fixed
                else:
                    results['failed'].extend([m['package'] for m in mismatches])
                    print(f"Error upgrading dependencies: {result.stderr}")
            except subprocess.CalledProcessError as e:
                results['failed'].extend([m['package'] for m in mismatches])
                print(f"Error upgrading dependencies: {e.stderr}")
            except Exception as e:
                results['failed'].extend([m['package'] for m in mismatches])
                print(f"Unexpected error during upgrade: {e}")

    # Handle update recommendations (install latest versions)
    updates_available = [u['package'] for u in report['updates_available']]
    if updates_available:
        cmd = f"pip install --upgrade {' '.join(updates_available)}"
        results['commands'].append(cmd)

        if auto_fix:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    results['fixed'].extend(updates_available)
                else:
                    results['failed'].extend(updates_available)
                    print(f"Error updating {updates_available}: {result.stderr}")
            except subprocess.CalledProcessError as e:
                results['failed'].extend(updates_available)
                print(f"Error updating {updates_available}: {e.stderr}")
            except Exception as e:
                results['failed'].extend(updates_available)
                print(f"Unexpected error during update: {e}")

    return results


# Utility Function: List installed packages
@lru_cache(maxsize=INSTALLED_PACKAGES_CACHE_SIZE)
async def list_installed_packages() -> Dict[str, str]:
    """Lists installed packages and their versions."""
    try:
        result = subprocess.run(['pip', 'list', '--format=json'], capture_output=True, text=True, check=True)
        if result.returncode == 0:
            installed = {pkg['name'].lower(): pkg['version'] for pkg in json.loads(result.stdout)}
            return installed
        else:
            raise PipCommandError(f"Error listing packages: {result.stderr}", cmd=['pip', 'list', '--format=json'], returncode=result.returncode, output=result.stderr)
    except subprocess.CalledProcessError as e:
        raise PipCommandError(f"Error listing packages: {e.stderr}", cmd=e.cmd, returncode=e.returncode, output=e.stderr) from e
    except Exception as e:
        print(f"Unexpected error listing packages: {e}")
        return {}


# Utility function: Check if a package is installed
async def is_package_installed(package_name: str) -> bool:
    """
    Checks if a package is installed.

    Args:
        package_name: The name of the package to check.

    Returns:
        True if the package is installed, False otherwise.
    """
    installed_packages = await list_installed_packages()
    return package_name.lower() in installed_packages


# Utility function: Get package version
async def get_package_version(package_name: str) -> Optional[str]:
    """
    Gets the version of an installed package.

    Args:
        package_name: The name of the package to check.

    Returns:
        The version of the package if installed, None otherwise.
    """
    installed_packages = await list_installed_packages()
    return installed_packages.get(package_name.lower())


# Enhanced Output: Generate Markdown Report
def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generates a Markdown report from the dependency check report."""
    markdown = f"# Dependency Check Report - {report['file']}\n\n"

    markdown += f"**Status:** {report['summary']['status'].capitalize()}\n"
    markdown += f"**Message:** {report['summary']['message']}\n\n"

    markdown += f"## Summary\n\n"
    markdown += f"- Total Dependencies: {report['summary']['total_dependencies']}\n"
    markdown += f"- Not Installed: {report['summary']['not_installed']}\n"
    markdown += f"- Version Mismatches: {report['summary']['version_mismatches']}\n"
    markdown += f"- Updates Available: {report['summary']['updates_available']}\n"
    markdown += f"- Security Warnings: {report['summary']['security_warnings']}\n\n"

    if report['issues']:
        markdown += "## Issues\n\n"
        for issue in report['issues']:
            markdown += f"- **Type:** {issue['type']}\n"
            if 'package' in issue:
                markdown += f"  **Package:** {issue['package']}\n"
            markdown += f"  **Message:** {issue['message']}\n\n"

    if report['updates_available']:
        markdown += "## Updates Available\n\n"
        for update in report['updates_available']:
            markdown += f"- **Package:** {update['package']}\n"
            markdown += f"  **Latest Version:** {update['latest_version']}\n"
            markdown += f"  **Available Versions:** {', '.join(update['available_versions'])}\n\n"

    if report['security_warnings']:
        markdown += "## Security Warnings\n\n"
        for warning in report['security_warnings']:
            markdown += f"- **Package:** {warning['package']}\n"
            markdown += f"  **Details:** {warning['details']}\n"
            markdown += f"  **Link:** {warning['link']}\n"
            markdown += f"  **ID:** {warning['id']}\n\n"

    if report['recommendations']:
        markdown += "## Recommendations\n\n"
        for recommendation in report['recommendations']:
            markdown += f"- **Type:** {recommendation['type']}\n"
            markdown += f"  **Message:** {recommendation['message']}\n\n"

    return markdown


# Export the functions
__all__ = ['check_dependencies', 'fix_dependency_issues', 'list_installed_packages', 'is_package_installed', 'get_package_version', 'generate_markdown_report']

# Example Usage (requires asyncio)
async def main():
    report = await check_dependencies()
    print(json.dumps(report, indent=2))
    #Example with specific requirements file
    #report = await check_dependencies(requirements_file='my_requirements.txt')
    #print(json.dumps(report, indent=2))
    # Uncomment to run fix (CAREFUL!)
    #fix_results = await fix_dependency_issues(report, auto_fix=True)
    #print(json.dumps(fix_results, indent=2))

    #Generate Markdown Report
    markdown_report = generate_markdown_report(report)
    print("\n--- Markdown Report ---\n")
    print(markdown_report)

if __name__ == "__main__":
    asyncio.run(main())