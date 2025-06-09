#!/usr/bin/env python3
"""
Dependency Security Scanner for CWMAI

Scans dependencies for known security vulnerabilities and provides 
security recommendations for Python packages.
"""

import os
import re
import json
import requests
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class VulnerabilityInfo:
    """Information about a security vulnerability."""
    package: str
    version: str
    vulnerability_id: str
    severity: str
    description: str
    fixed_versions: List[str]
    cve_id: Optional[str] = None
    reference_url: Optional[str] = None


@dataclass
class PackageInfo:
    """Information about an installed package."""
    name: str
    version: str
    latest_version: Optional[str] = None
    is_outdated: bool = False
    vulnerabilities: List[VulnerabilityInfo] = None
    
    def __post_init__(self):
        if self.vulnerabilities is None:
            self.vulnerabilities = []


class DependencySecurityScanner:
    """Scans Python dependencies for security vulnerabilities."""
    
    def __init__(self):
        """Initialize the dependency scanner."""
        self.known_vulnerabilities = {
            # Known vulnerable package patterns
            'requests': {
                '2.30.0': ['CVE-2023-32681'],
                '2.29.0': ['CVE-2023-32681'],
                '2.28.0': ['CVE-2022-24736'],
            },
            'urllib3': {
                '1.26.0': ['CVE-2023-45803'],
                '1.25.0': ['CVE-2020-26137'],
            },
            'certifi': {
                '2022.12.7': ['CVE-2023-37920'],
            },
            'pillow': {
                '9.5.0': ['CVE-2023-32682'],
                '9.4.0': ['CVE-2023-32681'],
            }
        }
        
        self.security_recommendations = {
            'requests': '>=2.32.0',
            'urllib3': '>=2.0.0',
            'certifi': '>=2023.7.22',
            'pillow': '>=10.0.0',
            'jinja2': '>=3.1.0',
            'werkzeug': '>=2.3.0',
            'flask': '>=2.3.0',
            'django': '>=4.2.0',
            'pyyaml': '>=6.0.0',
            'cryptography': '>=41.0.0'
        }
    
    def scan_requirements_file(self, requirements_path: str = "requirements.txt") -> List[PackageInfo]:
        """Scan requirements.txt file for security issues.
        
        Args:
            requirements_path: Path to requirements.txt file
        
        Returns:
            List of PackageInfo objects with vulnerability information
        """
        packages = []
        
        if not os.path.exists(requirements_path):
            print(f"Warning: {requirements_path} not found")
            return packages
        
        with open(requirements_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                package_info = self._parse_requirement_line(line)
                if package_info:
                    # Check for vulnerabilities
                    package_info.vulnerabilities = self._check_vulnerabilities(
                        package_info.name, package_info.version
                    )
                    
                    # Check if package is outdated
                    package_info.latest_version = self._get_latest_version(package_info.name)
                    package_info.is_outdated = self._is_outdated(
                        package_info.version, package_info.latest_version
                    )
                    
                    packages.append(package_info)
        
        return packages
    
    def scan_installed_packages(self) -> List[PackageInfo]:
        """Scan currently installed packages for security issues.
        
        Returns:
            List of PackageInfo objects with vulnerability information
        """
        packages = []
        
        try:
            # Get installed packages using pip
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            installed_packages = json.loads(result.stdout)
            
            for pkg_data in installed_packages:
                package_info = PackageInfo(
                    name=pkg_data['name'].lower(),
                    version=pkg_data['version']
                )
                
                # Check for vulnerabilities
                package_info.vulnerabilities = self._check_vulnerabilities(
                    package_info.name, package_info.version
                )
                
                # Check if package is outdated
                package_info.latest_version = self._get_latest_version(package_info.name)
                package_info.is_outdated = self._is_outdated(
                    package_info.version, package_info.latest_version
                )
                
                packages.append(package_info)
        
        except subprocess.CalledProcessError as e:
            print(f"Error scanning installed packages: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing pip output: {e}")
        
        return packages
    
    def _parse_requirement_line(self, line: str) -> Optional[PackageInfo]:
        """Parse a line from requirements.txt.
        
        Args:
            line: Line from requirements.txt
        
        Returns:
            PackageInfo object or None if line cannot be parsed
        """
        # Remove comments
        line = line.split('#')[0].strip()
        
        if not line:
            return None
        
        # Match package==version pattern
        match = re.match(r'^([a-zA-Z0-9\-_\.]+)==([0-9\.\-\w]+)', line)
        if match:
            return PackageInfo(
                name=match.group(1).lower(),
                version=match.group(2)
            )
        
        # Match package>=version pattern  
        match = re.match(r'^([a-zA-Z0-9\-_\.]+)>=([0-9\.\-\w]+)', line)
        if match:
            return PackageInfo(
                name=match.group(1).lower(),
                version=match.group(2)
            )
        
        # Match package name only
        match = re.match(r'^([a-zA-Z0-9\-_\.]+)', line)
        if match:
            return PackageInfo(
                name=match.group(1).lower(),
                version="unknown"
            )
        
        return None
    
    def _check_vulnerabilities(self, package_name: str, version: str) -> List[VulnerabilityInfo]:
        """Check if a package version has known vulnerabilities.
        
        Args:
            package_name: Name of the package
            version: Version of the package
        
        Returns:
            List of VulnerabilityInfo objects
        """
        vulnerabilities = []
        
        if package_name in self.known_vulnerabilities:
            pkg_vulns = self.known_vulnerabilities[package_name]
            
            for vuln_version, cve_list in pkg_vulns.items():
                if self._version_matches_or_older(version, vuln_version):
                    for cve_id in cve_list:
                        vulnerability = VulnerabilityInfo(
                            package=package_name,
                            version=version,
                            vulnerability_id=cve_id,
                            severity="high" if "2023" in cve_id else "medium",
                            description=f"Known vulnerability in {package_name} {version}",
                            fixed_versions=[self.security_recommendations.get(package_name, "latest")],
                            cve_id=cve_id,
                            reference_url=f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id}"
                        )
                        vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _get_latest_version(self, package_name: str) -> Optional[str]:
        """Get the latest version of a package from PyPI.
        
        Args:
            package_name: Name of the package
        
        Returns:
            Latest version string or None if not found
        """
        try:
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/json",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['info']['version']
        
        except Exception:
            # Fail silently for network issues
            pass
        
        return None
    
    def _is_outdated(self, current_version: str, latest_version: Optional[str]) -> bool:
        """Check if current version is outdated.
        
        Args:
            current_version: Current version string
            latest_version: Latest version string
        
        Returns:
            True if current version is outdated
        """
        if not latest_version or current_version == "unknown":
            return False
        
        try:
            current_parts = [int(x) for x in current_version.split('.')]
            latest_parts = [int(x) for x in latest_version.split('.')]
            
            # Pad shorter version with zeros
            max_length = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_length - len(current_parts)))
            latest_parts.extend([0] * (max_length - len(latest_parts)))
            
            return current_parts < latest_parts
        
        except ValueError:
            # Handle non-numeric version parts
            return current_version != latest_version
    
    def _version_matches_or_older(self, version: str, target_version: str) -> bool:
        """Check if version is the same or older than target version.
        
        Args:
            version: Version to check
            target_version: Target version to compare against
        
        Returns:
            True if version is same or older than target_version
        """
        if version == "unknown":
            return True  # Assume vulnerability for unknown versions
        
        try:
            version_parts = [int(x) for x in version.split('.')]
            target_parts = [int(x) for x in target_version.split('.')]
            
            # Pad shorter version with zeros
            max_length = max(len(version_parts), len(target_parts))
            version_parts.extend([0] * (max_length - len(version_parts)))
            target_parts.extend([0] * (max_length - len(target_parts)))
            
            return version_parts <= target_parts
        
        except ValueError:
            # Handle non-numeric version parts
            return version == target_version
    
    def generate_security_report(self, packages: List[PackageInfo]) -> Dict[str, Any]:
        """Generate a comprehensive security report.
        
        Args:
            packages: List of PackageInfo objects
        
        Returns:
            Security report dictionary
        """
        total_packages = len(packages)
        vulnerable_packages = [pkg for pkg in packages if pkg.vulnerabilities]
        outdated_packages = [pkg for pkg in packages if pkg.is_outdated]
        
        # Count vulnerabilities by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for pkg in vulnerable_packages:
            for vuln in pkg.vulnerabilities:
                severity = vuln.severity.lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
        
        # Generate recommendations
        recommendations = []
        
        for pkg in vulnerable_packages:
            for vuln in pkg.vulnerabilities:
                if vuln.fixed_versions:
                    recommendations.append(
                        f"Update {pkg.name} from {pkg.version} to {vuln.fixed_versions[0]} (fixes {vuln.vulnerability_id})"
                    )
        
        for pkg in outdated_packages:
            if not pkg.vulnerabilities and pkg.name in self.security_recommendations:
                recommendations.append(
                    f"Update {pkg.name} from {pkg.version} to {self.security_recommendations[pkg.name]} (security best practice)"
                )
        
        # Calculate security score
        security_score = self._calculate_security_score(packages)
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_packages': total_packages,
                'vulnerable_packages': len(vulnerable_packages),
                'outdated_packages': len(outdated_packages),
                'security_score': security_score
            },
            'vulnerabilities': {
                'by_severity': severity_counts,
                'total': sum(severity_counts.values()),
                'details': [
                    {
                        'package': pkg.name,
                        'version': pkg.version,
                        'vulnerabilities': [
                            {
                                'id': vuln.vulnerability_id,
                                'severity': vuln.severity,
                                'description': vuln.description,
                                'fixed_versions': vuln.fixed_versions,
                                'cve_id': vuln.cve_id,
                                'reference_url': vuln.reference_url
                            }
                            for vuln in pkg.vulnerabilities
                        ]
                    }
                    for pkg in vulnerable_packages
                ]
            },
            'outdated_packages': [
                {
                    'package': pkg.name,
                    'current_version': pkg.version,
                    'latest_version': pkg.latest_version
                }
                for pkg in outdated_packages
            ],
            'recommendations': recommendations
        }
    
    def _calculate_security_score(self, packages: List[PackageInfo]) -> int:
        """Calculate overall dependency security score (0-100).
        
        Args:
            packages: List of PackageInfo objects
        
        Returns:
            Security score (0-100, higher is better)
        """
        if not packages:
            return 100
        
        # Start with perfect score
        score = 100
        
        # Penalty for vulnerabilities
        vulnerability_penalties = {'critical': 20, 'high': 10, 'medium': 5, 'low': 2}
        
        for pkg in packages:
            for vuln in pkg.vulnerabilities:
                severity = vuln.severity.lower()
                penalty = vulnerability_penalties.get(severity, 2)
                score -= penalty
        
        # Small penalty for outdated packages
        outdated_count = sum(1 for pkg in packages if pkg.is_outdated and not pkg.vulnerabilities)
        score -= outdated_count * 1
        
        return max(0, score)


def scan_dependencies(requirements_path: str = "requirements.txt", check_installed: bool = False) -> None:
    """Scan dependencies and print security report.
    
    Args:
        requirements_path: Path to requirements.txt file
        check_installed: Whether to also check installed packages
    """
    print("🔍 CWMAI Dependency Security Scanner")
    print("=" * 50)
    
    scanner = DependencySecurityScanner()
    
    # Scan requirements.txt
    packages = scanner.scan_requirements_file(requirements_path)
    
    if check_installed:
        print("Scanning installed packages...")
        installed_packages = scanner.scan_installed_packages()
        # Merge with requirements (avoid duplicates)
        package_names = {pkg.name for pkg in packages}
        for pkg in installed_packages:
            if pkg.name not in package_names:
                packages.append(pkg)
    
    if not packages:
        print("No packages found to scan.")
        return
    
    # Generate report
    report = scanner.generate_security_report(packages)
    
    # Print summary
    summary = report['summary']
    print(f"\n📋 Security Summary")
    print(f"Total packages: {summary['total_packages']}")
    print(f"Vulnerable packages: {summary['vulnerable_packages']}")
    print(f"Outdated packages: {summary['outdated_packages']}")
    print(f"Security score: {summary['security_score']}/100")
    
    # Print vulnerabilities
    if report['vulnerabilities']['total'] > 0:
        print(f"\n🚨 Security Vulnerabilities ({report['vulnerabilities']['total']} found)")
        print("-" * 30)
        
        for vuln_pkg in report['vulnerabilities']['details']:
            print(f"\n📦 {vuln_pkg['package']} (v{vuln_pkg['version']})")
            for vuln in vuln_pkg['vulnerabilities']:
                severity_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                emoji = severity_emoji.get(vuln['severity'].lower(), '⚪')
                print(f"   {emoji} {vuln['id']} ({vuln['severity']})")
                print(f"      {vuln['description']}")
                if vuln['fixed_versions']:
                    print(f"      Fix: Update to {', '.join(vuln['fixed_versions'])}")
                if vuln['reference_url']:
                    print(f"      Reference: {vuln['reference_url']}")
    
    # Print recommendations
    if report['recommendations']:
        print(f"\n💡 Security Recommendations")
        print("-" * 30)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print(f"\n✅ Dependency security scan complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan Python dependencies for security vulnerabilities")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements.txt file")
    parser.add_argument("--installed", action="store_true", help="Also scan installed packages")
    
    args = parser.parse_args()
    
    scan_dependencies(args.requirements, args.installed)