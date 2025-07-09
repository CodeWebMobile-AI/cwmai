"""
Module Installer Helper

Automatically installs missing Python modules using arch -arm64 for ARM64 Macs.
"""

import subprocess
import sys
import logging
import os


class ModuleInstaller:
    """Handles automatic installation of missing Python modules."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.installed_modules = set()
        
    def try_install(self, module_name: str, pip_package: str = None) -> bool:
        """
        Try to install a missing module.
        
        Args:
            module_name: The name used in import statements
            pip_package: The pip package name (if different from module_name)
            
        Returns:
            True if installation succeeded, False otherwise
        """
        if module_name in self.installed_modules:
            return True
            
        pip_package = pip_package or module_name
        
        try:
            # Check if we're on ARM64 Mac
            is_arm64_mac = (
                sys.platform == "darwin" and 
                os.uname().machine == "arm64"
            )
            
            if is_arm64_mac:
                # Use arch -arm64 for ARM64 Macs
                cmd = [
                    "arch", "-arm64",
                    sys.executable, "-m", "pip", "install", pip_package
                ]
            else:
                # Regular pip install for other platforms
                cmd = [sys.executable, "-m", "pip", "install", pip_package]
                
            self.logger.info(f"Installing {pip_package} with command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.installed_modules.add(module_name)
            self.logger.info(f"Successfully installed {pip_package}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {pip_package}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error installing {pip_package}: {e}")
            return False
            
    def ensure_module(self, module_name: str, pip_package: str = None) -> bool:
        """
        Ensure a module is available, installing if necessary.
        
        Args:
            module_name: The name used in import statements
            pip_package: The pip package name (if different from module_name)
            
        Returns:
            True if module is available (already installed or successfully installed)
        """
        try:
            __import__(module_name)
            return True
        except ImportError:
            self.logger.info(f"Module {module_name} not found, attempting to install...")
            return self.try_install(module_name, pip_package)


# Global instance for convenience
module_installer = ModuleInstaller()


def ensure_modules(*modules):
    """
    Ensure multiple modules are available.
    
    Args:
        modules: List of tuples (module_name, pip_package) or just module names
        
    Example:
        ensure_modules(
            'dateparser',
            ('fuzzywuzzy', 'fuzzywuzzy[speedup]'),
            'rich'
        )
    """
    results = {}
    
    for module_info in modules:
        if isinstance(module_info, tuple):
            module_name, pip_package = module_info
        else:
            module_name = module_info
            pip_package = None
            
        results[module_name] = module_installer.ensure_module(module_name, pip_package)
        
    return results


# Common module mappings
COMMON_MODULES = {
    'dateparser': 'dateparser',
    'fuzzywuzzy': 'fuzzywuzzy[speedup]',
    'rich': 'rich',
    'redis': 'redis',
    'psutil': 'psutil',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'aiofiles': 'aiofiles',
    'httpx': 'httpx',
    'pytest': 'pytest',
    'black': 'black',
    'pylint': 'pylint',
    'mypy': 'mypy'
}


def auto_install_import(module_name: str):
    """
    Decorator that automatically installs a module if import fails.
    
    Example:
        @auto_install_import('dateparser')
        def parse_date(text):
            import dateparser
            return dateparser.parse(text)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to ensure module is available
            pip_package = COMMON_MODULES.get(module_name, module_name)
            if not module_installer.ensure_module(module_name, pip_package):
                raise ImportError(f"Failed to install required module: {module_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the installer
    logging.basicConfig(level=logging.INFO)
    
    # Test installing common modules
    print("Testing module installer...")
    
    results = ensure_modules(
        'dateparser',
        ('fuzzywuzzy', 'fuzzywuzzy[speedup]'),
        'rich'
    )
    
    print("\nInstallation results:")
    for module, success in results.items():
        print(f"  {module}: {'✓' if success else '✗'}")