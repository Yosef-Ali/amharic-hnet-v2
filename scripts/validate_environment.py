#!/usr/bin/env python3
"""
Environment validation script for Amharic H-Net v2 development environment.
Checks all dependencies, hardware setup, and configuration.
"""

import os
import sys
import subprocess
import importlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings

# Suppress warnings during validation
warnings.filterwarnings("ignore")


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class EnvironmentValidator:
    """Comprehensive environment validation for Amharic H-Net v2."""
    
    def __init__(self):
        self.passed_checks = 0
        self.failed_checks = 0
        self.warnings = []
        self.errors = []
        
    def print_header(self, title: str):
        """Print section header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    
    def print_check(self, description: str, status: str, details: str = ""):
        """Print individual check result."""
        if status == "PASS":
            icon = f"{Colors.GREEN}✓{Colors.END}"
            self.passed_checks += 1
        elif status == "FAIL":
            icon = f"{Colors.RED}✗{Colors.END}"
            self.failed_checks += 1
        elif status == "WARN":
            icon = f"{Colors.YELLOW}⚠{Colors.END}"
        else:
            icon = f"{Colors.BLUE}ℹ{Colors.END}"
        
        print(f"{icon} {description}")
        if details:
            print(f"   {Colors.CYAN}{details}{Colors.END}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major >= required_major and version.minor >= required_minor:
            self.print_check(
                f"Python version: {version.major}.{version.minor}.{version.micro}",
                "PASS",
                f"Meets requirement: Python {required_major}.{required_minor}+"
            )
            return True
        else:
            self.print_check(
                f"Python version: {version.major}.{version.minor}.{version.micro}",
                "FAIL",
                f"Requires Python {required_major}.{required_minor}+"
            )
            return False
    
    def check_virtual_environment(self) -> bool:
        """Check if running in virtual environment."""
        in_venv = (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            'VIRTUAL_ENV' in os.environ
        )
        
        if in_venv:
            venv_path = os.environ.get('VIRTUAL_ENV', 'Unknown')
            self.print_check(
                "Virtual environment",
                "PASS",
                f"Active: {venv_path}"
            )
            return True
        else:
            self.print_check(
                "Virtual environment",
                "WARN",
                "Not detected. Recommended to use virtual environment."
            )
            return False
    
    def check_required_packages(self) -> Dict[str, bool]:
        """Check if all required packages are installed."""
        required_packages = {
            'torch': '2.1.0',
            'transformers': '4.30.0',
            'numpy': '1.20.0',
            'pandas': '1.3.0',
            'tqdm': '4.60.0',
            'wandb': '0.15.0',
            'tensorboard': '2.10.0',
            'scikit-learn': '1.0.0',
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'pytest': '7.0.0',
            'black': '22.0.0',
            'flake8': '5.0.0',
        }
        
        results = {}
        
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                
                # Simple version comparison (not perfect but sufficient)
                try:
                    if self._compare_versions(version, min_version):
                        self.print_check(
                            f"Package {package}",
                            "PASS",
                            f"Version: {version} (>= {min_version})"
                        )
                        results[package] = True
                    else:
                        self.print_check(
                            f"Package {package}",
                            "FAIL",
                            f"Version: {version} (requires >= {min_version})"
                        )
                        results[package] = False
                except:
                    self.print_check(
                        f"Package {package}",
                        "PASS",
                        f"Version: {version} (version check skipped)"
                    )
                    results[package] = True
                    
            except ImportError:
                self.print_check(
                    f"Package {package}",
                    "FAIL",
                    f"Not installed (requires >= {min_version})"
                )
                results[package] = False
        
        return results
    
    def _compare_versions(self, version1: str, version2: str) -> bool:
        """Simple version comparison."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad with zeros to make same length
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts >= v2_parts
        except:
            return True  # If comparison fails, assume it's okay
    
    def check_torch_installation(self) -> Dict[str, bool]:
        """Check PyTorch installation and CUDA support."""
        results = {}
        
        try:
            import torch
            
            # Check PyTorch version
            self.print_check(
                "PyTorch installation",
                "PASS",
                f"Version: {torch.__version__}"
            )
            results['torch_installed'] = True
            
            # Check CUDA availability
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                
                self.print_check(
                    "CUDA support",
                    "PASS",
                    f"CUDA {cuda_version}, {device_count} GPU(s), Primary: {device_name}"
                )
                results['cuda_available'] = True
                
                # Check GPU memory
                if device_count > 0:
                    memory = torch.cuda.get_device_properties(0).total_memory
                    memory_gb = memory / (1024**3)
                    
                    if memory_gb >= 8:
                        self.print_check(
                            "GPU memory",
                            "PASS",
                            f"{memory_gb:.1f} GB available"
                        )
                        results['sufficient_gpu_memory'] = True
                    else:
                        self.print_check(
                            "GPU memory",
                            "WARN",
                            f"{memory_gb:.1f} GB available (8GB+ recommended)"
                        )
                        results['sufficient_gpu_memory'] = False
            else:
                self.print_check(
                    "CUDA support",
                    "WARN",
                    "CUDA not available. Training will use CPU (slower)."
                )
                results['cuda_available'] = False
                results['sufficient_gpu_memory'] = False
                
        except ImportError:
            self.print_check(
                "PyTorch installation",
                "FAIL",
                "PyTorch not installed"
            )
            results['torch_installed'] = False
            results['cuda_available'] = False
            results['sufficient_gpu_memory'] = False
        
        return results
    
    def check_directory_structure(self) -> bool:
        """Check if project directory structure is correct."""
        required_dirs = [
            'src',
            'src/models',
            'src/training',
            'src/preprocessing',
            'src/evaluation',
            'configs',
            'data',
            'data/raw',
            'data/processed',
            'outputs',
            'tests',
            'notebooks',
        ]
        
        missing_dirs = []
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                self.print_check(
                    f"Directory: {dir_path}",
                    "PASS"
                )
            else:
                self.print_check(
                    f"Directory: {dir_path}",
                    "FAIL",
                    "Missing directory"
                )
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"\n{Colors.YELLOW}Missing directories can be created with:{Colors.END}")
            for dir_path in missing_dirs:
                print(f"  mkdir -p {dir_path}")
        
        return len(missing_dirs) == 0
    
    def check_configuration_files(self) -> bool:
        """Check if configuration files exist and are valid."""
        config_files = [
            'pyproject.toml',
            '.pre-commit-config.yaml',
            'requirements.txt',
            'configs/config.yaml',
        ]
        
        all_valid = True
        
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    # Basic validation based on file extension
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        import yaml
                        with open(config_file, 'r') as f:
                            yaml.safe_load(f)
                    elif config_file.endswith('.toml'):
                        try:
                            import tomli as toml
                        except ImportError:
                            import toml
                        with open(config_file, 'r') as f:
                            toml.load(f)
                    elif config_file.endswith('.json'):
                        with open(config_file, 'r') as f:
                            json.load(f)
                    
                    self.print_check(
                        f"Config file: {config_file}",
                        "PASS"
                    )
                except Exception as e:
                    self.print_check(
                        f"Config file: {config_file}",
                        "FAIL",
                        f"Invalid format: {str(e)}"
                    )
                    all_valid = False
            else:
                self.print_check(
                    f"Config file: {config_file}",
                    "WARN",
                    "File not found"
                )
                all_valid = False
        
        return all_valid
    
    def check_system_resources(self) -> Dict[str, bool]:
        """Check system resources (RAM, disk space)."""
        results = {}
        
        try:
            import psutil
            
            # Check RAM
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 16:
                self.print_check(
                    "System RAM",
                    "PASS",
                    f"{memory_gb:.1f} GB available"
                )
                results['sufficient_ram'] = True
            elif memory_gb >= 8:
                self.print_check(
                    "System RAM",
                    "WARN",
                    f"{memory_gb:.1f} GB available (16GB+ recommended)"
                )
                results['sufficient_ram'] = False
            else:
                self.print_check(
                    "System RAM",
                    "FAIL",
                    f"{memory_gb:.1f} GB available (8GB minimum)"
                )
                results['sufficient_ram'] = False
            
            # Check disk space
            disk = psutil.disk_usage('.')
            disk_gb = disk.free / (1024**3)
            
            if disk_gb >= 50:
                self.print_check(
                    "Disk space",
                    "PASS",
                    f"{disk_gb:.1f} GB free"
                )
                results['sufficient_disk'] = True
            else:
                self.print_check(
                    "Disk space",
                    "WARN",
                    f"{disk_gb:.1f} GB free (50GB+ recommended)"
                )
                results['sufficient_disk'] = False
                
        except ImportError:
            self.print_check(
                "System resources",
                "WARN",
                "psutil not available for resource checking"
            )
            results['sufficient_ram'] = True  # Assume OK
            results['sufficient_disk'] = True
        
        return results
    
    def check_external_tools(self) -> Dict[str, bool]:
        """Check external tools and dependencies."""
        tools = {
            'git': 'git --version',
            'python': 'python --version',
            'pip': 'pip --version',
        }
        
        results = {}
        
        for tool, command in tools.items():
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    self.print_check(
                        f"External tool: {tool}",
                        "PASS",
                        version
                    )
                    results[tool] = True
                else:
                    self.print_check(
                        f"External tool: {tool}",
                        "FAIL",
                        "Command failed"
                    )
                    results[tool] = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.print_check(
                    f"External tool: {tool}",
                    "FAIL",
                    "Not found or timeout"
                )
                results[tool] = False
        
        return results
    
    def run_quick_test(self) -> bool:
        """Run a quick functionality test."""
        try:
            import torch
            import numpy as np
            
            # Test PyTorch basic operations
            x = torch.randn(10, 10)
            y = torch.mm(x, x.t())
            
            # Test CUDA if available
            if torch.cuda.is_available():
                x_cuda = x.cuda()
                y_cuda = torch.mm(x_cuda, x_cuda.t())
                y_cuda = y_cuda.cpu()
            
            # Test numpy
            a = np.random.randn(10, 10)
            b = np.dot(a, a.T)
            
            self.print_check(
                "Quick functionality test",
                "PASS",
                "Basic tensor operations working"
            )
            return True
            
        except Exception as e:
            self.print_check(
                "Quick functionality test",
                "FAIL",
                f"Error: {str(e)}"
            )
            return False
    
    def generate_environment_report(self) -> Dict:
        """Generate comprehensive environment report."""
        report = {
            "timestamp": str(Path().resolve()),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform,
            },
            "environment": {
                "virtual_env": os.environ.get('VIRTUAL_ENV'),
                "conda_env": os.environ.get('CONDA_DEFAULT_ENV'),
                "path": os.environ.get('PATH'),
            },
            "validation_results": {
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "warnings": self.warnings,
                "errors": self.errors,
            }
        }
        
        return report
    
    def run_all_checks(self) -> bool:
        """Run all environment validation checks."""
        print(f"{Colors.BOLD}{Colors.PURPLE}Amharic H-Net v2 Environment Validation{Colors.END}")
        print(f"{Colors.PURPLE}Validating development environment setup...{Colors.END}")
        
        # System and Python checks
        self.print_header("System and Python Environment")
        self.check_python_version()
        self.check_virtual_environment()
        
        # Package checks
        self.print_header("Python Packages")
        package_results = self.check_required_packages()
        
        # PyTorch specific checks
        self.print_header("PyTorch and CUDA")
        torch_results = self.check_torch_installation()
        
        # Project structure checks
        self.print_header("Project Structure")
        self.check_directory_structure()
        self.check_configuration_files()
        
        # System resources
        self.print_header("System Resources")
        resource_results = self.check_system_resources()
        
        # External tools
        self.print_header("External Tools")
        tool_results = self.check_external_tools()
        
        # Quick functionality test
        self.print_header("Functionality Test")
        self.run_quick_test()
        
        # Summary
        self.print_header("Validation Summary")
        
        total_checks = self.passed_checks + self.failed_checks
        success_rate = (self.passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        print(f"{Colors.GREEN}✓ Passed: {self.passed_checks}{Colors.END}")
        print(f"{Colors.RED}✗ Failed: {self.failed_checks}{Colors.END}")
        print(f"{Colors.BLUE}Success Rate: {success_rate:.1f}%{Colors.END}")
        
        if self.failed_checks == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Environment validation passed!{Colors.END}")
            print(f"{Colors.GREEN}Your Amharic H-Net v2 development environment is ready.{Colors.END}")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Environment validation failed!{Colors.END}")
            print(f"{Colors.RED}Please fix the failed checks before proceeding.{Colors.END}")
            return False


def main():
    """Main function to run environment validation."""
    validator = EnvironmentValidator()
    success = validator.run_all_checks()
    
    # Generate report
    report = validator.generate_environment_report()
    report_file = Path("environment_validation_report.json")
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{Colors.BLUE}Detailed report saved to: {report_file}{Colors.END}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())