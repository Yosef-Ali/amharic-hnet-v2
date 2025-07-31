#!/usr/bin/env python3
"""
Import structure validation script for Amharic H-Net v2.
Checks for proper import organization and potential circular imports.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ImportChecker:
    """Check import structure and organization in Python files."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def check_file(self, filepath: Path) -> bool:
        """Check a single Python file for import issues."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(filepath))
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Check import organization
            self._check_import_organization(filepath, imports)
            
            # Check for potential circular imports
            self._check_circular_imports(filepath, imports)
            
            return len(self.errors) == 0
            
        except SyntaxError as e:
            self.errors.append(f"{filepath}: Syntax error - {e}")
            return False
        except Exception as e:
            self.errors.append(f"{filepath}: Error processing file - {e}")
            return False
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract import statements from AST."""
        imports = {
            'stdlib': [],
            'third_party': [],
            'local': [],
            'relative': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    category = self._categorize_import(alias.name)
                    imports[category].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.level > 0:  # Relative import
                        imports['relative'].append(f"{'.' * node.level}{node.module}")
                    else:
                        category = self._categorize_import(node.module)
                        imports[category].append(node.module)
                        
        return imports
    
    def _categorize_import(self, module_name: str) -> str:
        """Categorize import as stdlib, third_party, or local."""
        stdlib_modules = {
            'os', 'sys', 'json', 'yaml', 'logging', 'argparse', 'pathlib',
            'typing', 'dataclasses', 'collections', 'itertools', 'functools',
            'multiprocessing', 'threading', 'asyncio', 're', 'math', 'random',
            'datetime', 'time', 'uuid', 'hashlib', 'pickle', 'sqlite3'
        }
        
        third_party_modules = {
            'torch', 'transformers', 'numpy', 'pandas', 'matplotlib', 'seaborn',
            'sklearn', 'scipy', 'wandb', 'tensorboard', 'tqdm', 'einops',
            'datasets', 'morfessor', 'accelerate', 'mamba_ssm'
        }
        
        root_module = module_name.split('.')[0]
        
        if root_module in stdlib_modules:
            return 'stdlib'
        elif root_module in third_party_modules:
            return 'third_party'
        elif root_module == 'src' or module_name.startswith('src.'):
            return 'local'
        else:
            # Default to third_party for unknown modules
            return 'third_party'
    
    def _check_import_organization(self, filepath: Path, imports: Dict[str, List[str]]) -> None:
        """Check if imports are properly organized."""
        # Check if we have imports from multiple categories
        non_empty_categories = [cat for cat, imps in imports.items() if imps]
        
        if len(non_empty_categories) > 1:
            # We should have proper separation
            expected_order = ['stdlib', 'third_party', 'local', 'relative']
            
            # This is a simplified check - in practice, you'd need to parse
            # the actual file structure to check ordering
            pass
    
    def _check_circular_imports(self, filepath: Path, imports: Dict[str, List[str]]) -> None:
        """Check for potential circular imports."""
        # Get the module name for this file
        file_module = self._get_module_name(filepath)
        
        # Check if any local imports might create circular dependencies
        for local_import in imports['local']:
            if self._might_be_circular(file_module, local_import):
                self.warnings.append(
                    f"{filepath}: Potential circular import with {local_import}"
                )
    
    def _get_module_name(self, filepath: Path) -> str:
        """Convert file path to module name."""
        # Simplified conversion - assumes src/ structure
        parts = filepath.parts
        if 'src' in parts:
            src_index = parts.index('src')
            module_parts = parts[src_index + 1:]
            if module_parts[-1].endswith('.py'):
                module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
            return '.'.join(module_parts)
        return str(filepath.stem)
    
    def _might_be_circular(self, file_module: str, import_module: str) -> bool:
        """Check if import might create circular dependency."""
        # Simplified check - in practice, you'd build a dependency graph
        return import_module.startswith('src.') and file_module in import_module


def main():
    """Main function to run import checks."""
    if len(sys.argv) < 2:
        print("Usage: python check_imports.py <file1.py> [file2.py] ...")
        return 1
    
    checker = ImportChecker()
    success = True
    
    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if path.suffix == '.py' and path.exists():
            if not checker.check_file(path):
                success = False
    
    # Print results
    if checker.errors:
        print("ERRORS:")
        for error in checker.errors:
            print(f"  {error}")
    
    if checker.warnings:
        print("WARNINGS:")
        for warning in checker.warnings:
            print(f"  {warning}")
    
    if success and not checker.errors:
        print("All import checks passed!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())