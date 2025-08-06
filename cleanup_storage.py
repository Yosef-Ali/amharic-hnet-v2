#!/usr/bin/env python3
"""
Storage Cleanup Script
Remove unnecessary files while keeping essential trained models and code
"""

import os
import shutil
from pathlib import Path

def cleanup_storage():
    """Clean up storage by removing large unnecessary files."""
    print("🧹 CLEANING UP LOCAL STORAGE")
    print("=" * 50)
    
    current_dir = Path("/Users/mekdesyared/amharic-hnet-v3/amharic-hnet-v2")
    freed_space = 0
    
    # 1. Remove virtual environment (largest space consumer)
    venv_dir = current_dir / "venv"
    if venv_dir.exists():
        size = get_dir_size(venv_dir)
        print(f"🗑️  Removing virtual environment: {size/1024/1024:.1f} MB")
        shutil.rmtree(venv_dir)
        freed_space += size
    
    # 2. Remove Git objects (mamba.py/.git)
    git_dir = current_dir / "mamba.py" / ".git"
    if git_dir.exists():
        size = get_dir_size(git_dir)
        print(f"🗑️  Removing Git objects: {size/1024/1024:.1f} MB")
        shutil.rmtree(git_dir)
        freed_space += size
    
    # 3. Remove large checkpoint files (keep only essential ones)
    checkpoints_to_remove = [
        "kaggle_gpu_production/production_model_optimized.pt",
        "outputs/checkpoint_*.pt",
        "*.pth",
        "model_*.pt"
    ]
    
    for pattern in checkpoints_to_remove:
        for file_path in current_dir.rglob(pattern):
            if file_path.exists() and file_path.is_file():
                size = file_path.stat().st_size
                print(f"🗑️  Removing checkpoint: {file_path.name} ({size/1024/1024:.1f} MB)")
                file_path.unlink()
                freed_space += size
    
    # 4. Remove cache directories
    cache_dirs = [
        "__pycache__",
        ".pytest_cache",
        "*.egg-info",
        ".mypy_cache",
        "node_modules"
    ]
    
    for pattern in cache_dirs:
        for cache_dir in current_dir.rglob(pattern):
            if cache_dir.exists() and cache_dir.is_dir():
                size = get_dir_size(cache_dir)
                if size > 1024*1024:  # Only report if > 1MB
                    print(f"🗑️  Removing cache: {cache_dir.name} ({size/1024/1024:.1f} MB)")
                shutil.rmtree(cache_dir)
                freed_space += size
    
    # 5. Remove temporary files
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*.log",
        "*.out",
        ".DS_Store"
    ]
    
    for pattern in temp_patterns:
        for temp_file in current_dir.rglob(pattern):
            if temp_file.exists() and temp_file.is_file():
                size = temp_file.stat().st_size
                temp_file.unlink()
                freed_space += size
    
    print(f"\n✅ CLEANUP COMPLETE!")
    print(f"💾 Space freed: {freed_space/1024/1024:.1f} MB ({freed_space/1024/1024/1024:.2f} GB)")
    
    # Show what's kept
    print(f"\n📁 ESSENTIAL FILES KEPT:")
    essential_files = [
        "src/",
        "*.py scripts",
        "kaggle_submission/",
        "lightweight_mle_star_results/",
        "README.md",
        "CLAUDE.md"
    ]
    for item in essential_files:
        print(f"   ✅ {item}")
    
    return freed_space

def get_dir_size(path):
    """Get total size of directory."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total_size

def show_remaining_size():
    """Show remaining directory size after cleanup."""
    current_dir = Path("/Users/mekdesyared/amharic-hnet-v3/amharic-hnet-v2")
    remaining_size = get_dir_size(current_dir)
    print(f"\n📊 REMAINING SIZE: {remaining_size/1024/1024:.1f} MB ({remaining_size/1024/1024/1024:.2f} GB)")

if __name__ == "__main__":
    freed_space = cleanup_storage()
    show_remaining_size()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Your Kaggle models are trained and ready on Kaggle platform")
    print(f"2. Download only the models you need: best_model.pt")
    print(f"3. Essential code and scripts are preserved")
    print(f"4. Virtual environment removed - recreate with: python -m venv venv")
    
    print(f"\n💎 READY FOR KAGGLE GOLD MEDAL COMPETITION! 💎")