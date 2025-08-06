#!/usr/bin/env python3
"""
Kaggle API Setup Script
Configures Kaggle API credentials for competition submission
"""

import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials from local file."""
    print("🔑 Setting up Kaggle API credentials...")
    
    # Load credentials from local file
    creds_file = Path("kaggle_credentials.json")
    if not creds_file.exists():
        print("❌ kaggle_credentials.json not found!")
        return False
    
    with open(creds_file, 'r') as f:
        credentials = json.load(f)
    
    # Create Kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Write kaggle.json
    kaggle_json = kaggle_dir / "kaggle.json"
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    # Set proper permissions (required by Kaggle API)
    os.chmod(kaggle_json, 0o600)
    
    print(f"✅ Kaggle credentials saved to: {kaggle_json}")
    print(f"👤 Username: {credentials['username']}")
    print("🔐 API key configured securely")
    
    # Test the API
    try:
        import kaggle
        kaggle.api.authenticate()
        print("✅ Kaggle API authentication successful!")
        return True
    except Exception as e:
        print(f"⚠️  Kaggle API test failed: {e}")
        print("💡 You may need to install kaggle: pip install kaggle")
        return False

def main():
    """Main setup function."""
    print("🚀 KAGGLE API SETUP")
    print("=" * 50)
    
    success = setup_kaggle_credentials()
    
    if success:
        print("\n✅ KAGGLE API SETUP COMPLETE!")
        print("🏆 Ready for competition submissions!")
    else:
        print("\n❌ SETUP FAILED")
        print("🔧 Please check the error messages above")
    
    print("=" * 50)

if __name__ == "__main__":
    main()