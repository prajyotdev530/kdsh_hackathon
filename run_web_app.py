#!/usr/bin/env python3
"""
Narrative Consistency Checker - Web Application Runner

This script starts the Flask web application for the narrative consistency checker.
The web app provides a user-friendly interface to check if statements are consistent
with classic literature narratives.

Usage:
    python run_web_app.py

Or make it executable and run:
    chmod +x run_web_app.py
    ./run_web_app.py

Then open your browser to: http://localhost:8000
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'sentence_transformers',
        'sklearn',
        'pandas',
        'numpy',
        'imblearn'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'sentence_transformers':
                import sentence_transformers
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages. Please install them:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'consistency_checker.py',
        'templates/index.html',
        'In_Search_of_the_Castaways_embeddings.pkl',
        'The_Count_of_Monte_Cristo_embeddings.pkl'
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure you have run the training pipeline first:")
        print("python consistency_checker.py")
        return False

    return True

def main():
    print("🚀 Narrative Consistency Checker - Web App Launcher")
    print("=" * 60)

    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        return 1

    # Check files
    print("📁 Checking files...")
    if not check_files():
        return 1

    print("✅ All checks passed!")

    # Start the web application
    print("\n🌐 Starting web application...")
    print("📖 Once started, open your browser to: http://localhost:8000")
    print("❌ Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Web application stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running web application: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())