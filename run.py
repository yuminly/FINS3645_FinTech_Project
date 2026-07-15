#!/usr/bin/env python3
"""
FinVest Pro - Systematic Multi-Asset Investment Platform
FINS3645 FinTech Project 2026

Run this script to start the Streamlit application.
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if all required packages are installed."""
    required = [
        'yfinance', 'pandas', 'numpy', 'scipy', 
        'streamlit', 'plotly', 'vaderSentiment'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])


def main():
    """Main entry point."""
    print("=" * 60)
    print("FinVest Pro - Systematic Multi-Asset Investment Platform")
    print("FINS3645 FinTech Project 2026")
    print("=" * 60)
    
    check_dependencies()
    
    app_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
    
    print("\nStarting application...")
    print("Open http://localhost:8501 in your browser\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path
    ])


if __name__ == "__main__":
    main()
