"""
CLI helper script with common usage patterns for ValueVision.

This script provides convenient shortcuts for common operations.
"""

import subprocess
import sys
import os


def run_command(cmd):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("""
ValueVision CLI Helper

Usage:
    python cli.py full                # Full pipeline from scratch
    python cli.py quick               # Use existing data if available
    python cli.py analyze             # Quick analysis only
    python cli.py reload              # Force reload existing data
    python cli.py specific DATASETS   # Process specific datasets (e.g., Electronics Appliances)

Examples:
    python cli.py quick               # Fast mode - uses existing data
    python cli.py analyze             # Just show analysis of existing data
    python cli.py specific Electronics Automotive  # Process only these datasets
        """)
        return

    mode = sys.argv[1].lower()

    if mode == "full":
        run_command("python main.py")
    
    elif mode == "quick":
        run_command("python main.py --use-existing")
    
    elif mode == "analyze":
        run_command("python main.py --analysis-only")
    
    elif mode == "reload":
        run_command("python main.py --use-existing --force-reload")
    
    elif mode == "specific":
        if len(sys.argv) < 3:
            print("Error: Please specify dataset names after 'specific'")
            return
        datasets = " ".join(sys.argv[2:])
        run_command(f"python main.py --datasets {datasets}")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'python cli.py' without arguments to see available options")


if __name__ == '__main__':
    main()
