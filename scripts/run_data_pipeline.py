"""
Run the complete data processing pipeline.
This script runs all data processing steps in the correct order:
1. Data cleaning
2. Feature engineering
3. Data transformation
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Run the complete data processing pipeline"""
    print("=" * 60)
    print("Starting Data Processing Pipeline")
    print("=" * 60)
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    
    # Step 1: Data Cleaning
    print("\n[Step 1/3] Data Cleaning...")
    print("-" * 60)
    try:
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "data_cleaning.py")],
            cwd=str(project_root),
            check=True,
            capture_output=False
        )
        print("✓ Data cleaning complete")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in data cleaning: {e}")
        sys.exit(1)
    
    # Step 2: Feature Engineering
    print("\n[Step 2/3] Feature Engineering...")
    print("-" * 60)
    try:
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "feature_engineering.py")],
            cwd=str(project_root),
            check=True,
            capture_output=False
        )
        print("✓ Feature engineering complete")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in feature engineering: {e}")
        sys.exit(1)
    
    # Step 3: Data Transformation
    print("\n[Step 3/3] Data Transformation...")
    print("-" * 60)
    try:
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "data_transformation.py")],
            cwd=str(project_root),
            check=True,
            capture_output=False
        )
        print("✓ Data transformation complete")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in data transformation: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Data Processing Pipeline Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

