"""
Setup Verification Script
=========================
Verify all dependencies and configurations before running main pipeline

Author: Data Science Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def print_success(text):
    """Print success message"""
    print(f"✓ {text}")

def print_error(text):
    """Print error message"""
    print(f"✗ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠ {text}")

# =========================================================
# 1. PYTHON VERSION CHECK
# =========================================================
def check_python_version():
    """Check Python version"""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python Version: {version_str}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python 3.8+ required (current: {version_str})")
        return False

# =========================================================
# 2. PACKAGE INSTALLATION CHECK
# =========================================================
def check_packages():
    """Check if all required packages are installed"""
    print_header("PACKAGE INSTALLATION CHECK")
    
    required_packages = {
        'pandas': '1.3.0',
        'numpy': '1.21.0',
        'scikit-learn': '0.24.0',
        'gensim': '4.0.0',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0'
    }
    
    all_installed = True
    
    for package, min_version in required_packages.items():
        try:
            mod = __import__(package)
            version = mod.__version__
            print_success(f"{package:20s}: {version}")
        except ImportError:
            print_error(f"{package:20s}: NOT INSTALLED")
            all_installed = False
        except AttributeError:
            print_warning(f"{package:20s}: Version unknown (but installed)")
    
    return all_installed

# =========================================================
# 3. DIRECTORY STRUCTURE CHECK
# =========================================================
def check_directories():
    """Check if required directories exist"""
    print_header("DIRECTORY STRUCTURE CHECK")
    
    required_dirs = {
        'data': 'Data directory (for Train.csv and Test.csv)',
        'models': 'Models directory (for trained_model.pkl)',
        'outputs': 'Outputs directory (for results)'
    }
    
    all_exist = True
    
    for dir_name, description in required_dirs.items():
        if os.path.exists(dir_name):
            print_success(f"{dir_name:20s}: {description}")
        else:
            print_warning(f"{dir_name:20s}: {description} - Will be created")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print_success(f"              Created {dir_name}/")
            except Exception as e:
                print_error(f"              Failed to create {dir_name}: {str(e)}")
                all_exist = False
    
    return all_exist

# =========================================================
# 4. DATA FILES CHECK
# =========================================================
def check_data_files():
    """Check if data files exist"""
    print_header("DATA FILES CHECK")
    
    required_files = {
        'data/Train.csv': 'Training dataset',
        'data/Test.csv': 'Test dataset'
    }
    
    all_exist = True
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print_success(f"{file_path:25s}: {description} ({file_size:.2f} MB)")
        else:
            print_error(f"{file_path:25s}: {description} - NOT FOUND")
            all_exist = False
    
    return all_exist

# =========================================================
# 5. PYTHON SCRIPT CHECK
# =========================================================
def check_python_scripts():
    """Check if required Python scripts exist"""
    print_header("PYTHON SCRIPTS CHECK")
    
    required_scripts = {
        'notebooks/01_data_analysis.py': 'Data analysis script',
        'notebooks/02_model_training.py': 'Model training script',
        'notebooks/03_predictions.py': 'Predictions script'
    }
    
    all_exist = True
    
    for script_path, description in required_scripts.items():
        if os.path.exists(script_path):
            file_size = os.path.getsize(script_path) / 1024  # Convert to KB
            print_success(f"{script_path:40s}: {description} ({file_size:.1f} KB)")
        else:
            print_error(f"{script_path:40s}: {description} - NOT FOUND")
            all_exist = False
    
    return all_exist

# =========================================================
# 6. DOCUMENTATION CHECK
# =========================================================
def check_documentation():
    """Check if documentation files exist"""
    print_header("DOCUMENTATION CHECK")
    
    required_docs = {
        'README.md': 'Main documentation',
        'requirements.txt': 'Dependencies file',
        'QUICKSTART.md': 'Quick start guide',
        'PROJECT_STRUCTURE.md': 'Project structure guide'
    }
    
    all_exist = True
    
    for doc_file, description in required_docs.items():
        if os.path.exists(doc_file):
            file_size = os.path.getsize(doc_file) / 1024  # Convert to KB
            print_success(f"{doc_file:30s}: {description} ({file_size:.1f} KB)")
        else:
            print_warning(f"{doc_file:30s}: {description} - NOT FOUND")
    
    return True  # Documentation is helpful but not critical

# =========================================================
# 7. SYSTEM INFORMATION
# =========================================================
def check_system_info():
    """Display system information"""
    print_header("SYSTEM INFORMATION")
    
    import platform
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python Executable: {sys.executable}")
    
    try:
        import psutil
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    except ImportError:
        print_warning("psutil not installed - Cannot display CPU/RAM info")

# =========================================================
# 8. CONFIGURATION TEST
# =========================================================
def test_imports():
    """Test critical imports"""
    print_header("CRITICAL IMPORTS TEST")
    
    imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn.ensemble', 'RandomForestClassifier'),
        ('sklearn.impute', 'SimpleImputer'),
        ('sklearn.preprocessing', 'LabelEncoder'),
        ('gensim.models', 'Word2Vec')
    ]
    
    all_success = True
    
    for import_module, import_name in imports:
        try:
            exec(f"from {import_module} import {import_name}")
            print_success(f"from {import_module} import {import_name}")
        except ImportError as e:
            print_error(f"from {import_module} import {import_name}")
            all_success = False
    
    return all_success

# =========================================================
# 9. DISK SPACE CHECK
# =========================================================
def check_disk_space():
    """Check available disk space"""
    print_header("DISK SPACE CHECK")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        total_gb = total / (1024**3)
        free_gb = free / (1024**3)
        
        print(f"Total Disk Space: {total_gb:.2f} GB")
        print(f"Free Disk Space: {free_gb:.2f} GB")
        print(f"Used Disk Space: {(total_gb - free_gb):.2f} GB")
        
        if free_gb >= 2:
            print_success("Sufficient disk space available (2+ GB required)")
            return True
        else:
            print_warning("Low disk space (2+ GB recommended)")
            return False
    except Exception as e:
        print_warning(f"Could not check disk space: {str(e)}")
        return True

# =========================================================
# 10. SUMMARY AND RECOMMENDATIONS
# =========================================================
def print_summary(results):
    """Print summary of all checks"""
    print_header("VERIFICATION SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"Total Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n" + "✓ " * 40)
        print("\nALL CHECKS PASSED! ✨".center(80))
        print("\nYou're ready to run the pipeline:".center(80))
        print("\n  1. python notebooks/01_data_analysis.py".center(80))
        print("  2. python notebooks/02_model_training.py".center(80))
        print("  3. python notebooks/03_predictions.py".center(80))
        print("\n" + "✓ " * 40)
    else:
        print(f"\n⚠ {failed} check(s) failed. Please address issues above.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Create directories: mkdir -p data models outputs")
        print("  - Place Train.csv and Test.csv in data/ directory")
    
    return failed == 0

# =========================================================
# 11. MAIN EXECUTION
# =========================================================
def main():
    """Run all verification checks"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + "SME Financial Health Prediction - Setup Verification".center(78) + "║")
    print("║" + "Checking dependencies, files, and configurations".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    results['Packages'] = check_packages()
    results['Directories'] = check_directories()
    results['Data Files'] = check_data_files()
    results['Python Scripts'] = check_python_scripts()
    check_documentation()  # Not critical
    check_system_info()
    results['Imports'] = test_imports()
    results['Disk Space'] = check_disk_space()
    
    # Print summary
    success = print_summary(results)
    
    print("\n")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
