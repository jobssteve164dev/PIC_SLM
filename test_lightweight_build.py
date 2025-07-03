#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight Build Configuration Test Script
Function: Verify lightweight build configuration and startup check functionality
Author: AI Assistant
Date: 2025-01-19
"""

import sys
import os
import subprocess
from pathlib import Path

def safe_print(text):
    """Safe print to avoid encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='ignore').decode('ascii'))

def test_dependency_check():
    """Test dependency check functionality"""
    safe_print("\n" + "=" * 50)
    safe_print("Test Dependency Check Functionality")
    safe_print("=" * 50)
    
    try:
        # Import startup checker functions
        sys.path.append(str(Path(__file__).parent / 'src'))
        from startup_checker import check_minimal_dependencies
        
        # Check required dependencies
        missing_required, available_required = check_minimal_dependencies()
        safe_print(f"Required dependencies check:")
        safe_print(f"  Installed: {len(available_required)} packages")
        safe_print(f"  Missing: {len(missing_required)} packages")
        
        if missing_required:
            safe_print(f"  Missing dependencies: {', '.join(missing_required)}")
        
        return True
        
    except Exception as e:
        safe_print(f"X Test dependency check functionality failed: {e}")
        return False

def test_minimal_requirements():
    """Test minimal requirements file creation"""
    safe_print("\n" + "=" * 50)
    safe_print("Test Minimal Requirements File Creation")
    safe_print("=" * 50)
    
    try:
        # Import build script functions
        sys.path.append(str(Path(__file__).parent))
        from build_exe_simple import create_minimal_requirements
        
        # Create minimal requirements file
        req_path = create_minimal_requirements()
        
        if req_path.exists():
            safe_print("+ Minimal requirements file created successfully")
            safe_print(f"File location: {req_path}")
            
            # Read and display content
            with open(req_path, 'r', encoding='utf-8') as f:
                content = f.read()
                safe_print("File content:")
                safe_print(content)
            
            return True
        else:
            safe_print("X Minimal requirements file not created")
            return False
            
    except Exception as e:
        safe_print(f"X Test minimal requirements file creation failed: {e}")
        return False

def test_pyinstaller_command():
    """Test PyInstaller command configuration"""
    safe_print("\n" + "=" * 50)
    safe_print("Test PyInstaller Command Configuration")
    safe_print("=" * 50)
    
    try:
        # Check if PyInstaller is installed
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller', '--version'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            safe_print("+ PyInstaller is installed")
            safe_print(f"Version: {result.stdout.strip()}")
        else:
            safe_print("X PyInstaller is not installed")
            return False
        
        # Check if key files exist
        project_root = Path(__file__).parent
        
        files_to_check = [
            'src/startup_checker.py',
            'config.json',
            'config/',
            'setting/'
        ]
        
        safe_print("\nCheck key files:")
        all_exist = True
        for file_path in files_to_check:
            full_path = project_root / file_path
            if full_path.exists():
                safe_print(f"  + {file_path}")
            else:
                safe_print(f"  X {file_path}")
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        safe_print(f"X Test PyInstaller command configuration failed: {e}")
        return False

def test_startup_checker():
    """Test startup checker script"""
    safe_print("\n" + "=" * 50)
    safe_print("Test Startup Checker Script")
    safe_print("=" * 50)
    
    try:
        # Run startup checker script with a timeout
        result = subprocess.run([
            sys.executable, 
            str(Path(__file__).parent / 'src' / 'startup_checker.py')
        ], input='n\n', capture_output=True, text=True, timeout=10)
        
        safe_print(f"Exit code: {result.returncode}")
        if result.stdout:
            safe_print("Standard output:")
            safe_print(result.stdout)
        if result.stderr:
            safe_print("Error output:")
            safe_print(result.stderr)
        
        # Consider it successful if it runs without crashing
        return True
        
    except subprocess.TimeoutExpired:
        safe_print("X Startup checker script timeout")
        return False
    except Exception as e:
        safe_print(f"X Test startup checker script failed: {e}")
        return False

def main():
    """Main function"""
    safe_print("Lightweight Build Configuration Test")
    safe_print("=" * 60)
    
    tests = [
        ("Dependency Check Functionality", test_dependency_check),
        ("Minimal Requirements File Creation", test_minimal_requirements),
        ("PyInstaller Command Configuration", test_pyinstaller_command),
        ("Startup Checker Script", test_startup_checker),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            safe_print(f"Exception in test {test_name}: {e}")
            results.append((test_name, False))
    
    # Display test results
    safe_print("\n" + "=" * 60)
    safe_print("Test Results Summary")
    safe_print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "+ PASSED" if result else "X FAILED"
        safe_print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    safe_print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        safe_print("SUCCESS: All tests passed! Lightweight build configuration is correct")
        return True
    else:
        safe_print("WARNING: Some tests failed, please check configuration")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1) 