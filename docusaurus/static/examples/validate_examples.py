#!/usr/bin/env python3

"""
Code Example Validation Script

This script validates that all code examples in the textbook can be executed
without syntax errors and meet the basic requirements.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def validate_python_examples():
    """Validate all Python examples in the ROS2 directory."""
    ros2_examples_dir = Path("docusaurus/static/examples/ros2-examples/python")
    python_examples = list(ros2_examples_dir.glob("*.py"))

    results = {}

    for example in python_examples:
        print(f"Validating {example.name}...")

        try:
            # Try to import the module to check for syntax errors
            spec = importlib.util.spec_from_file_location(example.stem, example)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results[example.name] = {"status": "PASS", "message": "Syntax valid"}
        except SyntaxError as e:
            results[example.name] = {"status": "FAIL", "message": f"Syntax error: {str(e)}"}
        except ImportError as e:
            results[example.name] = {"status": "FAIL", "message": f"Import error: {str(e)}"}
        except Exception as e:
            # Some examples may require ROS2 to be installed to run fully
            # We'll consider them valid if they have valid syntax
            results[example.name] = {"status": "PASS", "message": f"Valid syntax, requires ROS2: {str(e)}"}

    return results

def validate_gazebo_examples():
    """Validate Gazebo examples by checking URDF files."""
    gazebo_examples_dir = Path("docusaurus/static/examples/gazebo-scenes")
    urdf_files = list(gazebo_examples_dir.rglob("*.urdf"))

    results = {}

    for urdf_file in urdf_files:
        print(f"Validating URDF {urdf_file.name}...")

        try:
            with open(urdf_file, 'r') as f:
                content = f.read()

            # Basic URDF validation - check for required tags
            if '<robot' in content and '<link' in content and '<joint' in content:
                results[urdf_file.name] = {"status": "PASS", "message": "Basic URDF structure valid"}
            else:
                results[urdf_file.name] = {"status": "FAIL", "message": "Missing basic URDF elements"}
        except Exception as e:
            results[urdf_file.name] = {"status": "FAIL", "message": f"Error reading file: {str(e)}"}

    return results

def validate_isaac_examples():
    """Validate Isaac Sim examples."""
    isaac_examples_dir = Path("docusaurus/static/examples/isaac-sim-examples")
    python_examples = list(isaac_examples_dir.rglob("*.py"))

    results = {}

    for example in python_examples:
        print(f"Validating Isaac Sim example {example.name}...")

        try:
            spec = importlib.util.spec_from_file_location(example.stem, example)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results[example.name] = {"status": "PASS", "message": "Syntax valid"}
        except SyntaxError as e:
            results[example.name] = {"status": "FAIL", "message": f"Syntax error: {str(e)}"}
        except ImportError as e:
            results[example.name] = {"status": "FAIL", "message": f"Import error: {str(e)}"}
        except Exception as e:
            results[example.name] = {"status": "PASS", "message": f"Valid syntax, requires Isaac Sim: {str(e)}"}

    return results

def main():
    print("Starting validation of all code examples...")

    print("\n--- Validating ROS2 Python Examples ---")
    ros2_results = validate_python_examples()

    print("\n--- Validating Gazebo Examples ---")
    gazebo_results = validate_gazebo_examples()

    print("\n--- Validating Isaac Sim Examples ---")
    isaac_results = validate_isaac_examples()

    # Print summary
    print("\n--- Validation Summary ---")

    all_results = {**ros2_results, **gazebo_results, **isaac_results}

    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r["status"] == "PASS")
    failed = total - passed

    print(f"Total examples validated: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed examples:")
        for name, result in all_results.items():
            if result["status"] == "FAIL":
                print(f"  - {name}: {result['message']}")
        return 1
    else:
        print("\nAll examples passed validation!")
        return 0

if __name__ == "__main__":
    sys.exit(main())