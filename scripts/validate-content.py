#!/usr/bin/env python3
"""
Content Validation Script for AI-Native Humanoid Robotics Textbook

This script validates that textbook content conforms to official ROS 2, Isaac, and Gazebo documentation standards.
"""

import argparse
import os
import re
import sys
from pathlib import Path


def validate_ros2_content(filepath):
    """Validate ROS2-related content against official standards."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    warnings = []

    # Check for common ROS2 terminology accuracy
    ros2_checks = [
        (r'ros\s*2|ros2', 'ROS 2', 'Use "ROS 2" with space and capitalization'),
        (r'rospkg', 'ament_cmake', 'Use "ament_cmake" or "ament_python" instead of "rospkg" for ROS 2'),
        (r'catkin', 'ament', 'Use "ament" build system instead of "catkin" in ROS 2'),
        (r'roslaunch', 'ros2 launch', 'Use "ros2 launch" instead of "roslaunch" in ROS 2'),
        (r'rosrun', 'ros2 run', 'Use "ros2 run" instead of "rosrun" in ROS 2'),
        (r'rostopic', 'ros2 topic', 'Use "ros2 topic" instead of "rostopic" in ROS 2'),
        (r'rosservice', 'ros2 service', 'Use "ros2 service" instead of "rosservice" in ROS 2'),
        (r'rosnode', 'ros2 node', 'Use "ros2 node" instead of "rosnode" in ROS 2'),
        (r'rosmsg', 'ros2 msg', 'Use "ros2 msg" instead of "rosmsg" in ROS 2'),
        (r'rossrv', 'ros2 srv', 'Use "ros2 srv" instead of "rossrv" in ROS 2'),
        (r'rosparam', 'ros2 param', 'Use "ros2 param" instead of "rosparam" in ROS 2'),
    ]

    for pattern, correct, message in ros2_checks:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"ROS 2 terminology: {message}")

    # Check for ROS2 node creation patterns
    if re.search(r'rclpy\.init\(\)|Node\(\)', content):
        # Check if proper ROS2 initialization is shown
        if not re.search(r'import rclpy', content):
            warnings.append("Possible ROS2 Python code without proper import")
        if not re.search(r'rclpy\.spin|executor\.', content):
            warnings.append("Possible ROS2 node without proper spin/executor pattern")

    return issues, warnings


def validate_gazebo_content(filepath):
    """Validate Gazebo-related content against official standards."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    warnings = []

    # Check for Gazebo terminology
    gazebo_checks = [
        (r'gazebo\s+simulator|gazebo\s+simulation', 'Gazebo', 'Use "Gazebo" without redundant terms'),
        (r'gazebo_server', 'gz sim|gazebo', 'Use appropriate command for starting Gazebo'),
    ]

    for pattern, correct, message in gazebo_checks:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Gazebo terminology: {message}")

    return issues, warnings


def validate_isaac_content(filepath):
    """Validate Isaac-related content against official standards."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    warnings = []

    # Check for Isaac terminology
    isaac_checks = [
        (r'isaac\s+sim', 'Isaac Sim', 'Use "Isaac Sim" with proper capitalization'),
        (r'isaac\s+ros', 'Isaac ROS', 'Use "Isaac ROS" with proper capitalization'),
    ]

    for pattern, correct, message in isaac_checks:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Isaac terminology: {message}")

    return issues, warnings


def validate_content_standards(filepath):
    """Validate content against general robotics documentation standards."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    warnings = []

    # Check for technical accuracy patterns
    accuracy_checks = [
        (r'it is recommended that|it is suggested that', 'Recommend using direct, authoritative statements instead of tentative language'),
        (r'might be|could be|possibly', 'Use definitive technical language when possible'),
    ]

    for pattern, message in accuracy_checks:
        if re.search(pattern, content, re.IGNORECASE):
            warnings.append(f"Technical writing: {message}")

    # Check for code formatting consistency
    code_blocks = re.findall(r'```.*?\n(.*?)```', content, re.DOTALL)
    for block in code_blocks:
        if 'TODO' in block or 'FIXME' in block:
            warnings.append(f"Found TODO/FIXME in code block")

    return issues, warnings


def validate_file(filepath):
    """Validate a single file against all standards."""
    print(f"Validating: {filepath}")

    all_issues = []
    all_warnings = []

    # Run all validation checks
    issues, warnings = validate_ros2_content(filepath)
    all_issues.extend(issues)
    all_warnings.extend(warnings)

    issues, warnings = validate_gazebo_content(filepath)
    all_issues.extend(issues)
    all_warnings.extend(warnings)

    issues, warnings = validate_isaac_content(filepath)
    all_issues.extend(issues)
    all_warnings.extend(warnings)

    issues, warnings = validate_content_standards(filepath)
    all_issues.extend(issues)
    all_warnings.extend(warnings)

    # Report results
    if all_issues:
        print(f"❌ ISSUES FOUND in {filepath}:")
        for issue in all_issues:
            print(f"   • {issue}")

    if all_warnings:
        print(f"⚠️  WARNINGS in {filepath}:")
        for warning in all_warnings:
            print(f"   • {warning}")

    if not all_issues and not all_warnings:
        print(f"✅ {filepath} passed content validation")

    print()  # Empty line for readability

    return len(all_issues) == 0  # Return True if no issues found


def main():
    parser = argparse.ArgumentParser(description='Validate textbook content against official documentation standards')
    parser.add_argument('files', nargs='+', help='Files to validate')
    parser.add_argument('--all', action='store_true', help='Validate all content files in docusaurus/docs directory')

    args = parser.parse_args()

    files_to_check = []

    if args.all:
        # Find all markdown files in docusaurus/docs directory
        docs_path = Path('docusaurus/docs')
        for ext in ['*.md', '*.mdx']:
            files_to_check.extend(docs_path.rglob(ext))
    else:
        files_to_check = [Path(f) for f in args.files]

    total_files = len(files_to_check)
    passed_files = 0

    print(f"Validating {total_files} file(s) against official documentation standards...\n")

    for file_path in files_to_check:
        if validate_file(file_path):
            passed_files += 1

    print(f"Content validation results: {passed_files}/{total_files} files passed")

    if passed_files == total_files and total_files > 0:
        print("🎉 All files passed content validation!")
        return 0
    else:
        print(f"❌ {total_files - passed_files} file(s) had issues that should be addressed")
        return 1


if __name__ == '__main__':
    sys.exit(main())