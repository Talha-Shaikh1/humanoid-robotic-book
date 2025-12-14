#!/usr/bin/env python3
"""
Chapter Validation Script for AI-Native Humanoid Robotics Textbook

This script validates that a chapter meets all the requirements specified in the textbook specification.
"""

import argparse
import os
import re
import sys
from pathlib import Path
import yaml


def validate_chapter(filepath):
    """Validate a single chapter file against the textbook requirements."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not frontmatter_match:
        print(f"❌ FAIL: No frontmatter found in {filepath}")
        return False

    frontmatter = yaml.safe_load(frontmatter_match.group(1))

    issues = []
    warnings = []

    # Check required frontmatter fields
    required_fields = ['title', 'description', 'sidebar_position']
    for field in required_fields:
        if field not in frontmatter:
            issues.append(f"Missing required frontmatter field: {field}")

    # Check for learning outcomes
    if 'learning_outcomes' not in frontmatter or not frontmatter['learning_outcomes']:
        issues.append("Missing or empty learning_outcomes in frontmatter")
    elif len(frontmatter['learning_outcomes']) < 3:
        warnings.append("Less than 3 learning outcomes specified")

    # Check content structure
    content_without_frontmatter = content[frontmatter_match.end():].strip()

    # Check for required sections
    required_sections = ['Purpose', 'Learning Outcomes', 'Summary', 'Further Reading']
    for section in required_sections:
        if f'## {section}' not in content_without_frontmatter:
            issues.append(f"Missing required section: {section}")

    # Count conceptual sections (should be 3-6)
    conceptual_sections = re.findall(r'^## (?!Purpose|Learning Outcomes|Summary|Further Reading|Check Your Understanding|Hands-on Exercise|Diagram|Code Examples|Practice Questions)[\w\s-]+',
                                    content_without_frontmatter, re.MULTILINE)
    if len(conceptual_sections) < 3:
        issues.append(f"Fewer than 3 conceptual sections found: {len(conceptual_sections)}")
    elif len(conceptual_sections) > 6:
        issues.append(f"More than 6 conceptual sections found: {len(conceptual_sections)}")

    # Check for diagrams (minimum 1)
    diagram_count = 0
    if '![{' in content_without_frontmatter or '<img' in content_without_frontmatter:
        diagram_count += 1
    # Check for mermaid diagrams
    if '```mermaid' in content_without_frontmatter:
        diagram_count += 1
    if diagram_count < 1:
        issues.append("No diagrams found in chapter")

    # Check for code examples (minimum 2)
    code_example_count = len(re.findall(r'```python|```cpp|```bash', content_without_frontmatter))
    if code_example_count < 2:
        issues.append(f"Fewer than 2 code examples found: {code_example_count}")

    # Check for exercises
    if 'Hands-on Exercise' not in content_without_frontmatter:
        issues.append("No hands-on exercise found")

    # Check for practice questions (minimum 3)
    practice_questions = re.findall(r'\d+\.\s*.*\?', content_without_frontmatter)
    if len(practice_questions) < 3:
        issues.append(f"Fewer than 3 practice questions found: {len(practice_questions)}")

    # Check for MCQs
    if 'Check Your Understanding' not in content_without_frontmatter:
        issues.append("No 'Check Your Understanding' section found")
    else:
        mcq_match = re.search(r'## Check Your Understanding.*?(?=\n## |\Z)', content_without_frontmatter, re.DOTALL)
        if mcq_match:
            mcq_content = mcq_match.group(0)
            mcq_count = len(re.findall(r'\d+\.\s*.*\?', mcq_content))
            if mcq_count < 3:
                issues.append(f"Fewer than 3 MCQs found: {mcq_count}")

    # Check for RAG chunk IDs
    rag_chunks = re.findall(r'<!-- RAG_CHUNK_ID:', content_without_frontmatter)
    if len(rag_chunks) == 0:
        warnings.append("No RAG chunk IDs found")

    # Check for Urdu translation TODOs
    urdu_todos = re.findall(r'<!-- URDU_TODO:', content_without_frontmatter)
    if len(urdu_todos) == 0:
        warnings.append("No Urdu translation TODOs found")

    # Word count check (800-2000 words)
    words = len(content_without_frontmatter.split())
    if words < 800:
        issues.append(f"Chapter too short: {words} words (minimum 800 required)")
    elif words > 2000:
        issues.append(f"Chapter too long: {words} words (maximum 2000 recommended)")

    # Report results
    if issues:
        print(f"❌ FAIL: {filepath}")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print(f"✅ PASS: {filepath}")
        if warnings:
            for warning in warnings:
                print(f"   ⚠️  {warning}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate textbook chapters')
    parser.add_argument('files', nargs='+', help='Chapter files to validate')
    parser.add_argument('--all', action='store_true', help='Validate all chapters in docs directory')

    args = parser.parse_args()

    files_to_check = []

    if args.all:
        # Find all markdown files in docs directory
        docs_path = Path('docusaurus/docs')
        for ext in ['*.md', '*.mdx']:
            files_to_check.extend(docs_path.rglob(ext))
    else:
        files_to_check = [Path(f) for f in args.files]

    total_files = len(files_to_check)
    passed_files = 0

    print(f"Validating {total_files} chapter(s)...\n")

    for file_path in files_to_check:
        if validate_chapter(file_path):
            passed_files += 1
        print()  # Empty line for readability

    print(f"Results: {passed_files}/{total_files} chapters passed validation")

    if passed_files == total_files and total_files > 0:
        print("🎉 All chapters passed validation!")
        return 0
    else:
        print(f"❌ {total_files - passed_files} chapter(s) failed validation")
        return 1


if __name__ == '__main__':
    sys.exit(main())