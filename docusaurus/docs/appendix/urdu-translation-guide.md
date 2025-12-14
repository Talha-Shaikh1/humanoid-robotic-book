---
title: Urdu Translation Implementation Guide
description: Guide for implementing Urdu translation placeholders
sidebar_position: 3
---

# Urdu Translation Implementation Guide

## Overview

This document explains how to implement Urdu translation placeholders in the textbook content. These placeholders are for future multi-language support and do not affect current functionality.

## Implementation Method

### HTML Comments for Urdu Translation

Translation placeholders are marked with HTML comments in the following format:

```html
<!-- URDU_TODO: Translate this section to Urdu -->
```

### Placement Guidelines

1. **Major Sections**: Place after section headers
2. **Paragraphs**: Place at the end of significant paragraphs
3. **Code Explanations**: Place after code blocks with explanations
4. **Exercises**: Place after exercise descriptions
5. **Assessments**: Place after quiz questions

### Examples

#### For Section Headers
```markdown
## ROS2 Nodes
<!-- URDU_TODO: Translate this section to Urdu -->

In ROS2, nodes are...
```

#### For Paragraphs
```markdown
In ROS2, nodes are executable units that communicate with each other.
<!-- URDU_TODO: Translate this paragraph to Urdu -->
This communication happens through topics, services, and actions.
<!-- URDU_TODO: Translate this paragraph to Urdu -->
```

#### For Code Explanations
```markdown
```python
import rclpy
from rclpy.node import Node
```

The above code imports necessary modules for creating a ROS2 node.
<!-- URDU_TODO: Translate this explanation to Urdu -->
```

## Best Practices

### For Authors
- Place TODO comments consistently throughout content
- Focus on content that will need translation (not code syntax)
- Use the same format for all translation placeholders
- Do not attempt to translate content at this stage

### Content Considerations
- Technical terms may need special handling
- Maintain technical accuracy in future translations
- Consider cultural adaptation for examples and analogies

## Validation Process

### Current Validation
- Verify all URDU_TODO placeholders follow the correct format
- Ensure placeholders don't interfere with current content display
- Confirm placeholders are properly formatted HTML comments

### Future Validation
When translations are implemented:
- Verify technical accuracy of translated terms
- Ensure translated content maintains educational value
- Test that translated content renders correctly

## Implementation Scope

### What to Mark
- Explanatory paragraphs
- Exercise descriptions
- Quiz questions and options
- Section introductions
- Concept summaries

### What NOT to Mark
- Code syntax and variable names
- Technical terms that remain the same across languages
- File paths and system commands
- Mathematical formulas

## Metadata Only

These URDU_TODO placeholders are purely metadata for future translation work and:
- Do not affect current textbook functionality
- Are not visible to end users
- Serve as markers for future localization efforts
- Will be processed by future translation tools/systems

The implementation is metadata-only and does not require any translation at this time.