# Data Model: AI-Native Humanoid Robotics Textbook

## Core Entities

### Module
- **name**: string (e.g., "ROS 2: Robotic Nervous System")
- **description**: string - Overview of the module's learning objectives
- **chapters**: array[Chapter] - List of chapters in the module
- **learning_outcomes**: array[string] - High-level outcomes for the module
- **prerequisites**: array[string] - Knowledge required before starting

### Chapter
- **title**: string - Chapter title
- **purpose**: string - Purpose statement for the chapter
- **learning_outcomes**: array[string] - Specific learning outcomes
- **sections**: array[Section] - 3-6 conceptual sections
- **diagrams**: array[Diagram] - Minimum 1 diagram per chapter
- **code_examples**: array[CodeExample] - Minimum 2 runnable examples
- **exercises**: array[Exercise] - Hands-on simulation tasks
- **summary**: string - Chapter summary
- **further_reading**: array[string] - Additional resources
- **practice_questions**: array[string] - 3 practice questions
- **quiz_questions**: array[QuizQuestion] - "Check your understanding" MCQs
- **chunk_ids**: array[string] - RAG-anchorable text blocks
- **urdu_todo**: boolean - Flag for future Urdu translation

### Section
- **title**: string - Section title
- **content**: string - Main content of the section
- **concepts**: array[string] - Key concepts covered
- **examples**: array[ExampleReference] - References to examples in this section

### Diagram
- **id**: string - Unique identifier
- **type**: enum (mermaid, ascii, image) - Diagram type
- **content**: string - Diagram source code or path
- **caption**: string - Diagram caption
- **alt_text**: string - Accessibility alt text
- **related_concepts**: array[string] - Concepts illustrated by the diagram

### CodeExample
- **id**: string - Unique identifier
- **title**: string - Example title
- **description**: string - Purpose and functionality
- **language**: enum (python, cpp) - Programming language
- **source_code**: string - Complete source code
- **dependencies**: array[string] - Required packages/libraries
- **instructions**: string - How to run the example
- **ros_distro**: string - ROS2 distribution requirement
- **launch_commands**: array[string] - Commands to execute
- **expected_output**: string - What user should see
- **validation_script**: string - Script to verify functionality

### Exercise
- **id**: string - Unique identifier
- **title**: string - Exercise title
- **description**: string - Detailed description
- **difficulty**: enum (beginner, intermediate, advanced)
- **simulation_environment**: string - Required environment (Gazebo, Isaac, etc.)
- **steps**: array[string] - Step-by-step instructions
- **expected_outcome**: string - What should be achieved
- **validation_criteria**: array[string] - How to verify completion

### QuizQuestion
- **question**: string - The question text
- **options**: array[string] - Multiple choice options (A, B, C, D)
- **correct_answer**: string - The correct option
- **explanation**: string - Why this is correct
- **related_concept**: string - Which concept this tests

## Validation Rules

### Module Validation
- Must contain 3-6 chapters
- Must have a clear learning progression
- All referenced assets must exist

### Chapter Validation
- Must contain purpose and learning outcomes
- Must have 3-6 conceptual sections
- Must include minimum 1 diagram with caption/alt text
- Must include minimum 2 code examples with instructions
- Must include at least 1 hands-on exercise
- Must include summary, further reading, and 3 practice questions
- Must include 3 MCQs for "Check your understanding"
- Must include chunk IDs for RAG work
- Must include Urdu TODO marker if applicable

### Code Example Validation
- Must be executable in Ubuntu 22.04 + ROS2 environment
- Must include complete README with dependencies
- Must include launch commands
- Must match official ROS2/Isaac/Gazebo documentation

### Diagram Validation
- Must render correctly in Docusaurus
- Must include proper alt text for accessibility
- Must effectively illustrate concepts from the chapter

## State Transitions

### Chapter States
- `draft` → `review` (when initial content is complete)
- `review` → `revised` (when feedback is incorporated)
- `revised` → `validated` (when all requirements met)
- `validated` → `published` (when integrated into textbook)

### Module States
- `planning` → `in_progress` (when first chapter begins)
- `in_progress` → `complete` (when all chapters validated)
- `complete` → `published` (when integrated into textbook)