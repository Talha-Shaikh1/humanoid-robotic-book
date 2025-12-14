# Quickstart Guide: AI-Native Humanoid Robotics Textbook Development

## Prerequisites

- Ubuntu 22.04 LTS (or equivalent virtual machine)
- ROS2 Humble Hawksbill installed
- Node.js 18+ and npm/yarn
- Git and GitHub access
- Basic understanding of robotics concepts

## Setup Development Environment

### 1. Clone the Repository
```bash
git clone <repository-url>
cd humanoid-robotics-ai-book
```

### 2. Install Docusaurus Dependencies
```bash
cd docusaurus
npm install
```

### 3. Set up ROS2 Environment
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Create workspace for code examples
mkdir -p ~/ros2_textbook_ws/src
cd ~/ros2_textbook_ws
colcon build
source install/setup.bash
```

### 4. Install Additional Tools
```bash
# Install Gazebo Garden (or Fortress)
sudo apt install ros-humble-gazebo-*

# Install Isaac Sim (if available)
# Follow NVIDIA's installation guide
```

## Running the Textbook Locally

### 1. Start Docusaurus Development Server
```bash
cd docusaurus
npm start
```

### 2. Access the Textbook
Open your browser to `http://localhost:3000` to view the textbook.

## Creating New Content

### 1. Chapter Template
Each chapter should follow this structure in `docusaurus/docs/module-x-name/chapter-y-title.md`:

```markdown
---
title: Chapter Title
description: Chapter description
sidebar_position: Y
learning_outcomes:
  - Understand concept A
  - Implement technique B
  - Apply principle C
---

# Chapter Title

## Purpose
Brief explanation of the chapter's purpose and relevance.

## Learning Outcomes
- Outcome 1
- Outcome 2
- Outcome 3

## Section 1: Topic
Content for the first section...

## Section 2: Topic
Content for the second section...

## Diagrams
Include at least one diagram with proper caption and alt text:
![Diagram caption](/img/diagrams/diagram-name.png)

## Code Examples
Include at least 2 runnable code examples with complete instructions:

```python
# Example code with comments
import rclpy
from rclpy.node import Node

class ExampleNode(Node):
    def __init__(self):
        super().__init__('example_node')
        # Implementation
```

**Instructions:**
1. Create the node file
2. Build with `colcon build`
3. Run with `ros2 run package_name example_node`

## Hands-on Exercise
Include a practical exercise for students to complete.

## Summary
Brief summary of key concepts covered.

## Further Reading
Links to additional resources.

## Practice Questions
1. Question 1?
2. Question 2?
3. Question 3?

## Check Your Understanding
**Multiple Choice Questions:**
1. What is the primary purpose of ROS2 topics?
   A) Service calls between nodes
   B) Publish-subscribe communication
   C) Parameter storage
   D) Action execution

2. Which command is used to run a ROS2 node?
   A) ros2 launch
   B) ros2 run
   C) ros2 start
   D) ros2 execute

3. What does TF2 stand for in ROS2?
   A) Transform Framework 2
   B) Time Framework 2
   C) Task Framework 2
   D) Transfer Framework 2

## Chunk IDs for RAG
<!-- RAG_CHUNK_ID: chapter-concept-1 -->
<!-- RAG_CHUNK_ID: chapter-concept-2 -->

## TODO: Urdu Translation
<!-- URDU_TODO: Translate this chapter to Urdu -->
```

### 2. Adding Code Examples
Place code examples in `docusaurus/static/examples/` with proper directory structure:
```
static/examples/
├── ros2-examples/
│   ├── python/
│   │   └── chapter-1/
│   │       ├── example1.py
│   │       └── README.md
│   └── cpp/
│       └── chapter-1/
│           ├── example1.cpp
│           └── README.md
```

Each README should include:
- Dependencies required
- Build instructions
- Run commands
- Expected output

### 3. Adding Diagrams
Place diagrams in `docusaurus/static/img/diagrams/` with proper organization:
```
static/img/diagrams/
├── mermaid/
├── ascii/
└── module-specific/
```

## Validation Process

### 1. Chapter Validation Checklist
Before marking a chapter complete, verify:
- [ ] Learning outcomes are clear and measurable
- [ ] All 3-6 sections are complete with proper content
- [ ] At least 1 diagram with caption and alt text
- [ ] At least 2 code examples with complete instructions
- [ ] At least 1 hands-on exercise
- [ ] Summary, further reading, and 3 practice questions
- [ ] 3 MCQs in "Check your understanding"
- [ ] RAG chunk IDs included
- [ ] Urdu TODO marker if applicable
- [ ] All code examples tested in Ubuntu 22.04 + ROS2 environment

### 2. Code Example Testing
```bash
# Navigate to the example directory
cd ~/ros2_textbook_ws/src/your_example_package

# Build the package
cd ~/ros2_textbook_ws
colcon build --packages-select your_example_package

# Source the environment
source install/setup.bash

# Run the example
ros2 run your_example_package example_node
```

### 3. Docusaurus Build Validation
```bash
cd docusaurus
npm run build
```

## Deployment to GitHub Pages

1. Ensure all content is committed and pushed to the repository
2. Run the deployment script:
```bash
cd docusaurus
GIT_USER=<Your GitHub username> USE_SSH=true npm run deploy
```

## Quality Assurance

### Content Quality
- All robotics definitions match official documentation
- Code examples run successfully in specified environment
- Diagrams render correctly and enhance understanding
- Content follows pedagogical progression (Concepts → Diagrams → Code → Simulation)

### Technical Quality
- Docusaurus builds without warnings
- All links and references work correctly
- Page load times are acceptable
- Mobile responsiveness is maintained