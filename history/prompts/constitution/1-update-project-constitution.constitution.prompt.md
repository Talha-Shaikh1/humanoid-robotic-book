---
id: 1
title: Update Project Constitution
stage: constitution
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,specification,documentation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
/sp.constitution

Project: AI-Native Humanoid Robotics Textbook + RAG Chatbot System

Core principles:
- Technical accuracy validated against robotics documentation (ROS 2, Gazebo, Isaac, VLA)
- Clarity for students with mixed backgrounds (beginners → advanced robotics)
- Pedagogical progression: Concepts → Diagrams → Code → Simulation → Deployment
- Reproducibility: All code runnable (ROS2, Isaac, Gazebo examples tested)
- Consistency through reusable intelligence (Skills + Subagents)
- Multi-modal explanation: diagrams, mermaid flows, tables, and layered breakdowns
- AI-native writing: Every chapter generated from specs using Spec-Kit Plus + Claude Code

Key standards:
- All robotics definitions must match official docs (ROS 2 REP standards, Gazebo physics)
- Control-loop explanations follow robotics best practices (PID, URDF, TF2)
- All code: runnable, tested, ROS2-foxy/humble compatible
- RAG chatbot answers must reference ONLY book content (strict grounding)
- Personalization logic must adapt chapter difficulty to user profile (from better-auth signup)
- Urdu translation must preserve technical meaning (not literal translation)

Constraints:
- Format: Docusaurus book deployed on GitHub Pages
- Content must align with 4 modules:
  1. ROS 2 — Robotic Nervous System
  2. Gazebo & Unity — Digital Twin
  3. NVIDIA Isaac — AI Robot Brain
  4. VLA — Vision-Language-Action Robotics
- Must support integrated RAG chatbot using:
  - FastAPI backend
  - Qdrant Cloud (Free Tier)
  - Neon Serverless Postgres
  - OpenAI Agents / ChatKit SDK
- Must support personalization, Urdu translation, and chapter-level intelligence features
- Must support extendable Skills/Subagents (50 bonus points system)

Success criteria:
- Full book written using structured specs (spec.md → plan.md → tasks.md → implementation)
- 100% chapters follow the same structure, style, and pedagogy
- RAG chatbot answers accurately based on embedded book text
- Book content customizable (difficulty / personalized paths) per user session
```

## Response snapshot

Constitution file created with the following key elements:

1. Core Principles:
   - Technical Accuracy and Documentation Compliance
   - Pedagogical Progression and Accessibility
   - Reproducibility and Testing Standards
   - Multi-Modal Learning Experience
   - AI-Native Content Generation
   - Strict Information Grounding

2. Key Standards:
   - Technical Standards Compliance
   - Educational Standards

3. Constraints:
   - Platform and Format Requirements
   - Feature Requirements

4. Success Criteria:
   - Content Quality Standards
   - Technical Achievement Metrics

5. Governance section with version control and compliance requirements

## Outcome

- ✅ Impact: Project constitution established with 6 core principles for the AI-Native Humanoid Robotics Textbook system
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive project governance
- 🔁 Next prompts: Proceed with feature specification using /sp.specify
- 🧠 Reflection: Constitution provides clear governance framework for the robotics textbook project

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - Constitution follows template structure and incorporates all specified requirements
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Begin feature specification for the textbook modules