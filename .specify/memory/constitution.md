<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Modified principles: Added 6 specific principles for robotics textbook project
- Added sections: Core Principles, Key Standards, Constraints, Success Criteria
- Removed sections: None
- Templates requiring updates: N/A
- Follow-up TODOs: None
-->
# AI-Native Humanoid Robotics Textbook + RAG Chatbot System Constitution

## Core Principles

### Technical Accuracy and Documentation Compliance
All robotics definitions, concepts, and implementations must strictly match official documentation standards (ROS 2 REP standards, Gazebo physics, NVIDIA Isaac guidelines, VLA specifications). Every technical statement must be verifiable against authoritative sources to ensure educational integrity and practical applicability.

### Pedagogical Progression and Accessibility
Content must follow a structured learning path: Concepts → Diagrams → Code → Simulation → Deployment. All materials must accommodate students with mixed backgrounds ranging from beginners to advanced robotics practitioners, with clear pathways for different skill levels.

### Reproducibility and Testing Standards
All code examples, simulations, and demonstrations must be runnable and tested in controlled environments. All robotics code must be compatible with ROS2 Foxy/Humble distributions, and all examples must be validated in simulation environments (Gazebo, Isaac) before inclusion.

### Multi-Modal Learning Experience
Educational content must leverage multiple presentation formats including diagrams, mermaid flows, tables, and layered breakdowns to accommodate different learning styles. Visual representations must enhance understanding of complex robotics concepts.

### AI-Native Content Generation
Every chapter and educational module must be systematically generated from structured specifications using Spec-Kit Plus and Claude Code. Content creation follows the spec.md → plan.md → tasks.md → implementation pipeline for consistency and quality control.

### Strict Information Grounding
The RAG chatbot must reference ONLY book content for answers, maintaining strict grounding to prevent hallucination. All responses must be traceable to specific sections of the textbook content to ensure educational accuracy.

## Key Standards

### Technical Standards Compliance
- All robotics definitions must match official documentation (ROS 2 REP standards, Gazebo physics)
- Control-loop explanations follow robotics best practices (PID, URDF, TF2)
- All code must be runnable, tested, and ROS2-foxy/humble compatible
- RAG chatbot answers must reference ONLY book content (strict grounding)

### Educational Standards
- Personalization logic must adapt chapter difficulty to user profile (from better-auth signup)
- Urdu translation must preserve technical meaning (not literal translation)
- Content must align with 4 core modules: ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA
- All examples must be reproducible in simulation environments

## Constraints

### Platform and Format Requirements
- Format: Docusaurus book deployed on GitHub Pages
- Content must align with 4 modules: ROS 2 — Robotic Nervous System, Gazebo & Unity — Digital Twin, NVIDIA Isaac — AI Robot Brain, VLA — Vision-Language-Action Robotics
- Must support integrated RAG chatbot using: FastAPI backend, Qdrant Cloud (Free Tier), Neon Serverless Postgres, OpenAI Agents / ChatKit SDK

### Feature Requirements
- Must support personalization, Urdu translation, and chapter-level intelligence features
- Must support extendable Skills/Subagents (50 bonus points system)
- All content must be AI-native generated following structured specifications
- Deployment must support GitHub Pages hosting

## Success Criteria

### Content Quality Standards
- Full book written using structured specs (spec.md → plan.md → tasks.md → implementation)
- 100% chapters follow the same structure, style, and pedagogy
- RAG chatbot answers accurately based on embedded book text
- Book content customizable (difficulty / personalized paths) per user session

### Technical Achievement Metrics
- All code examples run successfully in specified ROS2 environments
- RAG system maintains strict grounding to book content
- Personalization adapts appropriately to user profiles
- Multi-language support preserves technical accuracy

## Governance

All development and content creation activities must comply with these constitutional principles. Changes to core principles require explicit documentation of the rationale and impact assessment. Educational content must undergo technical validation against official documentation before acceptance. Version control follows semantic versioning with major updates reflecting significant pedagogical or technical shifts. Quality assurance includes both technical accuracy verification and educational effectiveness validation.

**Version**: 1.1.0 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-10