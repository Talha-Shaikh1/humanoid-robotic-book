# ROS2 Examples for Humanoid Robotics

This directory contains comprehensive ROS2 examples for humanoid robotics applications, organized by programming language and complexity level.

## Directory Structure

- `python/` - Python-based ROS2 examples using rclpy
- `cpp/` - C++-based ROS2 examples using rclcpp (coming soon)

## ROS2 Distribution Requirements

All examples are designed for **ROS2 Humble Hawksbill** on Ubuntu 22.04 LTS. While they should work with other ROS2 distributions, testing has been performed primarily on Humble.

## Getting Started

1. Install ROS2 Humble on Ubuntu 22.04
2. Create a new workspace or add these examples to an existing workspace
3. Install dependencies as specified in individual README files
4. Build the workspace: `colcon build`
5. Source the workspace: `source install/setup.bash`
6. Run examples as described in individual README files

## Examples Overview

### Python Examples (Recommended)
- Basic ROS2 concepts (nodes, topics, services, actions)
- Sensor integration and perception
- Robot control and navigation
- AI integration with neural networks
- TF2 transformations
- URDF robot models

### C++ Examples (Coming Soon)
- Performance-critical applications
- Low-level control systems
- Real-time constraints

## Validation Status

All examples have been validated to run in the specified environment with documented dependencies and expected outputs.