# ROS2 Python Examples

This directory contains various ROS2 Python examples demonstrating different concepts and capabilities for humanoid robotics applications.

## Requirements

- Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill (or compatible ROS2 distribution)
- Python 3.8 or higher
- Required Python packages (install with pip):
  - rclpy
  - sensor_msgs
  - geometry_msgs
  - std_msgs
  - cv_bridge (for computer vision examples)
  - numpy
  - tensorflow (for AI integration examples)
  - torch torchvision (for PyTorch examples)
  - opencv-python (for computer vision examples)

## How to Run

1. Make sure ROS2 is sourced:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Navigate to the workspace containing these examples and source the workspace:
   ```bash
   cd <workspace_directory>
   source install/setup.bash
   ```

3. Run individual examples:
   ```bash
   python3 <example_name>.py
   ```

## Individual Example Instructions

### Basic Examples
- `basic_node_example.py` - Simple ROS2 node with parameter declaration
- `talker_example.py` - Publisher node example
- `listener_example.py` - Subscriber node example
- `service_client_example.py` and `service_server_example.py` - Service communication
- `action_client_example.py` and `action_server_example.py` - Action communication

### Advanced Examples
- `perception_pipeline_example.py` - Multi-sensor perception with object detection
- `controller_example.py` - Advanced walking controller with ZMP control
- `navigation_example.py` - Path planning and navigation system
- `ai_integration_example.py` - AI integration with TensorFlow and PyTorch
- `tf2_example.py` - Transform library usage
- `urdf_example.py` - URDF model loading and visualization

### Launch Files
- `example_launch.py` - Example launch file demonstrating ROS2 launch system
- `robot_launch_example.py` - Complete robot launch example

## Dependencies

Most examples require the following ROS2 packages:
- `rclpy` - Python client library
- `std_msgs`, `geometry_msgs`, `sensor_msgs` - Common message types
- `tf2_ros` - Transform library
- `cv_bridge` - Computer vision bridge
- `launch` and `launch_ros` - Launch system

## ROS2 Distro Compatibility

These examples are developed and tested with ROS2 Humble Hawksbill. They should be compatible with other ROS2 distributions with minor modifications.

## Expected Output

Each example includes comments explaining the expected behavior and output. Most examples will:
- Initialize ROS2 nodes
- Publish/subscribe to topics
- Process sensor data
- Display information to the console
- Respond to parameters or commands