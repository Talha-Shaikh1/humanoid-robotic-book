# Basic Robot Model for Gazebo Simulation

This example demonstrates a simple robot model that can be simulated in Gazebo with ROS2 integration.

## Files

- `simple_robot.urdf`: The robot description file with differential drive and camera sensor
- `launch/simple_robot.launch`: Launch file to start the robot in Gazebo
- `worlds/simple_world.world`: A simple world with obstacles for testing
- `scripts/robot_controller.py`: Python script to control the robot and process sensor data

## How to Run

1. Make the controller script executable:
   ```bash
   chmod +x scripts/robot_controller.py
   ```

2. Launch the simulation:
   ```bash
   # In one terminal, start Gazebo with the robot
   ros2 launch simple_robot_description simple_robot.launch
   ```

3. In another terminal, run the controller:
   ```bash
   ros2 run simple_robot_description robot_controller.py
   ```

## Features

- Differential drive robot with wheel joints
- RGB camera sensor mounted on front
- Basic obstacle avoidance behavior
- ROS2 integration for control and sensing

## Topics

The robot publishes and subscribes to the following ROS2 topics:

- `/simple_robot/cmd_vel` (Publish): Velocity commands for the robot
- `/simple_robot/odom` (Subscribe): Odometry data
- `/simple_robot/camera/image_raw` (Subscribe): Camera image data
- `/simple_robot/scan` (Subscribe): Laser scan data (if available)

## Customization

You can modify the robot model by editing the URDF file to add more joints, sensors, or change physical properties. The controller script can be extended to implement more complex behaviors.