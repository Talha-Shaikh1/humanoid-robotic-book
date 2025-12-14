# Controller Examples for Gazebo

This example demonstrates various control strategies for robots in Gazebo simulation, including path following and PID control.

## Files

- `path_follower_controller.py`: Advanced path following controller with obstacle avoidance
- `pid_controller.py`: PID controller for precise robot positioning
- `urdf/controller_robot.urdf`: Simple robot model for controller testing
- `launch/controller_examples.launch`: Launch file to start the simulation
- `worlds/controller_world.world`: A world with boundaries and obstacles for testing
- `CMakeLists.txt`: Build configuration (if needed for compilation)

## Features

- **Path Following Controller**: Follows a predefined path with obstacle avoidance
- **PID Controller**: Precise positioning using PID control
- **Obstacle Detection**: Uses laser scanner to detect and avoid obstacles
- **Visualization**: Shows robot path and target positions

## How to Run

1. Make the controller scripts executable:
   ```bash
   chmod +x path_follower_controller.py
   chmod +x pid_controller.py
   ```

2. Launch the simulation with path follower controller:
   ```bash
   # In one terminal, start Gazebo with the robot
   ros2 launch controller_examples controller_examples.launch controller_type:=path_follower
   ```

3. Or launch with PID controller:
   ```bash
   ros2 launch controller_examples controller_examples.launch controller_type:=pid
   ```

## Control Strategies

### Path Following Controller
- Follows a predefined sequence of waypoints
- Implements obstacle avoidance when obstacles are detected
- Uses a lookahead distance for smooth path following
- Visualizes the robot's path in RViz

### PID Controller
- Uses PID control for precise positioning
- Moves to specific coordinates with orientation control
- Executes a predefined trajectory (square path)
- Demonstrates both position and orientation control

## Topics

The controllers publish and subscribe to the following ROS2 topics:

- `/controller_robot/cmd_vel` (Publish): Velocity commands
- `/controller_robot/odom` (Subscribe): Odometry data
- `/controller_robot/scan` (Subscribe): Laser scan data
- `/controller_robot/camera/image_raw` (Subscribe): Camera image data (path follower only)
- `/robot_path` (Publish): Robot trajectory for visualization (path follower only)

## Customization

You can modify the controllers by adjusting the PID gains, path waypoints, or control parameters. The world file can be modified to create different obstacle configurations for testing.