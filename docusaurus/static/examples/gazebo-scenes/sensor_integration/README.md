# Sensor Integration Example for Gazebo

This example demonstrates how to integrate multiple sensors in a Gazebo simulation, including camera, IMU, and LiDAR data processing.

## Files

- `advanced_robot.urdf`: Robot model with multiple sensors (camera, IMU, LiDAR)
- `sensor_processor.py`: Python script to process and integrate sensor data
- `launch/sensor_integration.launch`: Launch file to start the simulation with all sensors
- `worlds/sensor_world.world`: A world with various obstacles for sensor testing
- `CMakeLists.txt`: Build configuration (if needed for compilation)

## Features

- **Camera**: RGB camera for visual perception
- **IMU**: Inertial measurement unit for orientation and acceleration
- **LiDAR**: 360-degree laser scanner for obstacle detection
- **Differential Drive**: Base for robot movement

## How to Run

1. Make the sensor processor script executable:
   ```bash
   chmod +x sensor_processor.py
   ```

2. Launch the simulation:
   ```bash
   # In one terminal, start Gazebo with the robot
   ros2 launch sensor_integration sensor_integration.launch
   ```

## Sensor Data Processing

The `sensor_processor.py` script demonstrates:

1. **Camera Processing**: Color-based object detection
2. **IMU Processing**: Orientation estimation from quaternion
3. **LiDAR Processing**: Obstacle detection and distance calculation
4. **Sensor Fusion**: Combining data from multiple sensors for navigation

## Topics

The robot publishes and subscribes to the following ROS2 topics:

- `/advanced_robot/cmd_vel` (Publish): Velocity commands
- `/advanced_robot/odom` (Subscribe): Odometry data
- `/advanced_robot/camera/image_raw` (Subscribe): Camera image data
- `/advanced_robot/imu/data` (Subscribe): IMU sensor data
- `/advanced_robot/scan` (Subscribe): LiDAR scan data

## Customization

You can modify the robot model by editing the URDF file to add more sensors or change physical properties. The sensor processor script can be extended to implement more complex sensor fusion algorithms.