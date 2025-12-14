# Mobile Manipulation Example for Gazebo

This example demonstrates mobile manipulation capabilities with a robot that has both navigation and manipulation abilities in Gazebo simulation.

## Files

- `mobile_manipulator.urdf`: Robot model with differential drive base and 4-DOF manipulator arm
- `manipulation_controller.py`: Controller for mobile manipulation task
- `launch/mobile_manipulation.launch`: Launch file to start the simulation
- `worlds/manipulation_world.world`: World with objects for manipulation
- `CMakeLists.txt`: Build configuration (if needed for compilation)

## Features

- **Differential Drive Base**: Mobile platform for navigation
- **4-DOF Manipulator Arm**: Arm with shoulder, elbow, and wrist joints
- **Parallel Gripper**: 1-DOF gripper for object manipulation
- **RGB Camera**: For visual perception
- **Complete Manipulation Task**: Navigate, grasp, transport, and deposit objects

## How to Run

1. Make the controller script executable:
   ```bash
   chmod +x manipulation_controller.py
   ```

2. Launch the simulation:
   ```bash
   # In one terminal, start Gazebo with the robot
   ros2 launch mobile_manipulation mobile_manipulation.launch
   ```

## Manipulation Task

The controller executes the following sequence:

1. **Navigate to Object**: Move the robot base to the vicinity of the target object
2. **Approach Object**: Position the robot base close to the object
3. **Grasp Object**: Position the arm and close the gripper to grasp the object
4. **Navigate to Target**: Move the robot to the target location
5. **Deposit Object**: Open the gripper to release the object at the target

## Robot Joints

The robot has the following controllable joints:

- `shoulder_pan_joint`: Rotates the shoulder left/right
- `shoulder_lift_joint`: Lifts/drops the shoulder
- `elbow_joint`: Moves the elbow up/down
- `wrist_joint`: Rotates the wrist
- `gripper_joint`: Controls gripper opening/closing

## Topics

The controller publishes and subscribes to the following ROS2 topics:

- `/mobile_manipulator/cmd_vel` (Publish): Base velocity commands
- `/mobile_manipulator/arm_controller/joint_trajectory` (Publish): Arm trajectory commands
- `/mobile_manipulator/gripper_controller/commands` (Publish): Gripper position commands
- `/mobile_manipulator/odom` (Subscribe): Odometry data
- `/mobile_manipulator/joint_states` (Subscribe): Joint position/velocity data
- `/mobile_manipulator/camera/image_raw` (Subscribe): Camera image data

## Customization

You can modify the manipulation task by changing the object and target positions in the controller code. The robot model can be extended with more complex kinematics or additional sensors.