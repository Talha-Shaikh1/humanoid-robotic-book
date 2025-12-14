# Gazebo Simulation Integration in Textbook

This document integrates the Gazebo simulation examples into the educational content of the AI-Native Humanoid Robotics textbook.

## Table of Contents
- [Introduction to Gazebo](#introduction-to-gazebo)
- [Basic Robot Model Simulation](#basic-robot-model-simulation)
- [Sensor Integration](#sensor-integration)
- [Controller Examples](#controller-examples)
- [Mobile Manipulation](#mobile-manipulation)
- [Learning Outcomes](#learning-outcomes)
- [Exercises](#exercises)

## Introduction to Gazebo

Gazebo is a powerful robotics simulator that provides realistic physics simulation, sensor models, and robot models. It's widely used in robotics research and development for testing algorithms before deployment on real robots.

### Key Concepts
- Physics simulation with ODE, Bullet, and DART engines
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Robot model description using URDF/SDF
- Plugin system for custom behaviors
- Integration with ROS/ROS2

## Basic Robot Model Simulation

The basic robot model example demonstrates how to create a simple robot in Gazebo with differential drive and basic sensors.

### URDF Model Structure

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Differential Drive Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="-0.15 0.2 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Gazebo Plugins -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>simple_robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <update_rate>30</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>
</robot>
```

### Key Learning Points
1. **URDF Structure**: Understanding links, joints, and inertial properties
2. **Gazebo Plugins**: Integrating ROS control with Gazebo physics
3. **Differential Drive**: Implementing basic mobile robot kinematics
4. **Namespace Management**: Proper ROS namespace organization

### Basic Controller Implementation

The robot controller demonstrates fundamental control concepts:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class GazeboRobotController(Node):
    def __init__(self):
        super().__init__('gazebo_robot_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_robot/cmd_vel', 10)

        # Create subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/simple_robot/scan',
            self.laser_callback,
            10
        )

        # Control parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.safety_distance = 0.5  # meters

        # Create timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def laser_callback(self, msg):
        # Process laser scan data for obstacle detection
        front_ranges = msg.ranges[:10] + msg.ranges[-10:]
        front_distances = [r for r in front_ranges if 0 < r < msg.range_max]

        if front_distances and min(front_distances) < self.safety_distance:
            # Obstacle detected, turn
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.angular_speed
        else:
            # Clear path, move forward
            cmd_vel = Twist()
            cmd_vel.linear.x = self.linear_speed
            cmd_vel.angular.z = 0.0

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down controller')
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Integration

The sensor integration example demonstrates how to work with multiple sensors simultaneously in Gazebo.

### Multi-Sensor Robot Model

```xml
<!-- Camera Sensor -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>advanced_robot</namespace>
        <remapping>image_raw:=camera/image_raw</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU Sensor -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <visualize>false</visualize>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <!-- Similar for y and z axes -->
      </angular_velocity>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>advanced_robot</namespace>
        <remapping>imu:=imu/data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
    </plugin>
  </sensor>
</gazebo>
```

### Sensor Fusion Implementation

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorIntegrationNode(Node):
    def __init__(self):
        super().__init__('sensor_integration_node')

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Create subscribers for all sensors
        self.image_sub = self.create_subscription(
            Image,
            '/advanced_robot/camera/image_raw',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/advanced_robot/imu/data',
            self.imu_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/advanced_robot/scan',
            self.lidar_callback,
            10
        )

        # Sensor data storage
        self.latest_image = None
        self.latest_imu = None
        self.latest_lidar = None

        # Robot state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Create timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def image_callback(self, msg):
        # Process image for object detection
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            processed_img = self.process_image(self.latest_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, image):
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color detection
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    def imu_callback(self, msg):
        # Extract orientation from quaternion
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        # Convert quaternion to Euler angles (simplified)
        # In practice, use scipy.spatial.transform.Rotation
        import math
        sinr_cosp = 2 * (quaternion[3] * quaternion[0] + quaternion[1] * quaternion[2])
        cosr_cosp = 1 - 2 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        # Similar for pitch and yaw
        sinp = 2 * (quaternion[3] * quaternion[1] - quaternion[2] * quaternion[0])
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.pitch = math.asin(sinp)

        siny_cosp = 2 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1])
        cosy_cosp = 1 - 2 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        # Process LiDAR data for obstacle detection
        ranges = np.array(msg.ranges)
        # Remove invalid ranges (inf or nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().debug(f'LIDAR - Min distance: {min_distance:.2f}m')

    def control_loop(self):
        # Integrate sensor data for navigation
        cmd_vel = Twist()

        # Default behavior: move forward
        cmd_vel.linear.x = 0.5
        cmd_vel.angular.z = 0.0

        # Check LiDAR data for obstacle avoidance
        if self.latest_lidar is not None:
            # Get front-facing ranges (±30 degrees)
            front_ranges = self.latest_lidar.ranges[150:210]
            valid_front_ranges = [r for r in front_ranges if 0 < r < self.latest_lidar.range_max]

            if valid_front_ranges and min(valid_front_ranges) < 0.5:
                # Obstacle detected in front, turn right
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = -0.5

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    sensor_node = SensorIntegrationNode()

    try:
        rclpy.spin(sensor_node)
    except KeyboardInterrupt:
        sensor_node.get_logger().info('Shutting down sensor integration node')
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        sensor_node.cmd_vel_pub.publish(stop_cmd)
        sensor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Controller Examples

The controller examples demonstrate advanced control strategies including path following and PID control.

### Path Following Controller

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math

class PathFollowerController(Node):
    def __init__(self):
        super().__init__('path_follower_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/advanced_robot/cmd_vel', 10)

        # Create subscribers for sensor data
        self.odom_sub = self.create_subscription(
            Odometry,
            '/advanced_robot/odom',
            self.odom_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/advanced_robot/scan',
            self.laser_callback,
            10
        )

        # Robot state
        self.current_pose = None

        # Path following parameters
        self.waypoints = [
            Point(x=2.0, y=1.0, z=0.0),
            Point(x=3.0, y=-1.0, z=0.0),
            Point(x=1.0, y=-2.0, z=0.0),
            Point(x=-1.0, y=-1.0, z=0.0),
            Point(x=-2.0, y=1.0, z=0.0),
            Point(x=0.0, y=0.0, z=0.0)  # Return to start
        ]
        self.current_waypoint_idx = 0
        self.arrival_threshold = 0.3  # meters
        self.lookahead_distance = 0.8  # meters

        # Control parameters
        self.linear_speed = 0.4  # m/s
        self.angular_speed = 0.6  # rad/s

        # Create timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def laser_callback(self, msg):
        self.laser_data = msg

    def get_yaw_from_quaternion(self, quaternion):
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def distance_to_waypoint(self, waypoint):
        if self.current_pose is None:
            return float('inf')

        dx = waypoint.x - self.current_pose.position.x
        dy = waypoint.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def angle_to_waypoint(self, waypoint):
        if self.current_pose is None:
            return 0.0

        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        dx = waypoint.x - self.current_pose.position.x
        dy = waypoint.y - self.current_pose.position.y
        target_angle = math.atan2(dy, dx)

        # Calculate the difference between target and current angle
        angle_diff = target_angle - current_yaw
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff

    def follow_path(self):
        if self.current_pose is None or self.current_waypoint_idx >= len(self.waypoints):
            return Twist()

        cmd_vel = Twist()

        # Get current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_idx]

        # Check if we've reached the current waypoint
        distance = self.distance_to_waypoint(current_waypoint)
        if distance < self.arrival_threshold:
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}')
            self.current_waypoint_idx += 1

            # Check if we've reached the final waypoint
            if self.current_waypoint_idx >= len(self.waypoints):
                self.get_logger().info('Reached final waypoint')
                return Twist()  # Stop the robot

            # Get the next waypoint
            current_waypoint = self.waypoints[self.current_waypoint_idx]

        # Calculate control commands
        angle_to_waypoint = self.angle_to_waypoint(current_waypoint)

        # Adjust linear speed based on angle to waypoint (slower when turning more)
        adjusted_linear_speed = self.linear_speed * max(0.3, math.cos(angle_to_waypoint))

        # Set control commands
        cmd_vel.linear.x = adjusted_linear_speed
        cmd_vel.angular.z = self.angular_speed * angle_to_waypoint

        # Limit angular velocity to prevent excessive turning
        cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, cmd_vel.angular.z))

        return cmd_vel

    def control_loop(self):
        # Execute path following
        cmd_vel = self.follow_path()

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = PathFollowerController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down path follower controller')
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Mobile Manipulation

The mobile manipulation example combines navigation with manipulation capabilities.

### Manipulation Controller

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class MobileManipulationController(Node):
    def __init__(self):
        super().__init__('mobile_manipulation_controller')

        # Create publishers for different control interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, '/mobile_manipulator/cmd_vel', 10)
        self.arm_controller_pub = self.create_publisher(JointTrajectory, '/mobile_manipulator/arm_controller/joint_trajectory', 10)
        self.gripper_controller_pub = self.create_publisher(Float64MultiArray, '/mobile_manipulator/gripper_controller/commands', 10)

        # Create subscribers for sensor data
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mobile_manipulator/odom',
            self.odom_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/mobile_manipulator/joint_states',
            self.joint_state_callback,
            10
        )

        # Robot state
        self.current_pose = None
        self.joint_positions = {}
        self.joint_velocities = {}

        # Control parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.4  # rad/s
        self.arrival_threshold = 0.2  # meters

        # Arm joint names
        self.arm_joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_joint'
        ]

        # Gripper joint
        self.gripper_joint = 'gripper_joint'

        # Create timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)

        # Task parameters
        self.object_position = Point(x=2.0, y=1.0, z=0.0)  # Known object location
        self.target_position = Point(x=-1.0, y=-1.0, z=0.0)  # Target location
        self.gripper_open_position = 0.05  # Gripper fully open
        self.gripper_closed_position = 0.01  # Gripper closed around object

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def get_yaw_from_quaternion(self, quaternion):
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def distance_to_point(self, point):
        if self.current_pose is None:
            return float('inf')

        dx = point.x - self.current_pose.position.x
        dy = point.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def angle_to_point(self, point):
        if self.current_pose is None:
            return 0.0

        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        dx = point.x - self.current_pose.position.x
        dy = point.y - self.current_pose.position.y
        target_angle = math.atan2(dy, dx)

        # Calculate the difference between target and current angle
        angle_diff = target_angle - current_yaw
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff

    def move_arm_to_pose(self, joint_positions):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.arm_joints

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)  # Zero velocity at goal
        point.time_from_start = Duration(sec=2, nanosec=0)  # 2 seconds to reach pose

        trajectory_msg.points.append(point)
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        self.arm_controller_pub.publish(trajectory_msg)

    def control_gripper(self, position):
        command = Float64MultiArray()
        command.data = [position]
        self.gripper_controller_pub.publish(command)

    def navigate_to_point(self, target_point):
        cmd_vel = Twist()

        if self.current_pose is None:
            return cmd_vel

        # Calculate distance and angle to target
        distance = self.distance_to_point(target_point)
        angle_to_target = self.angle_to_point(target_point)

        # If we're close enough to target, stop
        if distance < self.arrival_threshold:
            return cmd_vel  # Stopped

        # Adjust linear speed based on distance (slower when closer)
        adjusted_linear_speed = min(self.linear_speed, distance * 0.5)

        # Set control commands
        cmd_vel.linear.x = adjusted_linear_speed
        cmd_vel.angular.z = self.angular_speed * angle_to_target

        # Limit angular velocity
        cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, cmd_vel.angular.z))

        return cmd_vel

    def control_loop(self):
        # Mobile manipulation task execution
        cmd_vel = Twist()

        # Navigate to object location
        if self.distance_to_point(self.object_position) > 0.5:
            cmd_vel = self.navigate_to_point(self.object_position)
        else:
            # Object approach - stop navigation and prepare for manipulation
            cmd_vel = Twist()

            # Open gripper and position arm for grasping
            self.control_gripper(self.gripper_open_position)

            # Position arm for grasping (example positions)
            grasp_pose = [0.0, -0.5, 1.0, 0.0]  # Shoulder pan, lift, elbow, wrist
            self.move_arm_to_pose(grasp_pose)

    def execute_manipulation_task(self):
        self.get_logger().info('Starting manipulation task...')

def main(args=None):
    rclpy.init(args=args)
    controller = MobileManipulationController()

    # Start the manipulation task
    controller.execute_manipulation_task()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down mobile manipulation controller')
    finally:
        # Stop the robot and open gripper before shutting down
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.control_gripper(controller.gripper_open_position)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Learning Outcomes

After completing the Gazebo simulation examples, students should be able to:

1. **Design Robot Models**: Create URDF files for robots with appropriate links, joints, and inertial properties
2. **Integrate Sensors**: Add cameras, IMU, LIDAR, and other sensors to robot models
3. **Implement Control Systems**: Develop controllers for navigation, manipulation, and other tasks
4. **Process Sensor Data**: Handle data from multiple sensors simultaneously
5. **Execute Complex Tasks**: Combine navigation and manipulation in a single system
6. **Debug Simulations**: Identify and fix issues in simulation environments

## Exercises

1. **Modify Robot Model**: Change the basic robot model to include an additional sensor (e.g., a second camera or an IMU)
2. **Improve Path Following**: Enhance the path follower to handle dynamic obstacles
3. **Add New Behaviors**: Implement a new behavior such as wall following or object exploration
4. **Sensor Fusion**: Combine data from camera and LIDAR for more robust obstacle detection
5. **Manipulation Task**: Extend the mobile manipulator to pick up and place objects of different shapes

## Summary

The Gazebo simulation examples provide a comprehensive introduction to robotics simulation, covering everything from basic robot modeling to complex manipulation tasks. These examples form a solid foundation for students to understand and experiment with robotics concepts in a safe, controlled environment before working with real robots.