#!/usr/bin/env python3

"""
Mobile Manipulation Controller for Gazebo

This script demonstrates mobile manipulation capabilities with a robot that has
both navigation and manipulation abilities in Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Pose, Vector3
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time


class MobileManipulationController(Node):
    """
    Controller for a mobile manipulator robot with both navigation and manipulation capabilities.
    """

    def __init__(self):
        """
        Initialize the mobile manipulation controller.
        """
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

        self.image_sub = self.create_subscription(
            Image,
            '/mobile_manipulator/camera/image_raw',
            self.image_callback,
            10
        )

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.joint_positions = {}
        self.joint_velocities = {}
        self.latest_image = None

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

        # State machine for robot behavior
        self.state = 'NAVIGATE_TO_OBJECT'  # NAVIGATE_TO_OBJECT, APPROACH_OBJECT, GRASP_OBJECT, NAVIGATE_TO_TARGET, DEPOSIT_OBJECT
        self.state_start_time = self.get_clock().now()

        # Task parameters
        self.object_position = Point(x=2.0, y=1.0, z=0.0)  # Known object location
        self.target_position = Point(x=-1.0, y=-1.0, z=0.0)  # Target location
        self.object_approach_distance = 0.5  # Distance to approach object
        self.gripper_open_position = 0.05  # Gripper fully open
        self.gripper_closed_position = 0.01  # Gripper closed around object

        self.get_logger().info('Mobile manipulation controller initialized')

    def odom_callback(self, msg):
        """
        Callback function for odometry messages.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def joint_state_callback(self, msg):
        """
        Callback function for joint state messages.
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def image_callback(self, msg):
        """
        Callback function for image messages.
        """
        try:
            # Convert ROS Image to OpenCV format
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

    def get_yaw_from_quaternion(self, quaternion):
        """
        Extract yaw angle from quaternion.
        """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def distance_to_point(self, point):
        """
        Calculate distance from current position to a point.
        """
        if self.current_pose is None:
            return float('inf')

        dx = point.x - self.current_pose.position.x
        dy = point.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def angle_to_point(self, point):
        """
        Calculate angle from current orientation to a point.
        """
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
        """
        Move the robot arm to specified joint positions.
        """
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
        """
        Control the gripper position.
        """
        command = Float64MultiArray()
        command.data = [position]
        self.gripper_controller_pub.publish(command)

    def navigate_to_point(self, target_point):
        """
        Generate velocity commands to navigate to a target point.
        """
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
        """
        Main control loop that manages the manipulation task.
        """
        cmd_vel = Twist()

        if self.state == 'NAVIGATE_TO_OBJECT':
            # Navigate to the general area of the object
            distance_to_object = self.distance_to_point(self.object_position)

            if distance_to_object > self.object_approach_distance:
                # Still far from object, navigate toward it
                cmd_vel = self.navigate_to_point(self.object_position)
            else:
                # Close enough to object, switch to approach state
                self.state = 'APPROACH_OBJECT'
                self.get_logger().info('Switching to APPROACH_OBJECT state')

        elif self.state == 'APPROACH_OBJECT':
            # Approach the object more precisely
            distance_to_object = self.distance_to_point(self.object_position)

            if distance_to_object > 0.3:  # Very close to object
                cmd_vel = self.navigate_to_point(self.object_position)
            else:
                # Close enough to object, open gripper and position arm
                cmd_vel = Twist()  # Stop the base
                self.control_gripper(self.gripper_open_position)

                # Position arm for grasping
                grasp_pose = [0.0, -0.5, 1.0, 0.0]  # Shoulder pan, lift, elbow, wrist
                self.move_arm_to_pose(grasp_pose)

                # Wait for arm to reach position (in a real system, we'd check joint states)
                time.sleep(2)

                self.state = 'GRASP_OBJECT'
                self.get_logger().info('Switching to GRASP_OBJECT state')

        elif self.state == 'GRASP_OBJECT':
            # Close gripper to grasp the object
            self.control_gripper(self.gripper_closed_position)
            time.sleep(1)  # Wait for grasp

            # Lift the object slightly
            current_arm_pos = [
                self.joint_positions.get('shoulder_pan_joint', 0.0),
                self.joint_positions.get('shoulder_lift_joint', 0.0),
                self.joint_positions.get('elbow_joint', 0.0),
                self.joint_positions.get('wrist_joint', 0.0)
            ]
            lift_pose = [current_arm_pos[0], current_arm_pos[1] - 0.2, current_arm_pos[2], current_arm_pos[3]]
            self.move_arm_to_pose(lift_pose)

            time.sleep(1)  # Wait for lift

            self.state = 'NAVIGATE_TO_TARGET'
            self.get_logger().info('Object grasped, switching to NAVIGATE_TO_TARGET state')

        elif self.state == 'NAVIGATE_TO_TARGET':
            # Navigate to the target location
            distance_to_target = self.distance_to_point(self.target_position)

            if distance_to_target > self.arrival_threshold:
                # Navigate toward target
                cmd_vel = self.navigate_to_point(self.target_position)
            else:
                # Reached target, switch to deposit state
                cmd_vel = Twist()  # Stop the base
                self.state = 'DEPOSIT_OBJECT'
                self.get_logger().info('Reached target, switching to DEPOSIT_OBJECT state')

        elif self.state == 'DEPOSIT_OBJECT':
            # Position arm for depositing and open gripper
            deposit_pose = [0.0, 0.0, 0.0, 0.0]  # Return to home position
            self.move_arm_to_pose(deposit_pose)

            time.sleep(2)  # Wait for arm to position

            # Open gripper to release object
            self.control_gripper(self.gripper_open_position)

            time.sleep(0.5)  # Wait for release

            # Move arm back up
            final_pose = [0.0, -0.5, 1.0, 0.0]
            self.move_arm_to_pose(final_pose)

            self.state = 'TASK_COMPLETE'
            self.get_logger().info('Object deposited, task complete!')

        # Publish velocity command for base movement
        self.cmd_vel_pub.publish(cmd_vel)

        # Log current state and position
        if self.current_pose is not None:
            self.get_logger().info(
                f'State: {self.state}, '
                f'Pos: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), '
                f'Cmd: ({cmd_vel.linear.x:.2f}, {cmd_vel.angular.z:.2f})',
                throttle_duration_sec=1
            )

    def execute_manipulation_task(self):
        """
        Execute the complete manipulation task.
        """
        self.get_logger().info('Starting manipulation task...')
        self.state = 'NAVIGATE_TO_OBJECT'


def main(args=None):
    """
    Main function that initializes ROS2, creates the controller node, and starts spinning.
    """
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