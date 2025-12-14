#!/usr/bin/env python3

"""
Gazebo Robot Control Example

This script demonstrates how to control a robot in Gazebo simulation using ROS2.
It publishes velocity commands to move the robot and subscribes to sensor data.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class GazeboRobotController(Node):
    """
    A node that controls a robot in Gazebo simulation.
    """

    def __init__(self):
        """
        Initialize the Gazebo robot controller node.
        """
        super().__init__('gazebo_robot_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_robot/cmd_vel', 10)

        # Create subscribers for sensor data
        self.odom_sub = self.create_subscription(
            Odometry,
            '/simple_robot/odom',
            self.odom_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/simple_robot/camera/image_raw',
            self.image_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/simple_robot/scan',
            self.laser_callback,
            10
        )

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.laser_data = None
        self.latest_image = None

        # Control parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.safety_distance = 0.5  # meters

        # Create timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # State machine for robot behavior
        self.state = 'FORWARD'  # FORWARD, TURN, STOP
        self.state_start_time = self.get_clock().now()

        self.get_logger().info('Gazebo robot controller initialized')

    def odom_callback(self, msg):
        """
        Callback function for odometry messages.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def image_callback(self, msg):
        """
        Callback function for image messages.
        """
        try:
            # Convert ROS Image to OpenCV format
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

    def laser_callback(self, msg):
        """
        Callback function for laser scan messages.
        """
        self.laser_data = msg

    def control_loop(self):
        """
        Main control loop that determines robot behavior based on sensor data.
        """
        cmd_vel = Twist()

        if self.laser_data is not None:
            # Check for obstacles in front of the robot
            front_ranges = self.laser_data.ranges[:10] + self.laser_data.ranges[-10:]
            front_distances = [r for r in front_ranges if 0 < r < self.laser_data.range_max]

            if front_distances and min(front_distances) < self.safety_distance:
                # Obstacle detected, turn
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed
                self.state = 'TURN'
            else:
                # Clear path, move forward
                cmd_vel.linear.x = self.linear_speed
                cmd_vel.angular.z = 0.0
                self.state = 'FORWARD'
        else:
            # Default behavior if no laser data
            cmd_vel.linear.x = self.linear_speed
            cmd_vel.angular.z = 0.0

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        # Log current state
        self.get_logger().info(f'State: {self.state}, Linear: {cmd_vel.linear.x:.2f}, Angular: {cmd_vel.angular.z:.2f}',
                              throttle_duration_sec=1)

    def move_forward(self, distance):
        """
        Move the robot forward by a specified distance.
        """
        if self.current_pose is None:
            return

        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y

        target_x = start_x + distance * math.cos(self.get_yaw_from_quaternion(self.current_pose.orientation))
        target_y = start_y + distance * math.sin(self.get_yaw_from_quaternion(self.current_pose.orientation))

        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear_speed

        while rclpy.ok():
            if self.current_pose is not None:
                current_x = self.current_pose.position.x
                current_y = self.current_pose.position.y
                current_distance = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)

                if current_distance >= distance:
                    cmd_vel.linear.x = 0.0
                    self.cmd_vel_pub.publish(cmd_vel)
                    break

            self.cmd_vel_pub.publish(cmd_vel)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

    def rotate(self, angle):
        """
        Rotate the robot by a specified angle in radians.
        """
        if self.current_pose is None:
            return

        start_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        target_yaw = start_yaw + angle

        # Normalize target yaw to [-pi, pi]
        while target_yaw > math.pi:
            target_yaw -= 2 * math.pi
        while target_yaw < -math.pi:
            target_yaw += 2 * math.pi

        cmd_vel = Twist()
        cmd_vel.angular.z = self.angular_speed if angle > 0 else -self.angular_speed

        while rclpy.ok():
            if self.current_pose is not None:
                current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

                # Calculate the difference between current and target yaw
                angle_diff = target_yaw - current_yaw
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                if abs(angle_diff) < 0.1:  # 0.1 radian tolerance
                    cmd_vel.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd_vel)
                    break

            self.cmd_vel_pub.publish(cmd_vel)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

    def get_yaw_from_quaternion(self, quaternion):
        """
        Extract yaw angle from quaternion.
        """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw


def main(args=None):
    """
    Main function that initializes ROS2, creates the controller node, and starts spinning.
    """
    rclpy.init(args=args)

    controller = GazeboRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Gazebo robot controller')
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()