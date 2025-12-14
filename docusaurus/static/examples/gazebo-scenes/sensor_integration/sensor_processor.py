#!/usr/bin/env python3

"""
Advanced Sensor Integration Example

This script demonstrates how to integrate multiple sensors in a Gazebo simulation,
including camera, IMU, and LiDAR data processing.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image, Imu, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


class SensorIntegrationNode(Node):
    """
    A node that integrates data from multiple sensors in Gazebo simulation.
    """

    def __init__(self):
        """
        Initialize the sensor integration node.
        """
        super().__init__('sensor_integration_node')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/advanced_robot/cmd_vel', 10)

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

        self.odom_sub = self.create_subscription(
            Odometry,
            '/advanced_robot/odom',
            self.odom_callback,
            10
        )

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Sensor data storage
        self.latest_image = None
        self.latest_imu = None
        self.latest_lidar = None
        self.latest_odom = None

        # Robot state
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Control parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.safety_distance = 0.5  # meters

        # Create timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # State machine for robot behavior
        self.state = 'EXPLORATION'  # EXPLORATION, AVOIDANCE, STATIONARY
        self.state_start_time = self.get_clock().now()

        self.get_logger().info('Sensor integration node initialized')

    def image_callback(self, msg):
        """
        Callback function for image messages.
        """
        try:
            # Convert ROS Image to OpenCV format
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Process image for object detection (simple color-based detection)
            processed_img = self.process_image(self.latest_image)

            # Publish processed image information
            self.get_logger().debug(f'Processed image: {processed_img.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, image):
        """
        Process the image to detect objects or features.
        """
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

        # If we found any contours, log the detection
        if len(contours) > 0:
            self.get_logger().info(f'Detected {len(contours)} red objects')

        return result

    def imu_callback(self, msg):
        """
        Callback function for IMU messages.
        """
        self.latest_imu = msg

        # Extract orientation from quaternion
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        r = R.from_quat(quaternion)
        self.roll, self.pitch, self.yaw = r.as_euler('xyz')

        # Extract angular velocity and linear acceleration
        self.angular_velocity_x = msg.angular_velocity.x
        self.angular_velocity_y = msg.angular_velocity.y
        self.angular_velocity_z = msg.angular_velocity.z

        self.linear_acceleration_x = msg.linear_acceleration.x
        self.linear_acceleration_y = msg.linear_acceleration.y
        self.linear_acceleration_z = msg.linear_acceleration.z

        # Log IMU data periodically
        self.get_logger().debug(f'IMU - Roll: {self.roll:.2f}, Pitch: {self.pitch:.2f}, Yaw: {self.yaw:.2f}',
                               throttle_duration_sec=2)

    def lidar_callback(self, msg):
        """
        Callback function for LiDAR messages.
        """
        self.latest_lidar = msg

        # Process LiDAR data for obstacle detection
        ranges = np.array(msg.ranges)
        # Remove invalid ranges (inf or nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().debug(f'LIDAR - Min distance: {min_distance:.2f}m',
                                   throttle_duration_sec=1)

    def odom_callback(self, msg):
        """
        Callback function for odometry messages.
        """
        self.latest_odom = msg

        # Extract linear and angular velocities
        self.linear_velocity = msg.twist.twist.linear.x
        self.angular_velocity = msg.twist.twist.angular.z

    def control_loop(self):
        """
        Main control loop that integrates sensor data for navigation.
        """
        cmd_vel = Twist()

        # Default behavior: move forward
        cmd_vel.linear.x = self.linear_speed
        cmd_vel.angular.z = 0.0

        # Check LiDAR data for obstacle avoidance
        if self.latest_lidar is not None:
            # Get front-facing ranges (±30 degrees)
            front_ranges = self.latest_lidar.ranges[150:210]  # Approximate front 60 degrees
            valid_front_ranges = [r for r in front_ranges if 0 < r < self.latest_lidar.range_max]

            if valid_front_ranges and min(valid_front_ranges) < self.safety_distance:
                # Obstacle detected in front, switch to avoidance behavior
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed  # Turn right
                self.state = 'AVOIDANCE'
            else:
                # No obstacles ahead, continue exploration
                self.state = 'EXPLORATION'
        else:
            # If no LiDAR data, continue with default behavior
            self.state = 'EXPLORATION'

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        # Log current state and sensor data
        if self.latest_lidar is not None:
            front_ranges = self.latest_lidar.ranges[150:210]
            valid_ranges = [r for r in front_ranges if 0 < r < self.latest_lidar.range_max]
            min_dist = min(valid_ranges) if valid_ranges else float('inf')

            self.get_logger().info(
                f'State: {self.state}, '
                f'Linear: {cmd_vel.linear.x:.2f}, '
                f'Angular: {cmd_vel.angular.z:.2f}, '
                f'Min front dist: {min_dist:.2f}m',
                throttle_duration_sec=1
            )

    def get_robot_orientation(self):
        """
        Get the current orientation of the robot.
        """
        if self.latest_imu is not None:
            return self.roll, self.pitch, self.yaw
        else:
            return 0.0, 0.0, 0.0

    def get_robot_velocity(self):
        """
        Get the current velocity of the robot.
        """
        return self.linear_velocity, self.angular_velocity

    def get_obstacle_distances(self):
        """
        Get distances to obstacles from LiDAR data.
        """
        if self.latest_lidar is not None:
            ranges = self.latest_lidar.ranges
            angles = [self.latest_lidar.angle_min + i * self.latest_lidar.angle_increment
                     for i in range(len(ranges))]
            return ranges, angles
        else:
            return [], []


def main(args=None):
    """
    Main function that initializes ROS2, creates the sensor integration node, and starts spinning.
    """
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