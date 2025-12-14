#!/usr/bin/env python3

"""
Advanced Path Following Controller for Gazebo

This script demonstrates an advanced path following controller that uses sensor
data to navigate through waypoints while avoiding obstacles.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
from collections import deque


class PathFollowerController(Node):
    """
    Advanced path following controller with obstacle avoidance.
    """

    def __init__(self):
        """
        Initialize the path follower controller.
        """
        super().__init__('path_follower_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/advanced_robot/cmd_vel', 10)

        # Create publisher for visualization markers
        self.path_marker_pub = self.create_publisher(Marker, 'path_marker', 10)
        self.robot_path_pub = self.create_publisher(Path, 'robot_path', 10)

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

        self.image_sub = self.create_subscription(
            Image,
            '/advanced_robot/camera/image_raw',
            self.image_callback,
            10
        )

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # TF2 buffer and listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.laser_data = None
        self.latest_image = None

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
        self.max_linear_speed = 0.8
        self.max_angular_speed = 1.0
        self.safety_distance = 0.6  # meters

        # Obstacle avoidance parameters
        self.obstacle_detected = False
        self.avoidance_active = False
        self.avoidance_start_time = None
        self.avoidance_duration = 3.0  # seconds

        # Path storage
        self.robot_path = Path()
        self.robot_path.header.frame_id = 'map'

        # Create timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # State machine for robot behavior
        self.state = 'FOLLOW_PATH'  # FOLLOW_PATH, AVOID_OBSTACLE, RECOVERY
        self.state_start_time = self.get_clock().now()

        self.get_logger().info('Path follower controller initialized')

    def odom_callback(self, msg):
        """
        Callback function for odometry messages.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        # Add current position to path for visualization
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.robot_path.poses.append(pose_stamped)

        # Limit path length to prevent memory issues
        if len(self.robot_path.poses) > 1000:
            self.robot_path.poses.pop(0)

    def laser_callback(self, msg):
        """
        Callback function for laser scan messages.
        """
        self.laser_data = msg

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

    def distance_to_waypoint(self, waypoint):
        """
        Calculate distance from current position to waypoint.
        """
        if self.current_pose is None:
            return float('inf')

        dx = waypoint.x - self.current_pose.position.x
        dy = waypoint.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def angle_to_waypoint(self, waypoint):
        """
        Calculate angle from current orientation to waypoint.
        """
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

    def check_obstacles_ahead(self):
        """
        Check for obstacles in the robot's path using laser data.
        """
        if self.laser_data is None:
            return False

        # Check laser readings in front of the robot (±30 degrees)
        total_readings = len(self.laser_data.ranges)
        front_start = int(total_readings * 0.45)  # Slightly left of center
        front_end = int(total_readings * 0.55)    # Slightly right of center

        front_ranges = self.laser_data.ranges[front_start:front_end]
        valid_ranges = [r for r in front_ranges if 0 < r < self.laser_data.range_max]

        if valid_ranges and min(valid_ranges) < self.safety_distance:
            return True
        return False

    def follow_path(self):
        """
        Calculate control commands to follow the path.
        """
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
        cmd_vel.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd_vel.angular.z))

        return cmd_vel

    def avoid_obstacles(self):
        """
        Calculate control commands to avoid obstacles.
        """
        cmd_vel = Twist()

        # Simple obstacle avoidance: turn away from obstacles
        if self.laser_data is not None:
            # Get left and right side ranges
            total_readings = len(self.laser_data.ranges)
            left_ranges = self.laser_data.ranges[:total_readings//4]  # Left 90 degrees
            right_ranges = self.laser_data.ranges[3*total_readings//4:]  # Right 90 degrees

            left_distances = [r for r in left_ranges if 0 < r < self.laser_data.range_max]
            right_distances = [r for r in right_ranges if 0 < r < self.laser_data.range_max]

            left_clear = len(left_distances) == 0 or min(left_distances) > self.safety_distance
            right_clear = len(right_distances) == 0 or min(right_distances) > self.safety_distance

            if left_clear and right_clear:
                # Both sides clear, go forward
                cmd_vel.linear.x = self.linear_speed
                cmd_vel.angular.z = 0.0
            elif left_clear:
                # Left side clear, turn left
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed
            elif right_clear:
                # Right side clear, turn right
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = -self.angular_speed
            else:
                # Neither side clear, turn randomly
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = self.angular_speed  # Default to turning left

        return cmd_vel

    def control_loop(self):
        """
        Main control loop that manages path following and obstacle avoidance.
        """
        cmd_vel = Twist()

        # Check for obstacles
        obstacles_ahead = self.check_obstacles_ahead()

        if obstacles_ahead and not self.avoidance_active:
            # Start obstacle avoidance
            self.get_logger().info('Obstacle detected, switching to avoidance mode')
            self.avoidance_active = True
            self.avoidance_start_time = self.get_clock().now()
            self.state = 'AVOID_OBSTACLE'
        elif self.avoidance_active:
            # Check if avoidance period is over
            if (self.get_clock().now() - self.avoidance_start_time).nanoseconds / 1e9 > self.avoidance_duration:
                self.avoidance_active = False
                self.state = 'FOLLOW_PATH'
                self.get_logger().info('Resuming path following')

        # Execute appropriate behavior based on state
        if self.state == 'AVOID_OBSTACLE':
            cmd_vel = self.avoid_obstacles()
        else:
            cmd_vel = self.follow_path()

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish robot path for visualization
        self.robot_path.header.stamp = self.get_clock().now().to_msg()
        self.robot_path_pub.publish(self.robot_path)

        # Log current state and position
        if self.current_pose is not None:
            self.get_logger().info(
                f'State: {self.state}, '
                f'Waypoint: {self.current_waypoint_idx}/{len(self.waypoints)}, '
                f'Pos: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), '
                f'Cmd: ({cmd_vel.linear.x:.2f}, {cmd_vel.angular.z:.2f})',
                throttle_duration_sec=2
            )

    def visualize_path(self):
        """
        Publish visualization markers for the path.
        """
        if len(self.waypoints) == 0:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Set the scale of the marker
        marker.scale.x = 0.05  # Line width

        # Set the color (Blue)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Add points to the line strip
        for wp in self.waypoints:
            marker.points.append(wp)

        # Publish the marker
        self.path_marker_pub.publish(marker)


def main(args=None):
    """
    Main function that initializes ROS2, creates the controller node, and starts spinning.
    """
    rclpy.init(args=args)

    controller = PathFollowerController()

    # Visualize the path at startup
    controller.visualize_path()

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