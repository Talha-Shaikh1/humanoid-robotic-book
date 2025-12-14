#!/usr/bin/env python3

"""
ROS2 Navigation Example

This example demonstrates how to implement navigation capabilities in ROS2,
including path planning, obstacle avoidance, and robot movement control.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
import math
from collections import deque


class SimpleNavigation(Node):
    """
    A simple navigation node that demonstrates path planning and execution.
    """

    def __init__(self):
        """
        Initialize the navigation node.
        """
        super().__init__('simple_navigation')

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            sensor_qos
        )

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'current_path', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'current_goal', 10)

        # TF2 buffer and listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state variables
        self.current_pose = None
        self.current_twist = None
        self.laser_ranges = None
        self.current_goal = None
        self.path_to_goal = []
        self.waypoints = deque()
        self.navigation_active = False
        self.arrived_at_goal = False

        # Navigation parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.arrival_threshold = 0.2  # meters
        self.rotation_threshold = 0.1  # radians
        self.min_obstacle_distance = 0.5  # meters
        self.control_frequency = 10  # Hz

        # Robot dimensions for obstacle avoidance
        self.robot_radius = 0.3  # meters

        # Create timer for navigation control
        self.nav_timer = self.create_timer(1.0 / self.control_frequency, self.navigation_control)

        # Store robot path for visualization
        self.robot_path = Path()
        self.robot_path.header.frame_id = 'map'

        self.get_logger().info('Simple navigation node initialized')

    def odom_callback(self, msg):
        """
        Callback function for odometry messages.
        Updates the robot's current pose and twist.
        """
        # Update current pose
        self.current_pose = msg.pose.pose

        # Update current twist (velocity)
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
        Stores laser ranges for obstacle detection.
        """
        self.laser_ranges = msg.ranges
        self.laser_angle_min = msg.angle_min
        self.laser_angle_max = msg.angle_max
        self.laser_angle_increment = msg.angle_increment

    def set_goal(self, x, y, theta=0.0):
        """
        Set a new navigation goal.
        """
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0

        # Convert theta to quaternion
        goal_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_pose.pose.orientation.w = math.cos(theta / 2.0)

        self.current_goal = goal_pose
        self.navigation_active = True
        self.arrived_at_goal = False

        # Plan path to goal (simplified - in reality would use A*, Dijkstra, etc.)
        self.plan_path_to_goal()

        # Publish goal for visualization
        self.goal_pub.publish(self.current_goal)

        self.get_logger().info(f'Set navigation goal to ({x}, {y})')

    def plan_path_to_goal(self):
        """
        Plan a simple path to the goal (straight line in this example).
        In a real implementation, this would use path planning algorithms like A* or Dijkstra.
        """
        if self.current_pose is None or self.current_goal is None:
            return

        # Simple straight-line path
        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y
        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y

        # Calculate intermediate waypoints (every 0.1m)
        distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        num_waypoints = int(distance / 0.1)

        self.waypoints.clear()
        for i in range(1, num_waypoints + 1):
            ratio = i / num_waypoints
            wp_x = start_x + ratio * (goal_x - start_x)
            wp_y = start_y + ratio * (goal_y - start_y)

            waypoint = Point()
            waypoint.x = wp_x
            waypoint.y = wp_y
            waypoint.z = 0.0

            self.waypoints.append(waypoint)

    def navigation_control(self):
        """
        Main navigation control loop that runs at the specified frequency.
        """
        if not self.navigation_active or self.current_pose is None:
            return

        # Check if we have a current goal
        if self.current_goal is None:
            self.navigation_active = False
            return

        # Check if we have laser data for obstacle avoidance
        if self.laser_ranges is not None:
            # Check for obstacles in the path
            if self.detect_obstacles_ahead():
                self.get_logger().warn('Obstacle detected ahead, stopping navigation')
                self.stop_robot()
                return

        # Get next waypoint
        if self.waypoints:
            next_waypoint = self.waypoints[0]
        else:
            # If no waypoints, go directly to goal
            next_waypoint = Point()
            next_waypoint.x = self.current_goal.pose.position.x
            next_waypoint.y = self.current_goal.pose.position.y
            next_waypoint.z = 0.0

        # Calculate distance to next waypoint
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        dist_to_waypoint = math.sqrt(
            (next_waypoint.x - current_x)**2 + (next_waypoint.y - current_y)**2
        )

        # If close to waypoint, remove it and get the next one
        if dist_to_waypoint < self.arrival_threshold and self.waypoints:
            self.waypoints.popleft()

        # Calculate control commands
        cmd_vel = self.calculate_navigation_command(next_waypoint)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Check if arrived at goal
        goal_dist = math.sqrt(
            (self.current_goal.pose.position.x - current_x)**2 +
            (self.current_goal.pose.position.y - current_y)**2
        )

        if goal_dist < self.arrival_threshold:
            self.arrived_at_goal = True
            self.navigation_active = False
            self.stop_robot()
            self.get_logger().info('Arrived at goal!')
        else:
            # Publish current path for visualization
            self.robot_path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.robot_path)

    def calculate_navigation_command(self, target_point):
        """
        Calculate navigation command to move towards target point.
        """
        cmd = Twist()

        if self.current_pose is None:
            return cmd

        # Get current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        # Calculate desired angle to target
        desired_angle = math.atan2(
            target_point.y - current_y,
            target_point.x - current_x
        )

        # Get current orientation
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Calculate angle difference
        angle_diff = desired_angle - current_yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Calculate distance to target
        distance = math.sqrt(
            (target_point.x - current_x)**2 + (target_point.y - current_y)**2
        )

        # Control logic
        if abs(angle_diff) > self.rotation_threshold:
            # Rotate towards target
            cmd.angular.z = self.angular_speed * np.sign(angle_diff)
        elif distance > self.arrival_threshold:
            # Move forward toward target
            cmd.linear.x = min(self.linear_speed, distance * 2)  # Scale speed with distance
        else:
            # Already at target, stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def detect_obstacles_ahead(self):
        """
        Detect obstacles in front of the robot using laser scan data.
        """
        if self.laser_ranges is None:
            return False

        # Check laser readings in front of the robot (±30 degrees)
        front_indices = []
        for i, range_val in enumerate(self.laser_ranges):
            angle = self.laser_angle_min + i * self.laser_angle_increment
            if -math.pi/6 <= angle <= math.pi/6:  # Front 60 degrees
                if 0 < range_val < self.min_obstacle_distance:
                    front_indices.append(i)

        return len(front_indices) > 0

    def get_yaw_from_quaternion(self, quaternion):
        """
        Extract yaw angle from quaternion.
        """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def stop_robot(self):
        """
        Stop the robot by sending zero velocity commands.
        """
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_cmd)

    def emergency_stop(self):
        """
        Emergency stop function that immediately stops the robot.
        """
        self.navigation_active = False
        self.stop_robot()
        self.get_logger().warn('Emergency stop activated!')


class PathPlanner(Node):
    """
    A path planning node that implements A* algorithm for navigation.
    """

    def __init__(self):
        """
        Initialize the path planner node.
        """
        super().__init__('path_planner')

        # Create subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        self.start_sub = self.create_subscription(
            PoseStamped,
            'start_pose',
            self.start_pose_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_pose_callback,
            10
        )

        # Create publishers
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.global_plan_pub = self.create_publisher(Path, 'global_plan', 10)

        # Store map and planning data
        self.map_data = None
        self.start_pose = None
        self.goal_pose = None
        self.path_planning_active = False

        self.get_logger().info('Path planner node initialized')

    def map_callback(self, msg):
        """
        Callback function for map messages.
        """
        self.map_data = msg
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}')

    def start_pose_callback(self, msg):
        """
        Callback function for start pose messages.
        """
        self.start_pose = msg
        self.get_logger().info('Start pose received')

    def goal_pose_callback(self, msg):
        """
        Callback function for goal pose messages.
        """
        self.goal_pose = msg
        self.get_logger().info('Goal pose received')

        # Plan path if both start and goal are available
        if self.start_pose and self.goal_pose:
            self.plan_path()

    def plan_path(self):
        """
        Plan a path using A* algorithm.
        """
        if self.map_data is None:
            self.get_logger().warn('Cannot plan path: no map available')
            return

        # Convert start and goal poses to map coordinates
        start_cell = self.pose_to_map_coordinates(self.start_pose.pose)
        goal_cell = self.pose_to_map_coordinates(self.goal_pose.pose)

        # Run A* path planning
        path_cells = self.a_star_plan(start_cell, goal_cell)

        if path_cells:
            # Convert path cells back to world coordinates
            path_world = self.cells_to_world_coordinates(path_cells)

            # Create Path message
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'

            for point in path_world:
                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                pose_stamped.pose.position.x = point[0]
                pose_stamped.pose.position.y = point[1]
                pose_stamped.pose.position.z = 0.0
                path_msg.poses.append(pose_stamped)

            # Publish planned path
            self.path_pub.publish(path_msg)
            self.global_plan_pub.publish(path_msg)

            self.get_logger().info(f'Path planned with {len(path_world)} waypoints')
        else:
            self.get_logger().warn('No path found to goal')

    def pose_to_map_coordinates(self, pose):
        """
        Convert a pose in world coordinates to map cell coordinates.
        """
        if self.map_data is None:
            return None

        # Calculate map coordinates
        map_x = int((pose.position.x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        map_y = int((pose.position.y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)

        return (map_x, map_y)

    def cells_to_world_coordinates(self, cells):
        """
        Convert map cell coordinates to world coordinates.
        """
        if self.map_data is None:
            return []

        world_coords = []
        for cell_x, cell_y in cells:
            world_x = cell_x * self.map_data.info.resolution + self.map_data.info.origin.position.x
            world_y = cell_y * self.map_data.info.resolution + self.map_data.info.origin.position.y
            world_coords.append((world_x, world_y))

        return world_coords

    def a_star_plan(self, start, goal):
        """
        A* path planning algorithm implementation.
        """
        if self.map_data is None:
            return []

        # Implementation of A* algorithm
        # This is a simplified version - in practice would be more complex
        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            open_set.remove(current)

            # Get neighbors
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if self.is_valid_cell(neighbor):
                    tentative_g_score = g_score[current] + self.distance(current, neighbor)

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)

                        if neighbor not in open_set:
                            open_set.append(neighbor)

        return []  # No path found

    def heuristic(self, a, b):
        """
        Heuristic function for A* (Euclidean distance).
        """
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, cell):
        """
        Get neighboring cells (8-connected).
        """
        x, y = cell
        neighbors = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),             (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        return neighbors

    def is_valid_cell(self, cell):
        """
        Check if a cell is valid (within map bounds and not occupied).
        """
        if self.map_data is None:
            return False

        x, y = cell

        # Check bounds
        if x < 0 or x >= self.map_data.info.width or y < 0 or y >= self.map_data.info.height:
            return False

        # Check if cell is occupied (value > 50 means occupied in occupancy grid)
        map_index = y * self.map_data.info.width + x
        if 0 <= map_index < len(self.map_data.data):
            if self.map_data.data[map_index] > 50:  # Occupied
                return False

        return True

    def distance(self, a, b):
        """
        Calculate distance between two cells.
        """
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def main(args=None):
    """
    Main function that initializes ROS2, creates navigation nodes, and starts spinning.
    """
    rclpy.init(args=args)

    # Create navigation node
    nav_node = SimpleNavigation()

    # Create path planner node
    planner_node = PathPlanner()

    try:
        # Create executor and add both nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(nav_node)
        executor.add_node(planner_node)

        # Spin both nodes
        executor.spin()
    except KeyboardInterrupt:
        nav_node.get_logger().info('Shutting down navigation node')
        planner_node.get_logger().info('Shutting down path planner')
    finally:
        nav_node.destroy_node()
        planner_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()