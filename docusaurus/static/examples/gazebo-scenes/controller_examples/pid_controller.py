#!/usr/bin/env python3

"""
PID Controller Example for Gazebo

This script demonstrates a PID controller for precise robot control in Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import time


class PIDController:
    """
    Generic PID controller implementation.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-float('inf'), float('inf'))):
        """
        Initialize the PID controller with gains and output limits.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.reset()

    def reset(self):
        """
        Reset the PID controller state.
        """
        self.previous_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.previous_time = None

    def compute(self, setpoint, measured_value, dt=None):
        """
        Compute the PID output based on setpoint and measured value.
        """
        current_time = time.time()

        if dt is None:
            if self.previous_time is None:
                dt = 0.01  # Default to 10ms if first call
            else:
                dt = current_time - self.previous_time

        self.previous_time = current_time

        # Calculate error
        error = setpoint - measured_value

        # Calculate integral
        self.integral += error * dt

        # Apply integral windup protection
        self.integral = max(self.output_limits[0], min(self.integral, self.output_limits[1]))

        # Calculate derivative
        if dt > 0:
            self.derivative = (error - self.previous_error) / dt
        else:
            self.derivative = 0

        # Calculate PID output
        proportional = self.kp * error
        integral_term = self.ki * self.integral
        derivative_term = self.kd * self.derivative

        output = proportional + integral_term + derivative_term

        # Apply output limits
        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        # Store error for next iteration
        self.previous_error = error

        return output


class PIDRobotController(Node):
    """
    Robot controller using PID for precise movement control.
    """

    def __init__(self):
        """
        Initialize the PID robot controller.
        """
        super().__init__('pid_robot_controller')

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
        self.current_twist = None
        self.laser_data = None

        # Control parameters
        self.control_frequency = 50  # Hz
        self.linear_tolerance = 0.05  # meters
        self.angular_tolerance = 0.05  # radians

        # PID controllers
        self.linear_pid = PIDController(kp=2.0, ki=0.1, kd=0.05, output_limits=(-0.5, 0.5))
        self.angular_pid = PIDController(kp=3.0, ki=0.2, kd=0.1, output_limits=(-1.0, 1.0))

        # Target pose
        self.target_position = Point(x=2.0, y=2.0, z=0.0)
        self.target_yaw = 0.0  # radians
        self.moving_to_target = False

        # Create timer for control loop
        self.control_timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)

        self.get_logger().info('PID robot controller initialized')

    def odom_callback(self, msg):
        """
        Callback function for odometry messages.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def laser_callback(self, msg):
        """
        Callback function for laser scan messages.
        """
        self.laser_data = msg

    def get_yaw_from_quaternion(self, quaternion):
        """
        Extract yaw angle from quaternion.
        """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def set_target(self, x, y, theta=0.0):
        """
        Set a new target position for the robot.
        """
        self.target_position.x = x
        self.target_position.y = y
        self.target_yaw = theta
        self.moving_to_target = True

        self.get_logger().info(f'Set target position: ({x}, {y}), theta: {theta:.2f}')

    def control_loop(self):
        """
        Main control loop using PID controllers.
        """
        if self.current_pose is None:
            return

        cmd_vel = Twist()

        if self.moving_to_target:
            # Calculate distance to target
            dx = self.target_position.x - self.current_pose.position.x
            dy = self.target_position.y - self.current_pose.position.y
            distance_to_target = math.sqrt(dx*dx + dy*dy)

            # Calculate angle to target
            target_angle = math.atan2(dy, dx)
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

            # Calculate angle difference
            angle_diff = target_angle - current_yaw
            # Normalize to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Check if we're close enough to target position
            if distance_to_target < self.linear_tolerance:
                # Now align to target orientation
                orientation_error = self.target_yaw - current_yaw
                # Normalize to [-pi, pi]
                while orientation_error > math.pi:
                    orientation_error -= 2 * math.pi
                while orientation_error < -math.pi:
                    orientation_error += 2 * math.pi

                if abs(orientation_error) < self.angular_tolerance:
                    # Reached target, stop
                    self.moving_to_target = False
                    self.get_logger().info('Reached target position and orientation')
                else:
                    # Align to target orientation
                    cmd_vel.angular.z = self.angular_pid.compute(self.target_yaw, current_yaw)
            else:
                # Move toward target position
                # First, align to face the target
                if abs(angle_diff) > 0.1:  # 0.1 radian threshold
                    cmd_vel.angular.z = self.angular_pid.compute(target_angle, current_yaw)
                else:
                    # Move forward while maintaining orientation toward target
                    cmd_vel.linear.x = self.linear_pid.compute(distance_to_target, 0)
                    cmd_vel.angular.z = self.angular_pid.compute(target_angle, current_yaw)

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        # Log current state
        if self.current_pose is not None:
            current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
            self.get_logger().info(
                f'Pos: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), '
                f'Yaw: {current_yaw:.2f}, '
                f'Target: ({self.target_position.x:.2f}, {self.target_position.y:.2f}), '
                f'Cmd: ({cmd_vel.linear.x:.2f}, {cmd_vel.angular.z:.2f})',
                throttle_duration_sec=1
            )

    def execute_trajectory(self):
        """
        Execute a predefined trajectory using PID control.
        """
        # Define a simple trajectory (square path)
        trajectory = [
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 1.57),  # Turn 90 degrees
            (0.0, 2.0, 3.14),  # Turn 90 degrees
            (0.0, 0.0, -1.57), # Turn 90 degrees
        ]

        for x, y, theta in trajectory:
            self.set_target(x, y, theta)

            # Wait until target is reached
            while self.moving_to_target and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

            # Wait a bit at each waypoint
            start_time = time.time()
            while time.time() - start_time < 1.0 and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)


def main(args=None):
    """
    Main function that initializes ROS2, creates the controller node, and starts spinning.
    """
    rclpy.init(args=args)

    controller = PIDRobotController()

    # Execute a predefined trajectory
    controller.execute_trajectory()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down PID robot controller')
    finally:
        # Stop the robot before shutting down
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()