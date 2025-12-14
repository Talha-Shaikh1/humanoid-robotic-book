#!/usr/bin/env python3

"""
ROS2 Controller Example

This example demonstrates how to create and implement robot controllers in ROS2,
including joint controllers, trajectory controllers, and custom control algorithms.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
import numpy as np
import math
from collections import deque


class JointController(Node):
    """
    A simple joint controller that demonstrates basic control concepts.
    """

    def __init__(self):
        """
        Initialize the joint controller node.
        """
        super().__init__('joint_controller')

        # Store joint information
        self.joint_names = []
        self.current_positions = {}
        self.current_velocities = {}
        self.current_efforts = {}
        self.desired_positions = {}
        self.desired_velocities = {}
        self.desired_accelerations = {}

        # Control parameters
        self.control_frequency = 50  # Hz
        self.kp = 10.0  # Proportional gain
        self.ki = 0.1   # Integral gain
        self.kd = 0.5   # Derivative gain

        # PID error accumulators
        self.previous_errors = {}
        self.integral_errors = {}

        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Create publishers
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10
        )

        self.controller_state_pub = self.create_publisher(
            JointTrajectoryControllerState,
            'controller_state',
            10
        )

        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )

        self.get_logger().info('Joint controller initialized')

    def joint_state_callback(self, msg):
        """
        Callback function for joint state messages.
        Updates the current state of all joints.
        """
        # Update current joint states
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_efforts[name] = msg.effort[i]

    def control_loop(self):
        """
        Main control loop that calculates control commands.
        """
        # This is where the control algorithm would run
        # For now, we'll implement a simple position controller
        commands = Float64MultiArray()

        for joint_name in self.current_positions:
            # Simple PD controller for position control
            current_pos = self.current_positions[joint_name]
            desired_pos = self.desired_positions.get(joint_name, current_pos)

            # Calculate error
            error = desired_pos - current_pos

            # Calculate derivative of error
            prev_error = self.previous_errors.get(joint_name, 0.0)
            derivative = error - prev_error

            # Calculate control output
            output = self.kp * error + self.kd * derivative

            # Update error accumulators
            self.previous_errors[joint_name] = error

            # Add to commands
            commands.data.append(output)

        # Publish joint commands
        if commands.data:
            self.joint_command_pub.publish(commands)

    def set_joint_positions(self, joint_positions):
        """
        Set desired joint positions for the controller.
        """
        for joint_name, position in joint_positions.items():
            self.desired_positions[joint_name] = position


class TrajectoryController(Node):
    """
    A trajectory controller that follows joint trajectory commands.
    """

    def __init__(self):
        """
        Initialize the trajectory controller node.
        """
        super().__init__('trajectory_controller')

        # Store trajectory information
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.active_trajectory = False

        # Control parameters
        self.control_frequency = 100  # Hz
        self.lookahead_time = 0.1  # seconds

        # Create subscribers
        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            'joint_trajectory',
            self.trajectory_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Create publishers
        self.joint_command_pub = self.create_publisher(
            JointTrajectoryPoint,
            'joint_trajectory_commands',
            10
        )

        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.trajectory_control_loop
        )

        self.get_logger().info('Trajectory controller initialized')

    def trajectory_callback(self, msg):
        """
        Callback function for joint trajectory messages.
        Stores the trajectory for execution.
        """
        self.current_trajectory = msg
        self.trajectory_start_time = self.get_clock().now()
        self.active_trajectory = True

        self.get_logger().info(f'Received trajectory with {len(msg.points)} points')

    def joint_state_callback(self, msg):
        """
        Callback function for joint state messages.
        Updates current joint states.
        """
        self.current_joint_states = msg

    def trajectory_control_loop(self):
        """
        Main trajectory control loop that interpolates between trajectory points.
        """
        if not self.active_trajectory or not self.current_trajectory:
            return

        # Calculate current time in trajectory
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.trajectory_start_time).nanoseconds / 1e9

        # Find the appropriate trajectory point based on time
        current_point = self.interpolate_trajectory_point(elapsed_time)

        if current_point:
            # Publish interpolated trajectory point as command
            self.joint_command_pub.publish(current_point)

    def interpolate_trajectory_point(self, elapsed_time):
        """
        Interpolate the current trajectory point based on elapsed time.
        """
        if not self.current_trajectory:
            return None

        # Find the appropriate segment in the trajectory
        for i in range(len(self.current_trajectory.points) - 1):
            point_time = self.current_trajectory.points[i].time_from_start.sec + \
                         self.current_trajectory.points[i].time_from_start.nanosec / 1e9

            next_point_time = self.current_trajectory.points[i+1].time_from_start.sec + \
                              self.current_trajectory.points[i+1].time_from_start.nanosec / 1e9

            if point_time <= elapsed_time < next_point_time:
                # Interpolate between these two points
                t1, t2 = point_time, next_point_time
                p1, p2 = self.current_trajectory.points[i], self.current_trajectory.points[i+1]

                # Calculate interpolation factor
                alpha = (elapsed_time - t1) / (t2 - t1) if (t2 - t1) != 0 else 0

                # Interpolate position, velocity, and acceleration
                interpolated_point = JointTrajectoryPoint()

                for j in range(len(p1.positions)):
                    pos_interp = p1.positions[j] + alpha * (p2.positions[j] - p1.positions[j])
                    interpolated_point.positions.append(pos_interp)

                    if len(p1.velocities) > j and len(p2.velocities) > j:
                        vel_interp = p1.velocities[j] + alpha * (p2.velocities[j] - p1.velocities[j])
                        interpolated_point.velocities.append(vel_interp)

                    if len(p1.accelerations) > j and len(p2.accelerations) > j:
                        acc_interp = p1.accelerations[j] + alpha * (p2.accelerations[j] - p1.accelerations[j])
                        interpolated_point.accelerations.append(acc_interp)

                # Set time for the interpolated point
                interpolated_point.time_from_start.sec = int(elapsed_time)
                interpolated_point.time_from_start.nanosec = int((elapsed_time % 1) * 1e9)

                return interpolated_point

        # If we've reached the end of the trajectory
        if elapsed_time >= self.current_trajectory.points[-1].time_from_start.sec + \
                          self.current_trajectory.points[-1].time_from_start.nanosec / 1e9:
            # Return the last point
            self.active_trajectory = False
            self.get_logger().info('Trajectory execution completed')
            return self.current_trajectory.points[-1]

        return None


class PIDController:
    """
    Generic PID controller implementation for robotics applications.
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


class RobotController(Node):
    """
    Main robot controller that integrates multiple control strategies.
    """

    def __init__(self):
        """
        Initialize the main robot controller.
        """
        super().__init__('robot_controller')

        # Initialize PID controllers for different joints
        self.joint_controllers = {}

        # Joint names for the robot
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow', 'right_wrist_pitch', 'right_wrist_yaw'
        ]

        # Create PID controllers for each joint
        for joint_name in self.joint_names:
            self.joint_controllers[joint_name] = PIDController(
                kp=10.0, ki=0.1, kd=0.5,
                output_limits=(-100.0, 100.0)  # Torque limits
            )

        # Current joint states
        self.current_joint_states = JointState()
        self.desired_joint_positions = {name: 0.0 for name in self.joint_names}

        # Create subscribers and publishers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.joint_command_pub = self.create_publisher(
            JointState,
            'joint_commands',
            10
        )

        self.desired_positions_sub = self.create_subscription(
            JointState,
            'desired_joint_positions',
            self.desired_positions_callback,
            10
        )

        # Control timer
        self.control_frequency = 100  # Hz
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )

        self.get_logger().info(f'Robot controller initialized with {len(self.joint_names)} joints')

    def joint_state_callback(self, msg):
        """
        Update current joint states from incoming messages.
        """
        self.current_joint_states = msg

        # Update current positions in our dictionary
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_states[name] = msg.position[i]

    def desired_positions_callback(self, msg):
        """
        Update desired joint positions from incoming messages.
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.desired_joint_positions[name] = msg.position[i]

    def control_loop(self):
        """
        Main control loop that computes control commands for all joints.
        """
        if not self.current_joint_states.name:
            return  # Wait for first joint state message

        # Create command message
        command_msg = JointState()
        command_msg.header.stamp = self.get_clock().now().to_msg()
        command_msg.name = []
        command_msg.effort = []  # Using effort control

        # Compute control commands for each joint
        for joint_name in self.joint_names:
            if joint_name in self.current_joint_states.position:
                current_pos = self.current_joint_states.position[
                    self.current_joint_states.name.index(joint_name)
                ]
                desired_pos = self.desired_joint_positions.get(joint_name, current_pos)

                # Get the PID controller for this joint
                controller = self.joint_controllers[joint_name]

                # Compute control output
                control_output = controller.compute(desired_pos, current_pos)

                # Add to command message
                command_msg.name.append(joint_name)
                command_msg.effort.append(control_output)

        # Publish commands
        if command_msg.name:
            self.joint_command_pub.publish(command_msg)

    def set_desired_positions(self, positions_dict):
        """
        Set desired positions for multiple joints at once.
        """
        for joint_name, position in positions_dict.items():
            if joint_name in self.desired_joint_positions:
                self.desired_joint_positions[joint_name] = position

    def move_to_home_position(self):
        """
        Move all joints to their home (zero) position.
        """
        home_positions = {name: 0.0 for name in self.joint_names}
        self.set_desired_positions(home_positions)
        self.get_logger().info('Moving to home position')

    def execute_trajectory(self, trajectory_points):
        """
        Execute a series of trajectory points.
        """
        # This would implement trajectory following logic
        # For now, we'll just move to the first point
        if trajectory_points:
            self.set_desired_positions(trajectory_points[0])


def main(args=None):
    """
    Main function that initializes ROS2, creates the controller node, and starts spinning.
    """
    rclpy.init(args=args)

    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down robot controller')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()