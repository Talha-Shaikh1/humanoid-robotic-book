#!/usr/bin/env python3

"""
Advanced Humanoid Walking Gait Controller

This module implements an advanced walking gait controller for humanoid robots,
incorporating ZMP (Zero Moment Point) control, inverse kinematics, and balance
maintenance for stable locomotion.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, PoseStamped
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from humanoid_msgs.msg import BalanceState, WalkingGoal, WalkingFeedback, WalkingResult
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import math
import time


class ZMPPreviewController:
    """
    Zero Moment Point (ZMP) Preview Controller for humanoid walking.
    This controller uses preview control to maintain balance during walking.
    """

    def __init__(self, dt=0.01, preview_window=2.0):
        """
        Initialize the ZMP preview controller.

        Args:
            dt: Control loop timestep (seconds)
            preview_window: Time window for preview control (seconds)
        """
        self.dt = dt
        self.preview_steps = int(preview_window / dt)

        # Robot parameters
        self.com_height = 0.65  # Center of mass height (meters)
        self.gravity = 9.81

        # Precompute control matrices
        self.omega = math.sqrt(self.gravity / self.com_height)
        self.A = np.array([[1, self.dt, (math.cosh(self.omega*self.dt) - 1) / (self.omega**2)],
                          [0, 1, math.sinh(self.omega*self.dt) / self.omega],
                          [self.omega**2, 0, math.cosh(self.omega*self.dt)]])

        self.B = np.array([[(self.omega*self.dt - math.sinh(self.omega*self.dt)) / (self.omega**3)],
                          [(1 - math.cosh(self.omega*self.dt)) / (self.omega**2)],
                          [math.sinh(self.omega*self.dt) / self.omega]])

        # State: [x, dx, zmp_error_integral]
        self.state = np.zeros(3)

        # Feedback gain matrix (computed offline for stability)
        self.K = np.array([-2.0, -2.0, -0.5])  # Simplified gains for demonstration

    def update(self, desired_zmp, measured_zmp, reference_com_x, reference_com_y):
        """
        Update the controller with new measurements and references.

        Args:
            desired_zmp: Desired ZMP position [x, y]
            measured_zmp: Measured ZMP position [x, y]
            reference_com: Reference COM position [x, y]

        Returns:
            next_com_state: Next state for center of mass trajectory
        """
        # Calculate ZMP error
        zmp_error = np.array(desired_zmp) - np.array(measured_zmp)

        # Update integral term in state
        self.state[2] += zmp_error[0] * self.dt  # Integral of x error

        # Calculate control input using state feedback
        control_input = -np.dot(self.K, self.state)

        # Update state using system dynamics
        self.state = np.dot(self.A, self.state) + self.B * control_input

        # Return next COM position based on state
        next_com_x = self.state[0]
        next_com_y = reference_com_y  # Simplified - maintain y reference

        return [next_com_x, next_com_y]


class InverseKinematicsSolver:
    """
    Inverse Kinematics solver for humanoid leg using analytical and numerical methods.
    """

    def __init__(self, leg_params):
        """
        Initialize IK solver with leg parameters.

        Args:
            leg_params: Dictionary containing leg parameters like link lengths
        """
        self.upper_leg_length = leg_params.get('upper_leg', 0.35)  # meters
        self.lower_leg_length = leg_params.get('lower_leg', 0.35)  # meters
        self.foot_length = leg_params.get('foot', 0.15)  # meters

    def solve_leg_ik(self, target_pos, leg_side='left'):
        """
        Solve inverse kinematics for a single leg.

        Args:
            target_pos: Target foot position [x, y, z] relative to hip
            leg_side: 'left' or 'right' for coordinate system adjustment

        Returns:
            joint_angles: [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
        """
        x, y, z = target_pos

        # Calculate hip yaw angle
        hip_yaw = math.atan2(y, x) if math.sqrt(x**2 + y**2) > 0.01 else 0.0

        # Project to 2D plane for remaining calculations
        r_horiz = math.sqrt(x**2 + y**2)  # Horizontal distance from hip to foot
        r_total = math.sqrt(r_horiz**2 + z**2)  # 3D distance from hip to foot

        # Hip roll compensation for lateral movement
        hip_roll = math.atan2(-z, r_horiz)  # Simplified calculation

        # Calculate knee angle using law of cosines
        # Triangle formed by upper leg, lower leg, and distance from hip to foot
        cos_knee = (self.upper_leg_length**2 + self.lower_leg_length**2 - r_total**2) / \
                   (2 * self.upper_leg_length * self.lower_leg_length)

        # Clamp to valid range for acos
        cos_knee = max(-1.0, min(1.0, cos_knee))
        knee_angle = math.pi - math.acos(cos_knee)

        # Calculate hip pitch angle
        # First, calculate angle between upper leg and line from hip to foot
        alpha = math.acos((self.upper_leg_length**2 + r_total**2 - self.lower_leg_length**2) /
                         (2 * self.upper_leg_length * r_total))

        # Hip pitch is the angle from vertical minus alpha
        hip_pitch = math.atan2(z, r_horiz) + alpha

        # Calculate ankle angles to maintain foot orientation
        # Simplified: ankle compensates for knee and hip angles
        ankle_pitch = -(hip_pitch + (math.pi - knee_angle))
        ankle_roll = -hip_roll  # Compensate for hip roll

        # Adjust for leg side (left/right symmetry)
        multiplier = 1.0 if leg_side == 'left' else -1.0

        return [
            hip_yaw,  # Hip yaw
            hip_roll * multiplier,  # Hip roll (inverted for right leg)
            hip_pitch,  # Hip pitch
            knee_angle,  # Knee angle
            ankle_pitch,  # Ankle pitch
            ankle_roll * multiplier  # Ankle roll (inverted for right leg)
        ]


class WalkingPatternGenerator:
    """
    Generates walking patterns including footstep positions and body trajectories.
    """

    def __init__(self, step_length=0.3, step_height=0.05, step_duration=0.8):
        """
        Initialize walking pattern generator.

        Args:
            step_length: Forward step length (meters)
            step_height: Maximum foot lift height (meters)
            step_duration: Time for each step (seconds)
        """
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.dt = 0.01  # Control timestep

        # Walking parameters
        self.stride_length = step_length
        self.step_width = 0.18  # Lateral distance between feet
        self.swing_phase_ratio = 0.3  # Fraction of step for swing phase

    def generate_step_trajectory(self, start_pos, target_pos, current_time, total_time):
        """
        Generate smooth trajectory for a single step.

        Args:
            start_pos: Starting foot position [x, y, z]
            target_pos: Target foot position [x, y, z]
            current_time: Current time in the step cycle
            total_time: Total time for this step

        Returns:
            current_foot_pos: Current foot position along the trajectory
        """
        # Calculate phase (0.0 to 1.0) of the step
        phase = min(1.0, current_time / total_time)

        # Calculate stance/swing phases
        swing_start = (1.0 - self.swing_phase_ratio) / 2.0
        swing_end = swing_start + self.swing_phase_ratio

        # Calculate horizontal position using cubic interpolation
        if phase < swing_start:
            # Early stance phase - hold at start position
            t_interp = phase / swing_start
            pos_x = start_pos[0] + (target_pos[0] - start_pos[0]) * 0.5 * t_interp**2
            pos_y = start_pos[1] + (target_pos[1] - start_pos[1]) * 0.5 * t_interp**2
        elif phase <= swing_end:
            # Swing phase - move to target with lift
            t_swing = (phase - swing_start) / self.swing_phase_ratio
            # Horizontal movement
            pos_x = start_pos[0] + (target_pos[0] - start_pos[0]) * (
                0.5 + 0.5 * math.sin(math.pi * (t_swing - 0.5))
            )
            pos_y = start_pos[1] + (target_pos[1] - start_pos[1]) * (
                0.5 + 0.5 * math.sin(math.pi * (t_swing - 0.5))
            )
            # Vertical lift - sinusoidal profile
            if t_swing < 0.5:
                pos_z = start_pos[2] + self.step_height * math.sin(math.pi * t_swing)
            else:
                pos_z = target_pos[2] + self.step_height * math.sin(math.pi * (1 - t_swing))
        else:
            # Late stance phase - hold at target position
            t_interp = (phase - swing_end) / (1.0 - swing_end)
            pos_x = target_pos[0]
            pos_y = target_pos[1]
            pos_z = target_pos[2] + (start_pos[2] - target_pos[2]) * t_interp  # Smooth landing

        return [pos_x, pos_y, pos_z]

    def generate_com_trajectory(self, walk_params):
        """
        Generate Center of Mass trajectory for stable walking.

        Args:
            walk_params: Walking parameters dictionary

        Returns:
            com_trajectory: Smooth COM trajectory
        """
        # Simplified: generate a smooth body trajectory that moves forward
        # while maintaining balance over the supporting foot
        pass


class AdvancedWalkingController(Node):
    """
    Advanced walking controller that integrates ZMP control, IK, and walking pattern generation.
    """

    def __init__(self):
        """
        Initialize the advanced walking controller node.
        """
        super().__init__('advanced_walking_controller')

        # Initialize controller components
        self.zmp_controller = ZMPPreviewController(dt=0.01)
        self.ik_solver = InverseKinematicsSolver({
            'upper_leg': 0.35,
            'lower_leg': 0.35,
            'foot': 0.15
        })
        self.pattern_generator = WalkingPatternGenerator()

        # Walking state
        self.is_walking = False
        self.walk_velocity = [0.0, 0.0, 0.0]  # x, y, theta velocities
        self.current_step_time = 0.0
        self.total_step_time = 0.8
        self.support_leg = 'left'  # Which leg is currently supporting
        self.step_count = 0

        # Robot state
        self.joint_states = {}
        self.imu_data = None
        self.com_estimate = [0.0, 0.0, 0.65]  # Estimated center of mass

        # Publishers and subscribers
        self.joint_cmd_pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)
        self.balance_state_pub = self.create_publisher(BalanceState, '/balance_state', 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Action server for walking goals
        self.walking_action_server = ActionServer(
            self,
            WalkingController,
            'walking_controller',
            execute_callback=self.execute_walking_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz control

        self.get_logger().info('Advanced Walking Controller initialized')

    def joint_state_callback(self, msg):
        """
        Callback for joint state messages.
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = msg.position[i]

    def imu_callback(self, msg):
        """
        Callback for IMU data.
        """
        self.imu_data = msg

    def cmd_vel_callback(self, msg):
        """
        Callback for velocity commands.
        """
        self.walk_velocity = [msg.linear.x, msg.linear.y, msg.angular.z]

    def goal_callback(self, goal_request):
        """
        Handle incoming walking goal requests.
        """
        self.get_logger().info(f'Received walking goal: {goal_request}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Handle goal cancellation requests.
        """
        self.get_logger().info('Received request to cancel walking goal')
        return CancelResponse.ACCEPT

    def execute_walking_callback(self, goal_handle):
        """
        Execute the walking action.
        """
        self.get_logger().info('Executing walking goal')

        # Get goal parameters
        goal = goal_handle.request

        # Initialize walking state
        self.is_walking = True
        self.walk_velocity = [goal.linear_velocity.x, goal.linear_velocity.y, goal.angular_velocity.z]

        feedback_msg = WalkingFeedback()
        result_msg = WalkingResult()

        # Walking control loop
        while self.is_walking:
            # Check if goal was cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result_msg.success = False
                self.is_walking = False
                return result_msg

            # Publish feedback
            feedback_msg.progress = self.current_step_time / self.total_step_time
            goal_handle.publish_feedback(feedback_msg)

            # Small delay to allow other callbacks to execute
            time.sleep(0.01)

        # Complete successfully
        goal_handle.succeed()
        result_msg.success = True
        return result_msg

    def estimate_zmp(self):
        """
        Estimate Zero Moment Point from current robot state.
        """
        # Simplified ZMP estimation based on force distribution
        # In a real implementation, this would use force/torque sensors

        # For now, estimate from joint positions and IMU data
        if self.imu_data is not None:
            # Extract roll and pitch from IMU quaternion
            w, x, y, z = (self.imu_data.orientation.w,
                          self.imu_data.orientation.x,
                          self.imu_data.orientation.y,
                          self.imu_data.orientation.z)

            # Convert quaternion to roll/pitch
            roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = math.asin(2*(w*y - z*x))

            # Simplified ZMP estimation based on tilt
            zmp_x = -self.com_height * math.tan(pitch)
            zmp_y = self.com_height * math.tan(roll)

            return [zmp_x, zmp_y]

        return [0.0, 0.0]  # Default if no IMU data

    def calculate_support_polygon(self):
        """
        Calculate the support polygon based on foot positions.
        """
        # Get current foot positions from joint states and kinematics
        # This would use forward kinematics in a real implementation

        # Simplified: assume feet are at fixed positions when on ground
        if self.support_leg == 'left':
            # Right foot is in air, left foot provides support
            return [
                [-0.1, -0.09],  # Left foot corners
                [0.1, -0.09],
                [0.1, 0.09],
                [-0.1, 0.09]
            ]
        else:
            # Left foot is in air, right foot provides support
            return [
                [-0.1, -0.09-self.pattern_generator.step_width],  # Right foot corners
                [0.1, -0.09-self.pattern_generator.step_width],
                [0.1, 0.09-self.pattern_generator.step_width],
                [-0.1, 0.09-self.pattern_generator.step_width]
            ]

    def is_zmp_stable(self, zmp_pos, support_polygon):
        """
        Check if the ZMP is within the support polygon.
        """
        # Simplified point-in-polygon check for rectangular support area
        zmp_x, zmp_y = zmp_pos

        # Get bounding box of support polygon
        min_x = min(p[0] for p in support_polygon)
        max_x = max(p[0] for p in support_polygon)
        min_y = min(p[1] for p in support_polygon)
        max_y = max(p[1] for p in support_polygon)

        return min_x <= zmp_x <= max_x and min_y <= zmp_y <= max_y

    def control_loop(self):
        """
        Main control loop running at 100Hz.
        """
        if not self.is_walking:
            return

        # Update step timing
        self.current_step_time += 0.01
        if self.current_step_time >= self.total_step_time:
            # Complete current step, switch support leg
            self.current_step_time = 0.0
            self.support_leg = 'right' if self.support_leg == 'left' else 'left'
            self.step_count += 1

        # Estimate current ZMP
        current_zmp = self.estimate_zmp()

        # Calculate desired ZMP based on walking pattern
        # For forward walking, desired ZMP should follow a stable trajectory
        step_phase = self.current_step_time / self.total_step_time
        desired_zmp_x = self.walk_velocity[0] * 0.2  # Simplified: proportional to velocity
        desired_zmp_y = 0.0  # Try to maintain centered ZMP

        # Adjust for turning
        if abs(self.walk_velocity[2]) > 0.01:
            turn_effect = self.walk_velocity[2] * 0.1  # Simplified turning effect
            desired_zmp_y = turn_effect

        desired_zmp = [desired_zmp_x, desired_zmp_y]

        # Get support polygon
        support_poly = self.calculate_support_polygon()

        # Check stability
        is_stable = self.is_zmp_stable(current_zmp, support_poly)

        # Update ZMP controller
        reference_com = [0.0, 0.0]  # Reference center of mass position
        next_com_state = self.zmp_controller.update(
            desired_zmp, current_zmp, reference_com[0], reference_com[1]
        )

        # Generate foot trajectories
        if self.support_leg == 'left':
            # Right foot is swinging
            right_foot_target = [
                self.step_count * self.pattern_generator.stride_length + 0.2,  # Forward position
                -self.pattern_generator.step_width / 2,  # Lateral position
                0.0  # Height (on ground)
            ]

            # Generate swing trajectory for right foot
            right_foot_pos = self.pattern_generator.generate_step_trajectory(
                [self.step_count * self.pattern_generator.stride_length,
                 -self.pattern_generator.step_width / 2, 0.0],  # Previous position
                right_foot_target,
                self.current_step_time,
                self.total_step_time
            )

            # Left foot stays in place for now (simplified)
            left_foot_pos = [
                (self.step_count-1) * self.pattern_generator.stride_length,
                self.pattern_generator.step_width / 2,
                0.0
            ]
        else:
            # Left foot is swinging
            left_foot_target = [
                self.step_count * self.pattern_generator.stride_length + 0.2,  # Forward position
                self.pattern_generator.step_width / 2,  # Lateral position
                0.0  # Height (on ground)
            ]

            # Generate swing trajectory for left foot
            left_foot_pos = self.pattern_generator.generate_step_trajectory(
                [(self.step_count-1) * self.pattern_generator.stride_length,
                 self.pattern_generator.step_width / 2, 0.0],  # Previous position
                left_foot_target,
                self.current_step_time,
                self.total_step_time
            )

            # Right foot stays in place for now (simplified)
            right_foot_pos = [
                (self.step_count-1) * self.pattern_generator.stride_length,
                -self.pattern_generator.step_width / 2,
                0.0
            ]

        # Solve inverse kinematics for both legs
        left_leg_joints = self.ik_solver.solve_leg_ik(left_foot_pos, 'left')
        right_leg_joints = self.ik_solver.solve_leg_ik(right_foot_pos, 'right')

        # Combine with arm movements for balance
        # Simplified: add small arm movements to counteract leg motion
        arm_compensation = math.sin(step_phase * 2 * math.pi) * 0.1

        # Prepare joint commands
        joint_names = [
            # Left leg
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            # Right leg
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            # Left arm for balance
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_yaw',
            # Right arm for balance
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow', 'right_wrist_pitch', 'right_wrist_yaw'
        ]

        positions = (
            left_leg_joints +
            right_leg_joints +
            # Simplified arm positions for balance
            [-0.2, 0.1, 0.0, -0.5, 0.0, 0.0,  # Left arm
             0.2, -0.1, 0.0, -0.5, 0.0, 0.0]   # Right arm
        )

        # Create and publish joint trajectory message
        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)  # Zero velocities for simplicity
        point.accelerations = [0.0] * len(positions)  # Zero accelerations for simplicity

        # Set time from start (very small since this is a single point)
        point.time_from_start = Duration(sec=0, nanosec=10000000)  # 10ms

        joint_trajectory.points = [point]

        # Publish the joint commands
        self.joint_cmd_pub.publish(joint_trajectory)

        # Publish balance state for monitoring
        balance_state = BalanceState()
        balance_state.header.stamp = self.get_clock().now().to_msg()
        balance_state.zmp_x = current_zmp[0]
        balance_state.zmp_y = current_zmp[1]
        balance_state.desired_zmp_x = desired_zmp[0]
        balance_state.desired_zmp_y = desired_zmp[1]
        balance_state.is_balanced = is_stable
        balance_state.support_leg = self.support_leg
        balance_state.walk_velocity_x = self.walk_velocity[0]
        balance_state.walk_velocity_y = self.walk_velocity[1]
        balance_state.turn_velocity = self.walk_velocity[2]

        self.balance_state_pub.publish(balance_state)

        # Log stability information
        stability_msg = "STABLE" if is_stable else "UNSTABLE"
        self.get_logger().debug(
            f'Walk control - Step: {self.step_count}, Phase: {step_phase:.2f}, '
            f'ZMP: ({current_zmp[0]:.3f}, {current_zmp[1]:.3f}), '
            f'Desired: ({desired_zmp[0]:.3f}, {desired_zmp[1]:.3f}), '
            f'Stability: {stability_msg}'
        )


def main(args=None):
    """
    Main function to run the advanced walking controller.
    """
    rclpy.init(args=args)

    walking_controller = AdvancedWalkingController()

    try:
        rclpy.spin(walking_controller)
    except KeyboardInterrupt:
        walking_controller.get_logger().info('Shutting down advanced walking controller')
    finally:
        walking_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()