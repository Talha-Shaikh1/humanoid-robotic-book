# Robotics Software Documentation Standards Guide

## Purpose
This guide establishes documentation standards for the AI-Native Humanoid Robotics textbook project. It ensures consistent, comprehensive, and accessible documentation that meets the needs of students, developers, and educators working with humanoid robotics systems.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Apply consistent documentation standards across all robotics software components
- Create comprehensive API documentation for robotic systems
- Document robot models, simulation environments, and control systems
- Implement effective commenting and code documentation practices
- Create user manuals and technical guides for robotic applications

## Documentation Philosophy

### The Three Pillars of Robotics Documentation
Effective robotics documentation rests on three pillars:

1. **Accessibility**: Documentation should be understandable by the target audience (students → intermediate developers)
2. **Completeness**: Documentation should cover all necessary aspects of the system
3. **Maintainability**: Documentation should be easy to update and maintain over time

<!-- RAG_CHUNK_ID: robotics-documentation-pillars -->

### Audience-Centric Approach
Documentation must be tailored to different audiences:

- **Students**: Focus on learning objectives, conceptual understanding, and step-by-step instructions
- **Developers**: Emphasize API details, implementation patterns, and integration guidelines
- **Educators**: Provide teaching materials, learning outcomes, and assessment guidelines
- **Researchers**: Include technical specifications, experimental procedures, and validation results

<!-- RAG_CHUNK_ID: documentation-audience-approach -->

## Code Documentation Standards

### Inline Comments
Follow these principles for inline comments in robotics code:

```python
# Good: Explains WHY, not WHAT
# Use PD controller with position feedback to maintain balance
# instead of pure velocity control which could cause oscillation
def balance_control(self, com_error, angular_velocity):
    kp = 10.0  # Proportional gain for balance (tuned experimentally)
    kd = 2.0   # Derivative gain for damping (prevents overshoot)
    return kp * com_error - kd * angular_velocity

# Bad: Just restates the obvious
# Multiply error by gain
torque = kp * error
```

<!-- RAG_CHUNK_ID: inline-comment-best-practices -->

### Function/Method Documentation
Use comprehensive docstrings for all functions:

```python
def calculate_inverse_kinematics(
    self,
    target_position,
    target_orientation,
    initial_joint_angles,
    max_iterations=100,
    tolerance=1e-6
):
    """
    Calculate inverse kinematics solution using Jacobian transpose method.

    This method solves the inverse kinematics problem for a humanoid leg,
    determining the required joint angles to achieve a target foot position
    and orientation. Uses iterative approach with Jacobian transpose for
    numerical solution.

    Args:
        target_position (list[float]): Target position [x, y, z] in base frame
        target_orientation (list[float]): Target orientation [qx, qy, qz, qw] as quaternion
        initial_joint_angles (list[float]): Initial joint configuration [hip, knee, ankle]
        max_iterations (int, optional): Maximum iterations for convergence. Defaults to 100
        tolerance (float, optional): Position/orientation tolerance for convergence. Defaults to 1e-6

    Returns:
        tuple[list[float], bool]: Joint angles and convergence success flag
        Returns ([0, 0, 0], False) if solution not found within constraints

    Raises:
        ValueError: If target position is unreachable or joint limits violated
        RuntimeError: If numerical method fails to converge

    Example:
        >>> leg_ik = LegIKSolver()
        >>> joints, success = leg_ik.calculate_inverse_kinematics(
        ...     [0.3, 0.0, 0.1],  # Target: 30cm forward, at 10cm height
        ...     [0, 0, 0, 1],     # Identity orientation (foot pointing down)
        ...     [0, 0, 0]         # Starting from neutral position
        ... )
        >>> if success:
        ...     print(f"Joint angles: {joints}")
        ... else:
        ...     print("IK solution not found")
    """
    # Implementation details...
    pass
```

<!-- RAG_CHUNK_ID: function-docstring-standards -->

### Class Documentation
Document classes with comprehensive information:

```python
class ZMPPreviewController:
    """
    Zero Moment Point (ZMP) Preview Controller for humanoid walking.

    This controller implements preview control to maintain balance during
    walking by tracking desired ZMP positions. The controller uses a
    finite preview horizon to anticipate future support polygon changes
    and adjust the center of mass trajectory accordingly.

    The ZMP preview controller operates by:
    1. Calculating the desired ZMP trajectory based on planned footsteps
    2. Using preview control to minimize ZMP tracking error
    3. Generating center of mass reference trajectory for balance control
    4. Providing feedback corrections based on actual ZMP measurements

    Attributes:
        preview_horizon (float): Time horizon for preview control (seconds)
        sampling_time (float): Controller sampling interval (seconds)
        com_height (float): Nominal center of mass height (meters)
        gravity (float): Gravitational constant (m/s^2)
        state_vector (numpy.ndarray): Current controller state [x, dx, y, dy]

    Typical usage:
        >>> controller = ZMPPreviewController(preview_horizon=2.0, sampling_time=0.01)
        >>> for t in range(1000):  # 10 seconds of control at 100Hz
        ...     desired_zmp = get_desired_zmp_trajectory(t)
        ...     current_zmp = measure_current_zmp()
        ...     com_ref = controller.update(desired_zmp, current_zmp)
        ...     apply_balance_control(com_ref)
    """

    def __init__(self, preview_horizon=2.0, sampling_time=0.01, com_height=0.8):
        """
        Initialize the ZMP preview controller.

        Args:
            preview_horizon (float): Time horizon for preview control (default 2.0s)
            sampling_time (float): Controller sampling time (default 0.01s)
            com_height (float): Nominal center of mass height (default 0.8m)

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if preview_horizon <= 0:
            raise ValueError(f"Preview horizon must be positive, got {preview_horizon}")
        if sampling_time <= 0:
            raise ValueError(f"Sampling time must be positive, got {sampling_time}")
        if com_height <= 0:
            raise ValueError(f"CoM height must be positive, got {com_height}")

        self.preview_horizon = preview_horizon
        self.sampling_time = sampling_time
        self.com_height = com_height
        self.gravity = 9.81

        # Calculate preview steps
        self.preview_steps = int(preview_horizon / sampling_time)

        # Initialize state vector [x, dx, y, dy]
        self.state_vector = np.zeros(4)

        # Pre-compute control matrices for efficiency
        self._initialize_control_matrices()
```

<!-- RAG_CHUNK_ID: class-documentation-standards -->

## API Documentation Standards

### RESTful API Documentation
For robotics services and APIs:

```python
class RobotControlAPI:
    """
    RESTful API for humanoid robot control and monitoring.

    This API provides endpoints for controlling the humanoid robot,
    monitoring its state, and managing its operations. All endpoints
    follow RESTful principles with appropriate HTTP methods and status codes.

    Security: All endpoints require authentication via JWT tokens.
    Rate limiting: Maximum 100 requests per minute per authenticated user.

    Base URL: https://robot-api.example.com/v1
    """

    @app.route('/v1/robot/state', methods=['GET'])
    def get_robot_state():
        """
        Retrieve current robot state information.

        Returns comprehensive information about the robot's current state
        including joint positions, velocities, efforts, base pose, and
        operational status.

        Returns:
            200: Robot state information
                {
                    "timestamp": "2023-10-15T14:30:00.123Z",
                    "joints": {
                        "left_hip_yaw": {"position": 0.1, "velocity": 0.0, "effort": 0.5},
                        "left_hip_roll": {"position": 0.05, "velocity": 0.01, "effort": 0.3},
                        ...
                    },
                    "base_pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.8},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    },
                    "status": "ACTIVE",
                    "battery_level": 0.85,
                    "temperature": 35.2
                }
            401: Unauthorized - Invalid or missing authentication token
            503: Service Unavailable - Robot is not responding

        Example:
            curl -H "Authorization: Bearer <token>" \
                 https://robot-api.example.com/v1/robot/state
        """
        # Implementation details...
        pass

    @app.route('/v1/robot/control/move', methods=['POST'])
    def execute_motion():
        """
        Execute a motion command on the robot.

        Initiates a motion command that moves the robot according to
        the specified parameters. The command is queued for execution
        and returns immediately with a command ID for status tracking.

        Request Body:
            {
                "motion_type": "walk_forward|turn|reach|grasp",
                "parameters": {
                    "distance": 1.0,  // meters (for walk_forward)
                    "angle": 0.5,     // radians (for turn)
                    "target_pose": {  // (for reach/grasp)
                        "position": {"x": 0.5, "y": 0.2, "z": 0.8},
                        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}
                    }
                },
                "duration": 2.0,  // seconds
                "priority": 1     // 0=low, 1=normal, 2=high, 3=critical
            }

        Returns:
            202: Accepted - Command queued for execution
                {
                    "command_id": "cmd_1234567890",
                    "estimated_completion": "2023-10-15T14:30:02.123Z",
                    "status": "QUEUED"
                }
            400: Bad Request - Invalid motion parameters
            401: Unauthorized - Invalid or missing authentication token
            422: Unprocessable Entity - Robot cannot execute command
            503: Service Unavailable - Robot is busy or not ready

        Example:
            curl -X POST \
                 -H "Authorization: Bearer <token>" \
                 -H "Content-Type: application/json" \
                 -d '{"motion_type": "walk_forward", "parameters": {"distance": 1.0}, "duration": 2.0}' \
                 https://robot-api.example.com/v1/robot/control/move
        """
        # Implementation details...
        pass
```

<!-- RAG_CHUNK_ID: api-documentation-standards -->

### ROS2 Message Documentation
Document ROS2 message types clearly:

```python
"""
Humanoid Robot Message Definitions

This module defines custom message types for humanoid robot applications.
All messages follow ROS2 message conventions and are compatible with
the ROS2 message generation tools.

Message Type: humanoid_robot_msgs/JointStateExtended
Purpose: Extended joint state information with safety and diagnostic data

Definition:
    # Standard joint state fields (compatible with sensor_msgs/JointState)
    string[] name
    float64[] position
    float64[] velocity
    float64[] effort

    # Extended fields for humanoid applications
    float64[] temperature          # Joint motor temperatures (°C)
    bool[] fault_status           # Joint fault status (true = fault)
    float64[] commanded_position  # Desired position from controller
    float64[] tracking_error      # Position tracking error (actual - desired)
    float64[] safety_margin       # Remaining safety margin (0.0 = at limit)

Constants:
    JOINT_FAULT_NONE = 0          # No fault detected
    JOINT_FAULT_OVERTEMP = 1      # Joint temperature exceeded limit
    JOINT_FAULT_OVERCURRENT = 2   # Joint current exceeded limit
    JOINT_FAULT_LIMIT_EXCEEDED = 3 # Joint limit exceeded

Usage:
    # Publisher example
    import rclpy
    from humanoid_robot_msgs.msg import JointStateExtended

    def publish_joint_states():
        joint_msg = JointStateExtended()
        joint_msg.name = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch']
        joint_msg.position = [0.1, 0.05, -0.2]
        joint_msg.velocity = [0.0, 0.01, -0.02]
        joint_msg.temperature = [35.2, 36.1, 34.8]
        joint_msg.fault_status = [False, False, False]
        publisher.publish(joint_msg)
"""

# Actual message definition would go in .msg files, but this documents usage
class JointStateExtended:
    """
    Documentation class showing proper usage of the message type.
    """
    pass
```

<!-- RAG_CHUNK_ID: ros2-message-documentation -->

## Robot Model Documentation

### URDF/XACRO Documentation
Document robot models with clarity:

```xml
<?xml version="1.0"?>
<!--
Humanoid Robot URDF Model - HR-01

This URDF model defines the physical structure of the HR-01 humanoid robot.
The model includes 28 degrees of freedom with 14 joints per leg and arms
designed for bipedal locomotion and manipulation tasks.

Model Structure:
- Base Link: pelvis with IMU and main computer
- Legs: 6 DOF each (hip yaw, roll, pitch; knee; ankle pitch, roll)
- Arms: 6 DOF each (shoulder pitch, roll, yaw; elbow; wrist pitch, yaw)
- Head: 2 DOF (yaw, pitch) with stereo camera

Coordinate Frames:
- base_link: Origin at robot pelvis, x-forward, y-left, z-up
- left_foot: Center of left foot sole
- right_foot: Center of right foot sole
- left_gripper: Center of left gripper, x-forward, z-up
- right_gripper: Center of right gripper, x-forward, z-up

Physical Properties:
- Total mass: ~35 kg
- Height: 1.2 m (standing)
- CoM height: ~0.65 m (standing)

Joint Limits (soft limits for safety):
- Hip joints: ±1.57 rad (±90°)
- Knee joints: 0 to 2.35 rad (0 to 135°)
- Ankle joints: ±0.5 rad (±28°)
- Shoulder joints: ±2.0 rad (±114°)
- Elbow joints: -2.0 to 0 rad (-114° to 0°)
- Wrist joints: ±1.57 rad (±90°)

Authors: Robotics Research Lab
Date: 2025-01-15
Version: 1.2.0
-->
<robot name="hr01_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include common definitions -->
  <xacro:include filename="$(find hr01_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find hr01_description)/urdf/transmissions.xacro"/>
  <xacro:include filename="$(find hr01_description)/urdf/gazebo.xacro"/>

  <!-- Base/Pelvis link -->
  <link name="base_link">
    <!-- Pelvis link containing main computer and IMU -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.15"/>
      </geometry>
      <material name="light_gray"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.15"/>
      </geometry>
    </collision>

    <inertial>
      <!-- Mass properties estimated from CAD model -->
      <mass value="8.0"/>  <!-- 8kg for pelvis structure + computer -->
      <inertia
        ixx="0.0833" ixy="0.0" ixz="0.0"
        iyy="0.1083" iyz="0.0"
        izz="0.1417"/>
    </inertial>
  </link>

  <!-- IMU sensor in pelvis -->
  <sensor name="pelvis_imu" type="imu">
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="imu_link"/>
  </sensor>

  <!-- Left leg definition -->
  <xacro:macro name="left_leg" params="prefix parent_link">
    <!-- Hip yaw joint (rotation around Z axis) -->
    <joint name="${prefix}_hip_yaw" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${prefix}_hip_yaw_link"/>
      <origin xyz="0 0.1 0" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="3.0"/>
      <safety_controller k_position="100" k_velocity="10" soft_lower_limit="${-M_PI/2 + 0.1}" soft_upper_limit="${M_PI/2 - 0.1}"/>
    </joint>

    <link name="${prefix}_hip_yaw_link">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.08"/>
        </geometry>
        <material name="dark_gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.08"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>

    <!-- Hip roll joint (rotation around X axis) -->
    <joint name="${prefix}_hip_roll" type="revolute">
      <parent link="${prefix}_hip_yaw_link"/>
      <child link="${prefix}_hip_roll_link"/>
      <origin xyz="0 0 -0.04" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="150" velocity="2.5"/>
      <safety_controller k_position="150" k_velocity="15" soft_lower_limit="${-M_PI/6 + 0.05}" soft_upper_limit="${M_PI/6 - 0.05}"/>
    </joint>

    <link name="${prefix}_hip_roll_link">
      <visual>
        <geometry>
          <cylinder radius="0.04" length="0.06"/>
        </geometry>
        <material name="dark_gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.04" length="0.06"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <inertia ixx="0.0008" ixy="0" ixz="0" iyy="0.0008" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <!-- Hip pitch joint (rotation around Y axis) -->
    <joint name="${prefix}_hip_pitch" type="revolute">
      <parent link="${prefix}_hip_roll_link"/>
      <child link="${prefix}_thigh_link"/>
      <origin xyz="0 0 -0.03" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-M_PI}" upper="0.5" effort="200" velocity="2.0"/>
      <safety_controller k_position="200" k_velocity="20" soft_lower_limit="${-M_PI + 0.1}" soft_upper_limit="${0.5 - 0.1}"/>
    </joint>

    <link name="${prefix}_thigh_link">
      <visual>
        <geometry>
          <capsule radius="0.06" length="0.35"/>
        </geometry>
        <material name="medium_gray"/>
      </visual>
      <collision>
        <geometry>
          <capsule radius="0.06" length="0.35"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="3.0"/>
        <inertia ixx="0.026" ixy="0" ixz="0" iyy="0.026" iyz="0" izz="0.0054"/>
      </inertial>
    </link>

    <!-- Knee joint -->
    <joint name="${prefix}_knee" type="revolute">
      <parent link="${prefix}_thigh_link"/>
      <child link="${prefix}_shank_link"/>
      <origin xyz="0 0 -0.35" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="${2*M_PI/3}" effort="200" velocity="2.0"/>
      <safety_controller k_position="200" k_velocity="20" soft_lower_limit="0.05" soft_upper_limit="${2*M_PI/3 - 0.05}"/>
    </joint>

    <link name="${prefix}_shank_link">
      <visual>
        <geometry>
          <capsule radius="0.05" length="0.35"/>
        </geometry>
        <material name="medium_gray"/>
      </visual>
      <collision>
        <geometry>
          <capsule radius="0.05" length="0.35"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.5"/>
        <inertia ixx="0.021" ixy="0" ixz="0" iyy="0.021" iyz="0" izz="0.0042"/>
      </inertial>
    </link>

    <!-- Ankle pitch joint -->
    <joint name="${prefix}_ankle_pitch" type="revolute">
      <parent link="${prefix}_shank_link"/>
      <child link="${prefix}_ankle_link"/>
      <origin xyz="0 0 -0.35" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="100" velocity="2.5"/>
      <safety_controller k_position="100" k_velocity="10" soft_lower_limit="${-M_PI/3 + 0.05}" soft_upper_limit="${M_PI/3 - 0.05}"/>
    </joint>

    <link name="${prefix}_ankle_link">
      <visual>
        <geometry>
          <box size="0.15 0.1 0.04"/>
        </geometry>
        <material name="dark_gray"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.15 0.1 0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.2"/>
        <inertia ixx="0.0025" ixy="0" ixz="0" iyy="0.0038" iyz="0" izz="0.0059"/>
      </inertial>
    </link>

    <!-- Ankle roll joint -->
    <joint name="${prefix}_ankle_roll" type="revolute">
      <parent link="${prefix}_ankle_link"/>
      <child link="${prefix}_foot_link"/>
      <origin xyz="0 0 -0.02" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="50" velocity="3.0"/>
      <safety_controller k_position="50" k_velocity="5" soft_lower_limit="${-M_PI/6 + 0.05}" soft_upper_limit="${M_PI/6 - 0.05}"/>
    </joint>

    <link name="${prefix}_foot_link">
      <visual>
        <geometry>
          <box size="0.2 0.1 0.04"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.2 0.1 0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <inertia ixx="0.0016" ixy="0" ixz="0" iyy="0.0047" iyz="0" izz="0.0059"/>
      </inertial>
    </link>

    <!-- Foot contact sensor -->
    <gazebo reference="${prefix}_foot_link">
      <sensor name="${prefix}_foot_contact" type="contact">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <contact>
          <collision>foot_collision</collision>
        </contact>
      </sensor>
    </gazebo>
  </xacro:macro>

  <!-- Instantiate left and right legs -->
  <xacro:left_leg prefix="left" parent_link="base_link"/>
  <xacro:left_leg prefix="right" parent_link="base_link">
    <!-- Override right leg origin -->
    <origin xyz="0 -0.1 0" rpy="0 0 0"/>
  </xacro:left_leg>

  <!-- Similar definitions for arms and head would follow -->
  <!-- ... -->

</robot>
```

<!-- RAG_CHUNK_ID: urdf-documentation-standards -->

## Simulation Environment Documentation

### Gazebo Integration Documentation
Document simulation environments clearly:

```python
"""
Gazebo Simulation Environment Configuration

This module documents the Gazebo simulation environment for the humanoid robot.
It includes world definitions, robot spawning procedures, and sensor configurations.
"""

# Example configuration file documentation
GAZEBO_ENVIRONMENT_CONFIG = {
    "world": {
        "name": "humanoid_arena",
        "gravity": [0, 0, -9.81],  # Standard Earth gravity
        "time_step": 0.001,        # Physics update rate (1ms)
        "real_time_factor": 1.0,   # Real-time simulation
        "solver": {
            "type": "ode",
            "iters": 1000,         # Iterations for constraint solver
            "precon_iters": 2,     # Preconditioning iterations
            "sor": 1.3,            # Successive Over Relaxation
            "use_dynamic_moi_rescaling": True
        },
        "constraints": {
            "cfm": 0.0,            # Constraint Force Mixing
            "erp": 0.2,            # Error Reduction Parameter
            "contact_max_correcting_vel": 100.0,
            "contact_surface_layer": 0.001
        }
    },
    "robot_spawn": {
        "model_name": "hr01_humanoid",
        "initial_pose": {
            "position": [0.0, 0.0, 0.85],  # Standing position
            "orientation": [0.0, 0.0, 0.0, 1.0]  # No rotation (quaternion)
        },
        "spawn_timeout": 30.0,     # Seconds to wait for successful spawn
        "plugins": [
            "libgazebo_ros2_control.so",  # ROS2 control plugin
            "libgazebo_ros_imu.so",       # IMU sensor plugin
            "libgazebo_ros_camera.so"     # Camera sensor plugin
        ]
    },
    "sensors": {
        "imu": {
            "name": "pelvis_imu",
            "update_rate": 100,            # Hz
            "topic": "/imu/data",
            "noise": {
                "gyroscope": {
                    "bias_mean": 0.0,
                    "bias_stddev": 0.001,
                    "precision": 1e-5
                },
                "accelerometer": {
                    "bias_mean": 0.0,
                    "bias_stddev": 0.01,
                    "precision": 1e-3
                }
            }
        },
        "cameras": {
            "head_camera": {
                "topic": "/camera/rgb/image_raw",
                "update_rate": 30,         # Hz
                "resolution": [640, 480],
                "fov": 1.047,              # Field of view in radians (~60 degrees)
                "image_format": "R8G8B8",
                "noise": {
                    "type": "gaussian",
                    "mean": 0.0,
                    "stddev": 0.007
                }
            }
        },
        "lidar": {
            "front_2d_lidar": {
                "topic": "/scan",
                "update_rate": 10,         # Hz
                "range": {
                    "min": 0.1,            # meters
                    "max": 10.0,           # meters
                    "resolution": 0.01     # meters
                },
                "scan": {
                    "horizontal": {
                        "samples": 720,      # Points per revolution
                        "resolution": 1,     # Points per degree
                        "min_angle": -2.356, # -135 degrees in radians
                        "max_angle": 2.356   # 135 degrees in radians
                    }
                }
            }
        }
    },
    "controllers": {
        "joint_state_controller": {
            "type": "joint_state_controller/JointStateController",
            "publish_rate": 50
        },
        "balance_controller": {
            "type": "position_controllers/JointGroupPositionController",
            "joints": [
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                "left_knee", "left_ankle_pitch", "left_ankle_roll",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                "right_knee", "right_ankle_pitch", "right_ankle_roll"
            ]
        }
    }
}


class GazeboEnvironmentManager:
    """
    Manages Gazebo simulation environment for humanoid robot testing.

    This class handles environment setup, robot spawning, sensor configuration,
    and simulation control for the humanoid robot in Gazebo. It provides
    methods for initializing the simulation, configuring sensors, and managing
    the simulation lifecycle.

    The environment manager ensures that:
    1. Robot model is correctly spawned in the simulation
    2. All sensors are properly configured with appropriate noise models
    3. Controllers are loaded and ready for operation
    4. Simulation parameters match real-world physics as closely as possible

    Usage:
        # Initialize environment
        env_manager = GazeboEnvironmentManager()
        env_manager.setup_environment()

        # Configure robot
        env_manager.spawn_robot(initial_pose=[0, 0, 0.85])

        # Start simulation
        env_manager.start_simulation()

        # Run experiments
        # ... perform robotics experiments ...

        # Clean up
        env_manager.cleanup()
    """

    def __init__(self, config_path=None):
        """
        Initialize the Gazebo environment manager.

        Args:
            config_path (str, optional): Path to custom configuration file.
                                        If None, uses default configuration.
        """
        self.config = self._load_configuration(config_path)
        self.simulation_running = False
        self.robot_spawned = False

        # Initialize ROS2 node for communication with Gazebo
        rclpy.init()
        self.node = rclpy.create_node('gazebo_env_manager')

        # Create publishers and subscribers for Gazebo communication
        self.spawn_client = self.node.create_client(Spawn, '/spawn_entity')
        self.delete_client = self.node.create_client(DeleteEntity, '/delete_entity')
        self.get_state_client = self.node.create_client(GetEntityState, '/get_entity_state')

    def _load_configuration(self, config_path):
        """
        Load environment configuration from file or use default.

        Args:
            config_path (str): Path to configuration file

        Returns:
            dict: Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return GAZEBO_ENVIRONMENT_CONFIG

    def setup_environment(self):
        """
        Set up the Gazebo environment with specified world and parameters.

        This method configures the physics engine, sets up the world environment,
        and prepares the simulation for robot spawning. It applies the physics
        parameters defined in the configuration to ensure accurate simulation.
        """
        self.node.get_logger().info('Setting up Gazebo environment...')

        # Set physics parameters via Gazebo service calls
        # This would involve calling Gazebo's physics parameter services
        self._configure_physics()

        # Load world if specified
        if 'world_file' in self.config:
            self._load_world(self.config['world_file'])

        self.node.get_logger().info('Gazebo environment setup complete')

    def spawn_robot(self, initial_pose=None):
        """
        Spawn the robot model in the Gazebo simulation.

        Args:
            initial_pose (list, optional): Initial [x,y,z,rx,ry,rz,rw] pose.
                                          If None, uses configuration default.

        Returns:
            bool: True if spawn successful, False otherwise
        """
        self.node.get_logger().info('Spawning robot in simulation...')

        if initial_pose is None:
            initial_pose = self.config['robot_spawn']['initial_pose']

        # Create spawn request
        spawn_request = Spawn.Request()
        spawn_request.name = self.config['robot_spawn']['model_name']
        spawn_request.xml = self._load_robot_model()
        spawn_request.initial_pose.position.x = initial_pose[0]
        spawn_request.initial_pose.position.y = initial_pose[1]
        spawn_request.initial_pose.position.z = initial_pose[2]
        spawn_request.initial_pose.orientation.x = initial_pose[3]
        spawn_request.initial_pose.orientation.y = initial_pose[4]
        spawn_request.initial_pose.orientation.z = initial_pose[5]
        spawn_request.initial_pose.orientation.w = initial_pose[6]

        # Send spawn request
        spawn_future = self.spawn_client.call_async(spawn_request)
        rclpy.spin_until_future_complete(self.node, spawn_future)

        response = spawn_future.result()
        if response.success:
            self.robot_spawned = True
            self.node.get_logger().info(f'Robot {spawn_request.name} spawned successfully')
            return True
        else:
            self.node.get_logger().error(f'Failed to spawn robot: {response.status_message}')
            return False

    def start_simulation(self):
        """
        Start the Gazebo simulation and begin robot operation.

        This method starts the physics simulation and ensures that all
        robot controllers are properly initialized and operational.
        """
        self.node.get_logger().info('Starting simulation...')

        # In a real implementation, this would call Gazebo's simulation services
        # to unpause the simulation and start physics updates

        self.simulation_running = True
        self.node.get_logger().info('Simulation started')

    def stop_simulation(self):
        """
        Stop the Gazebo simulation and safely shut down robot operation.
        """
        self.node.get_logger().info('Stopping simulation...')

        # In a real implementation, this would pause the simulation
        # and ensure safe robot shutdown

        self.simulation_running = False
        self.node.get_logger().info('Simulation stopped')

    def cleanup(self):
        """
        Clean up the Gazebo environment and shut down ROS2 node.
        """
        self.node.get_logger().info('Cleaning up environment...')

        # Stop simulation if running
        if self.simulation_running:
            self.stop_simulation()

        # Delete robot if spawned
        if self.robot_spawned:
            self._delete_robot()

        # Shutdown ROS2 node
        self.node.destroy_node()
        rclpy.shutdown()

        self.node.get_logger().info('Environment cleanup complete')
```

<!-- RAG_CHUNK_ID: gazebo-environment-documentation -->

## Quality Assurance Documentation

### Testing Documentation Standards
Document testing procedures and standards:

```python
"""
Robotics System Testing Documentation

This module defines testing standards and procedures for humanoid robotics systems.
It includes unit testing, integration testing, system testing, and validation
protocols for robotic applications.
"""

import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np
import time


class TestDocumentationStandards:
    """
    Standards for documenting robotics tests.

    All robotics tests should follow these documentation standards to ensure
    reproducibility, maintainability, and proper validation of robotic systems.
    """

    def test_case_documentation_template(self):
        """
        Template for documenting test cases in robotics systems.

        Each test case should include:

        1. Purpose: Clear statement of what is being tested
        2. Pre-conditions: State required before test execution
        3. Procedure: Step-by-step execution process
        4. Expected Results: What should happen during test
        5. Pass Criteria: Conditions for test success
        6. Fail Criteria: Conditions for test failure
        7. Post-conditions: State after test execution
        8. Notes: Additional information for test execution

        Example:
        """
        # Test: Joint Position Control Accuracy
        # Purpose: Verify that joint position control achieves desired positions within tolerance
        # Pre-conditions: Robot is powered on, joint controllers are loaded and active
        # Procedure:
        #   1. Send position command to joint
        #   2. Wait for movement to complete
        #   3. Measure actual joint position
        #   4. Compare with commanded position
        # Expected Results: Joint moves to commanded position within tolerance
        # Pass Criteria: Position error < 0.01 radians
        # Fail Criteria: Position error >= 0.01 radians OR timeout occurs
        # Post-conditions: Joint remains at final position
        # Notes: Test should be repeated for multiple joints and positions
        pass

    def test_suite_organization(self):
        """
        Organize tests in a hierarchical structure:

        1. Unit Tests: Individual functions/classes (e.g., IK solvers, controllers)
        2. Integration Tests: Component interactions (e.g., sensor-fusion, perception-action)
        3. System Tests: Full system behavior (e.g., walking, manipulation)
        4. Acceptance Tests: User requirements validation (e.g., task completion)
        5. Regression Tests: Previously fixed bugs (e.g., specific failure modes)
        """
        pass


class JointControllerTest(unittest.TestCase):
    """
    Unit tests for joint controller functionality.

    These tests validate the low-level joint control system of the humanoid robot,
    ensuring that individual joints can be commanded to specific positions with
    adequate accuracy and response characteristics.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment before running any tests in this class.

        Initializes ROS2 context and creates test node for communication with
        the robot's joint controllers.
        """
        rclpy.init()
        cls.test_node = rclpy.create_node('joint_controller_test_node')

        # Create publisher for joint commands
        cls.joint_cmd_pub = cls.test_node.create_publisher(
            JointState, '/joint_commands', 10
        )

        # Create subscriber for joint state feedback
        cls.joint_state_sub = cls.test_node.create_subscription(
            JointState, '/joint_states', cls.joint_state_callback, 10
        )

        # Initialize test state
        cls.current_joint_states = JointState()
        cls.test_executed = False

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the test environment after all tests in this class have run.

        Shuts down ROS2 context and cleans up test resources.
        """
        cls.test_node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def joint_state_callback(cls, msg):
        """
        Callback function for receiving joint state updates.

        Args:
            msg (JointState): Received joint state message
        """
        cls.current_joint_states = msg

    def setUp(self):
        """
        Set up before each individual test method.

        Resets test state to ensure independence between tests.
        """
        self.test_executed = False
        time.sleep(0.1)  # Allow state to stabilize

    def test_single_joint_position_control(self):
        """
        Test that a single joint can be commanded to a specific position.

        Purpose: Validate basic joint position control functionality
        Pre-conditions: Joint controller is active and responsive
        Procedure: Command joint to specific position and verify achievement
        Expected Results: Joint reaches commanded position within tolerance
        Pass Criteria: Position error < 0.01 radians AND no exceptions raised
        Fail Criteria: Position error >= 0.01 radians OR timeout OR exception
        Post-conditions: Joint remains at commanded position
        Notes: Test performed with safety limits active
        """
        # Select a joint to test (e.g., left hip yaw)
        test_joint_name = 'left_hip_yaw'
        target_position = 0.5  # radians

        # Find joint index
        try:
            joint_idx = self.current_joint_states.name.index(test_joint_name)
        except ValueError:
            self.fail(f"Joint {test_joint_name} not found in joint states")

        # Record initial position
        initial_position = self.current_joint_states.position[joint_idx]

        # Create joint command
        cmd_msg = JointState()
        cmd_msg.name = [test_joint_name]
        cmd_msg.position = [target_position]
        cmd_msg.header.stamp = self.test_node.get_clock().now().to_msg()

        # Publish command
        self.joint_cmd_pub.publish(cmd_msg)

        # Wait for movement to complete (with timeout)
        timeout = 5.0  # seconds
        start_time = time.time()
        reached_tolerance = False

        while time.time() - start_time < timeout and not reached_tolerance:
            time.sleep(0.01)  # Poll every 10ms

            # Check current position
            if len(self.current_joint_states.position) > joint_idx:
                current_position = self.current_joint_states.position[joint_idx]
                position_error = abs(current_position - target_position)

                if position_error < 0.01:  # 0.01 radian tolerance
                    reached_tolerance = True

        # Verify results
        self.assertTrue(
            reached_tolerance,
            f"Joint {test_joint_name} did not reach target position within tolerance. "
            f"Target: {target_position}, Actual: {current_position}, Error: {position_error}"
        )

        # Additional validation: ensure joint didn't exceed limits
        self.assertLessEqual(
            abs(current_position), 1.57,  # Joint should not exceed ±90 degrees
            f"Joint {test_joint_name} exceeded position limits: {current_position}"
        )

        self.test_executed = True

    def test_joint_trajectory_execution(self):
        """
        Test that joints can follow a trajectory of positions.

        Purpose: Validate trajectory execution capabilities for coordinated motion
        Pre-conditions: Joint controllers are responsive and coordinated
        Procedure: Send trajectory and verify smooth execution
        Expected Results: Joints follow trajectory smoothly within tolerances
        Pass Criteria: All points achieved within tolerance AND smooth motion
        Fail Criteria: Points missed OR jerky motion OR timeout
        Post-conditions: Joints remain at final trajectory positions
        Notes: Trajectory timing and coordination are critical
        """
        # Define a simple trajectory (triangular wave pattern)
        trajectory_points = [
            0.0,   # Start position
            0.25,  # Peak 1
            0.0,   # Midpoint
            -0.25, # Valley
            0.0    # Return to start
        ]

        test_joint_name = 'left_knee'
        trajectory_duration = 0.5  # seconds per point

        # Execute trajectory point by point
        for i, target_pos in enumerate(trajectory_points):
            # Create joint command
            cmd_msg = JointState()
            cmd_msg.name = [test_joint_name]
            cmd_msg.position = [target_pos]
            cmd_msg.header.stamp = self.test_node.get_clock().now().to_msg()

            # Publish command
            self.joint_cmd_pub.publish(cmd_msg)

            # Wait for this point
            time.sleep(trajectory_duration)

            # Verify position achievement
            try:
                joint_idx = self.current_joint_states.name.index(test_joint_name)
                current_pos = self.current_joint_states.position[joint_idx]
                error = abs(current_pos - target_pos)

                with self.subTest(point=i, target=target_pos, actual=current_pos):
                    self.assertLess(
                        error, 0.02,  # 0.02 radian tolerance for trajectory
                        f"Trajectory point {i} not achieved. "
                        f"Target: {target_pos}, Actual: {current_pos}, Error: {error}"
                    )
            except ValueError:
                self.fail(f"Joint {test_joint_name} not found in joint states during trajectory")

        # Verify final position
        final_idx = self.current_joint_states.name.index(test_joint_name)
        final_pos = self.current_joint_states.position[final_idx]
        final_error = abs(final_pos - trajectory_points[-1])

        self.assertLess(
            final_error, 0.01,
            f"Final trajectory position not maintained. "
            f"Target: {trajectory_points[-1]}, Actual: {final_pos}"
        )

        self.test_executed = True

    def test_joint_limits_enforcement(self):
        """
        Test that joint limits are properly enforced during control.

        Purpose: Validate safety systems that prevent joint damage
        Pre-conditions: Joint limits are configured in controllers
        Procedure: Attempt to command joint beyond limits and verify prevention
        Expected Results: Commands beyond limits are clipped or rejected
        Pass Criteria: Joint positions remain within limits AND no damage occurs
        Fail Criteria: Joint exceeds limits OR controller error OR damage
        Post-conditions: Joint remains in safe position
        Notes: Critical for robot safety and longevity
        """
        test_joint_name = 'left_ankle_pitch'

        # Attempt to command beyond soft limits (±0.5 radians)
        dangerous_positions = [
            -1.0,  # Well beyond lower limit
            1.0    # Well beyond upper limit
        ]

        for dangerous_pos in dangerous_positions:
            # Create joint command
            cmd_msg = JointState()
            cmd_msg.name = [test_joint_name]
            cmd_msg.position = [dangerous_pos]
            cmd_msg.header.stamp = self.test_node.get_clock().now().to_msg()

            # Publish command
            self.joint_cmd_pub.publish(cmd_msg)

            # Wait briefly for controller response
            time.sleep(0.1)

            # Check that actual position is within safe limits
            try:
                joint_idx = self.current_joint_states.name.index(test_joint_name)
                actual_pos = self.current_joint_states.position[joint_idx]

                with self.subTest(target=dangerous_pos, actual=actual_pos):
                    # Verify position is within safe limits (-0.5 to 0.5 radians)
                    self.assertGreaterEqual(
                        actual_pos, -0.51,  # Slightly above limit for tolerance
                        f"Joint limit violated! Dangerous command: {dangerous_pos}, "
                        f"Actual position: {actual_pos}"
                    )

                    self.assertLessEqual(
                        actual_pos, 0.51,  # Slightly below limit for tolerance
                        f"Joint limit violated! Dangerous command: {dangerous_pos}, "
                        f"Actual position: {actual_pos}"
                    )
            except ValueError:
                self.fail(f"Joint {test_joint_name} not found in joint states during limit test")

        self.test_executed = True


class WalkControllerIntegrationTest(unittest.TestCase):
    """
    Integration tests for walking controller and related systems.

    These tests validate the coordination between perception, planning, control,
    and actuation systems during walking behavior. They verify that the complete
    walking pipeline functions correctly.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up integration test environment.

        Creates multiple nodes to test system interactions.
        """
        rclpy.init()
        cls.test_node = rclpy.create_node('walk_controller_integration_test')

        # Publishers for commanding the walking system
        cls.cmd_vel_pub = cls.test_node.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # Subscribers for monitoring system state
        cls.odom_sub = cls.test_node.create_subscription(
            Odometry, '/odom', cls.odom_callback, 10
        )
        cls.imu_sub = cls.test_node.create_subscription(
            Imu, '/imu/data', cls.imu_callback, 10
        )
        cls.zmp_sub = cls.test_node.create_subscription(
            PointStamped, '/zmp_estimator/zmp', cls.zmp_callback, 10
        )

        # Initialize test data
        cls.odom_data = None
        cls.imu_data = None
        cls.zmp_data = None

    @classmethod
    def tearDownClass(cls):
        """
        Clean up integration test environment.
        """
        cls.test_node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def odom_callback(cls, msg):
        """Store odometry data for analysis."""
        cls.odom_data = msg

    @classmethod
    def imu_callback(cls, msg):
        """Store IMU data for analysis."""
        cls.imu_data = msg

    @classmethod
    def zmp_callback(cls, msg):
        """Store ZMP data for analysis."""
        cls.zmp_data = msg

    def test_walk_forward_stability(self):
        """
        Test that the robot can walk forward stably.

        Purpose: Validate complete walking pipeline under forward motion
        Pre-conditions: Robot is standing in stable position
        Procedure: Command forward motion and monitor stability metrics
        Expected Results: Robot walks forward while maintaining balance
        Pass Criteria: Forward progress > 0.5m AND ZMP within support polygon
                      AND no falls AND reasonable orientation
        Fail Criteria: Robot falls OR ZMP outside support OR no forward progress
        Post-conditions: Robot stops safely in stable position
        Notes: Monitor for 10 seconds of forward walking
        """
        # Command forward velocity
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.3  # 0.3 m/s forward
        cmd_msg.angular.z = 0.0  # No turning

        # Record initial state
        initial_x = self.odom_data.pose.pose.position.x if self.odom_data else 0.0
        test_duration = 10.0  # seconds
        start_time = time.time()

        # Execute walking command
        while time.time() - start_time < test_duration:
            self.cmd_vel_pub.publish(cmd_msg)
            time.sleep(0.1)  # 10Hz command rate

            # Monitor stability metrics during walk
            if self.imu_data:
                # Check orientation is reasonable (not falling)
                orientation = self.imu_data.orientation
                roll, pitch = self._quat_to_rpy(orientation)

                # If robot is tilting too much, it's probably falling
                self.assertLess(
                    abs(roll), 0.5,  # 0.5 rad ~ 28 degrees
                    "Robot is tilting excessively - possible fall detected"
                )
                self.assertLess(
                    abs(pitch), 0.5,
                    "Robot is pitching excessively - possible fall detected"
                )

            # Monitor ZMP if available
            if self.zmp_data and self.support_polygon:
                zmp_in_polygon = self._is_point_in_polygon(
                    (self.zmp_data.point.x, self.zmp_data.point.y),
                    self.support_polygon
                )

                self.assertTrue(
                    zmp_in_polygon,
                    "ZMP outside support polygon - robot is unstable"
                )

        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        # Verify forward progress
        final_x = self.odom_data.pose.pose.position.x if self.odom_data else initial_x
        forward_progress = final_x - initial_x

        self.assertGreater(
            forward_progress, 0.5,  # Should have moved at least 0.5m forward
            f"Insufficient forward progress: {forward_progress}m, expected > 0.5m"
        )

        # Verify final stability
        time.sleep(1.0)  # Allow robot to settle
        if self.imu_data:
            final_orientation = self.imu_data.orientation
            final_roll, final_pitch = self._quat_to_rpy(final_orientation)

            self.assertLess(
                abs(final_roll), 0.2,
                "Robot not stable after walking stopped"
            )
            self.assertLess(
                abs(final_pitch), 0.2,
                "Robot not stable after walking stopped"
            )

    def _quat_to_rpy(self, quaternion):
        """
        Convert quaternion to roll-pitch-yaw angles.

        Args:
            quaternion: geometry_msgs/Quaternion message

        Returns:
            tuple: (roll, pitch) in radians (ignoring yaw)
        """
        # Simplified conversion (ignoring singularity handling for testing)
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        return roll, pitch

    def _is_point_in_polygon(self, point, polygon):
        """
        Check if a 2D point is inside a polygon.

        Args:
            point: (x, y) tuple representing the point
            polygon: List of (x, y) tuples representing polygon vertices

        Returns:
            bool: True if point is inside polygon, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


def run_tests():
    """
    Execute all tests in the module.

    Discovers and runs all test cases defined in this module, providing
    comprehensive validation of the robotic systems.
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)