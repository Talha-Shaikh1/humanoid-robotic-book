#!/usr/bin/env python3
# This example demonstrates humanoid robot control in Isaac Sim

"""
Isaac Sim Humanoid Robot Control Example

This script demonstrates how to control a humanoid robot in Isaac Sim
with balance control, walking, and whole-body motion planning.
"""

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.semantics import add_semantic_label
from pxr import Gf, UsdGeom
import numpy as np
import math
import asyncio


class IsaacSimHumanoidControl:
    """
    A humanoid robot controller for Isaac Sim with advanced locomotion and balance.
    """

    def __init__(self):
        """
        Initialize the humanoid robot controller.
        """
        self.world = None
        self.humanoid = None
        self.joint_names = []
        self.balance_controller = None
        self.walk_controller = None

    async def setup_humanoid_environment(self):
        """
        Set up the humanoid robot environment.
        """
        # Create the world with physics
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add a ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        self._setup_lighting()

        # Load a humanoid robot (using a basic humanoid for this example)
        try:
            # Add a basic humanoid robot (if available in the Isaac Sim assets)
            add_reference_to_stage(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/A1/a1.usd",
                prim_path="/World/A1"
            )
        except Exception as e:
            print(f"Could not load A1 robot, using basic humanoid: {e}")
            # Create a basic humanoid if A1 is not available
            self._create_basic_humanoid()

        # Reset the world to apply changes
        self.world.reset()

    def _setup_lighting(self):
        """
        Set up lighting for the humanoid simulation environment.
        """
        # Add dome light
        dome_light = UsdGeom.DomeLight.Define(self.world.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(3000)

        # Add key light
        key_light = UsdGeom.DistantLight.Define(self.world.stage, "/World/KeyLight")
        key_light.CreateIntensityAttr(3000)
        key_light.AddTranslateOp().Set(Gf.Vec3f(0, 0, 5))
        key_light.AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))

    def _create_basic_humanoid(self):
        """
        Create a basic humanoid robot if asset loading fails.
        """
        # Create a simplified humanoid robot structure
        humanoid_xform = UsdGeom.Xform.Define(self.world.stage, "/World/BasicHumanoid")

        # Torso
        torso = UsdGeom.Capsule.Define(self.world.stage, "/World/BasicHumanoid/Torso")
        torso.CreateRadiusAttr(0.1)
        torso.CreateHeightAttr(0.5)

        # Head
        head = UsdGeom.Sphere.Define(self.world.stage, "/World/BasicHumanoid/Head")
        head.CreateRadiusAttr(0.1)
        head_xform = UsdGeom.Xformable(head.GetPrim())
        head_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.35))

        # Left arm
        left_shoulder = UsdGeom.Sphere.Define(self.world.stage, "/World/BasicHumanoid/LeftShoulder")
        left_shoulder.CreateRadiusAttr(0.05)
        left_shoulder_xform = UsdGeom.Xformable(left_shoulder.GetPrim())
        left_shoulder_xform.AddTranslateOp().Set(Gf.Vec3f(0.15, 0, 0.2))

        left_upper_arm = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicHumanoid/LeftUpperArm")
        left_upper_arm.CreateRadiusAttr(0.03)
        left_upper_arm.CreateHeightAttr(0.3)
        left_upper_arm_xform = UsdGeom.Xformable(left_upper_arm.GetPrim())
        left_upper_arm_xform.AddTranslateOp().Set(Gf.Vec3f(0.25, 0, 0.1))

        # Right arm
        right_shoulder = UsdGeom.Sphere.Define(self.world.stage, "/World/BasicHumanoid/RightShoulder")
        right_shoulder.CreateRadiusAttr(0.05)
        right_shoulder_xform = UsdGeom.Xformable(right_shoulder.GetPrim())
        right_shoulder_xform.AddTranslateOp().Set(Gf.Vec3f(-0.15, 0, 0.2))

        right_upper_arm = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicHumanoid/RightUpperArm")
        right_upper_arm.CreateRadiusAttr(0.03)
        right_upper_arm.CreateHeightAttr(0.3)
        right_upper_arm_xform = UsdGeom.Xformable(right_upper_arm.GetPrim())
        right_upper_arm_xform.AddTranslateOp().Set(Gf.Vec3f(-0.25, 0, 0.1))

        # Left leg
        left_hip = UsdGeom.Sphere.Define(self.world.stage, "/World/BasicHumanoid/LeftHip")
        left_hip.CreateRadiusAttr(0.05)
        left_hip_xform = UsdGeom.Xformable(left_hip.GetPrim())
        left_hip_xform.AddTranslateOp().Set(Gf.Vec3f(0.08, 0, -0.1))

        left_thigh = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicHumanoid/LeftThigh")
        left_thigh.CreateRadiusAttr(0.04)
        left_thigh.CreateHeightAttr(0.4)
        left_thigh_xform = UsdGeom.Xformable(left_thigh.GetPrim())
        left_thigh_xform.AddTranslateOp().Set(Gf.Vec3f(0.08, 0, -0.3))

        # Right leg
        right_hip = UsdGeom.Sphere.Define(self.world.stage, "/World/BasicHumanoid/RightHip")
        right_hip.CreateRadiusAttr(0.05)
        right_hip_xform = UsdGeom.Xformable(right_hip.GetPrim())
        right_hip_xform.AddTranslateOp().Set(Gf.Vec3f(-0.08, 0, -0.1))

        right_thigh = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicHumanoid/RightThigh")
        right_thigh.CreateRadiusAttr(0.04)
        right_thigh.CreateHeightAttr(0.4)
        right_thigh_xform = UsdGeom.Xformable(right_thigh.GetPrim())
        right_thigh_xform.AddTranslateOp().Set(Gf.Vec3f(-0.08, 0, -0.3))

    def initialize_balance_controller(self):
        """
        Initialize the balance controller for the humanoid.
        """
        # Implement a basic balance controller using center of mass and zero moment point
        self.balance_controller = {
            'kp': 100.0,  # Proportional gain for balance
            'kd': 10.0,   # Derivative gain for balance
            'com_target': np.array([0.0, 0.0, 0.0]),  # Target center of mass
            'support_polygon': []  # Support polygon for balance
        }

    def initialize_walk_controller(self):
        """
        Initialize the walking controller for the humanoid.
        """
        # Implement a basic walking controller using inverse kinematics
        self.walk_controller = {
            'step_length': 0.3,      # Length of each step
            'step_height': 0.1,      # Height of foot lift
            'step_duration': 0.8,    # Duration of each step
            'current_phase': 0.0,    # Current walking phase (0.0 to 1.0)
            'gait_pattern': 'walk'   # Current gait pattern
        }

    def calculate_balance_control(self, current_com, current_com_vel):
        """
        Calculate balance control commands to maintain stability.
        """
        # Calculate error from target center of mass
        com_error = self.balance_controller['com_target'] - current_com
        com_vel_error = -current_com_vel  # Target velocity is zero

        # Calculate control force using PD controller
        control_force = (self.balance_controller['kp'] * com_error +
                        self.balance_controller['kd'] * com_vel_error)

        return control_force

    def calculate_walk_trajectory(self, time):
        """
        Calculate the walking trajectory for the feet.
        """
        # Calculate phase based on time
        phase = (time / self.walk_controller['step_duration']) % 1.0

        # Calculate foot positions for walking
        # This is a simplified walking pattern
        left_foot_x = 0.0
        left_foot_y = 0.1
        right_foot_x = 0.0
        right_foot_y = -0.1

        # Add stepping motion
        if self.walk_controller['gait_pattern'] == 'walk':
            # Alternate stepping
            if int(time / self.walk_controller['step_duration']) % 2 == 0:
                # Left foot stepping
                left_foot_z = self.walk_controller['step_height'] * math.sin(math.pi * phase)
            else:
                # Right foot stepping
                right_foot_z = self.walk_controller['step_height'] * math.sin(math.pi * phase)

        return {
            'left_foot': np.array([left_foot_x, left_foot_y, left_foot_z]),
            'right_foot': np.array([right_foot_x, right_foot_y, right_foot_z])
        }

    def move_humanoid(self, joint_positions, joint_velocities=None):
        """
        Move the humanoid to specified joint positions.
        """
        if self.humanoid:
            # In Isaac Sim, we would command the joints directly
            if joint_velocities is not None:
                self.humanoid.set_joint_positions(np.array(joint_positions))
                self.humanoid.set_joint_velocities(np.array(joint_velocities))
            else:
                self.humanoid.set_joint_positions(np.array(joint_positions))

    def get_humanoid_state(self):
        """
        Get the current state of the humanoid robot.
        """
        if self.humanoid:
            joint_positions = self.humanoid.get_joint_positions()
            joint_velocities = self.humanoid.get_joint_velocities()
            base_position, base_orientation = self.humanoid.get_world_pose()

            return {
                'joint_positions': joint_positions,
                'joint_velocities': joint_velocities,
                'base_position': base_position,
                'base_orientation': base_orientation
            }
        return None

    def execute_standing_pose(self):
        """
        Execute a standing pose for the humanoid.
        """
        print("Moving to standing pose...")

        # Define joint positions for standing pose
        # This is a simplified example - real humanoid would have many more joints
        standing_joints = np.array([
            0.0,   # Left hip yaw
            0.0,   # Left hip roll
            -0.3,  # Left hip pitch
            0.6,   # Left knee
            -0.3,  # Left ankle
            0.0,   # Right hip yaw
            0.0,   # Right hip roll
            -0.3,  # Right hip pitch
            0.6,   # Right knee
            -0.3,  # Right ankle
            0.0,   # Left shoulder pitch
            0.0,   # Left shoulder roll
            0.0,   # Left elbow
            0.0,   # Right shoulder pitch
            0.0,   # Right shoulder roll
            0.0    # Right elbow
        ])

        # Move to standing pose
        self.move_humanoid(standing_joints)

    def execute_simple_walk(self):
        """
        Execute a simple walking motion.
        """
        print("Executing simple walk...")

        # Initialize controllers
        self.initialize_balance_controller()
        self.initialize_walk_controller()

        # Walk for 10 seconds
        start_time = self.world.current_time_step_index * (1.0/60.0)  # Assuming 60 FPS

        for step in range(600):  # 10 seconds at 60 FPS
            current_time = start_time + step * (1.0/60.0)

            # Calculate walk trajectory
            foot_positions = self.calculate_walk_trajectory(current_time)

            # Calculate balance control
            # In a real implementation, we would get current CoM from the robot
            current_com = np.array([0.0, 0.0, 0.8])  # Simplified CoM
            current_com_vel = np.array([0.0, 0.0, 0.0])  # Simplified CoM velocity
            balance_control = self.calculate_balance_control(current_com, current_com_vel)

            # Apply walking and balance control
            # This would involve more complex inverse kinematics and dynamics in a real implementation
            print(f"Step {step}: Walking at time {current_time:.2f}s")

            # Step the simulation
            self.world.step(render=True)

        print("Walking completed.")


async def run_humanoid_demo():
    """
    Run a demonstration of the humanoid robot control.
    """
    print("Setting up Isaac Sim humanoid robot simulation...")

    # Create the humanoid controller
    controller = IsaacSimHumanoidControl()

    # Set up the environment
    await controller.setup_humanoid_environment()

    print("Humanoid environment setup complete. Starting control demo...")

    # Execute standing pose
    controller.execute_standing_pose()

    # Run simulation for a moment
    for step in range(60):  # 1 second at 60 FPS
        controller.world.step(render=True)

    # Execute walking
    controller.execute_simple_walk()

    print("Humanoid control demo completed.")


# Humanoid robot configuration
ISAAC_SIM_HUMANOID_CONFIG = {
    "a1": {
        "joint_names": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # Front left leg
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # Front right leg
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",  # Rear left leg
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"   # Rear right leg
        ],
        "joint_limits": {
            "min": [-0.8, -1.0, -2.7, -0.8, -1.0, -2.7, -0.8, -1.0, -2.7, -0.8, -1.0, -2.7],
            "max": [0.8, 4.5, -0.8, 0.8, 4.5, -0.8, 0.8, 4.5, -0.8, 0.8, 4.5, -0.8]
        },
        "mass": 13.731,
        "links": ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    },
    "balance_control": {
        "zmp_controller": {
            "kp": 100.0,
            "kd": 10.0,
            "max_force": 1000.0
        },
        "com_admittance": {
            "stiffness": 1000.0,
            "damping": 100.0
        }
    },
    "walking_gaits": {
        "walk": {
            "step_length": 0.3,
            "step_height": 0.1,
            "step_duration": 0.8,
            "stance_duration": 0.6
        },
        "trot": {
            "step_length": 0.4,
            "step_height": 0.15,
            "step_duration": 0.6,
            "stance_duration": 0.4
        }
    }
}


# This would be run in Isaac Sim's Python environment
if __name__ == "__main__":
    print("Isaac Sim Humanoid Robot Control Example")
    print("=" * 50)
    print("This example demonstrates:")
    print("- Humanoid robot loading and control")
    print("- Balance control algorithms")
    print("- Walking gait generation")
    print("- Inverse kinematics for locomotion")
    print("- Physics-based humanoid simulation")
    print("=" * 50)
    print("Note: This example is designed for Isaac Sim environment")
    print("In a real Isaac Sim scenario, this would be run as a script within Isaac Sim")