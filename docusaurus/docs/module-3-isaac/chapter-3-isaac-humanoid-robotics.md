---
title: Isaac Sim for Humanoid Robotics
description: Specialized applications of Isaac Sim for humanoid robotics development
sidebar_position: 3
learning_outcomes:
  - Implement humanoid robot models in Isaac Sim
  - Configure advanced control systems for bipedal locomotion
  - Simulate complex humanoid behaviors and interactions
  - Integrate AI for humanoid perception and control
---

# Isaac Sim for Humanoid Robotics: Specialized Applications and Control Systems

## Purpose
This chapter focuses on specialized applications of NVIDIA Isaac Sim for humanoid robotics development. You'll learn how to model complex humanoid robots, implement advanced control systems for bipedal locomotion, simulate human-like behaviors, and integrate AI for perception and control in humanoid systems.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Implement humanoid robot models in Isaac Sim
- Configure advanced control systems for bipedal locomotion
- Simulate complex humanoid behaviors and interactions
- Integrate AI for humanoid perception and control

## Humanoid Robot Modeling in Isaac Sim

### Advanced Humanoid Kinematics

```python
# Advanced humanoid robot model with proper kinematics
import omni
from pxr import UsdGeom, UsdPhysics, Gf, UsdSkel
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.robots.robot import Robot
import numpy as np

class HumanoidRobotModel:
    def __init__(self, robot_name="/World/HumanoidRobot"):
        self.robot_name = robot_name
        self.stage = omni.usd.get_context().get_stage()
        self.joint_names = []
        self.link_names = []

    def create_humanoid_skeleton(self):
        """
        Create a humanoid skeleton with proper kinematic chains
        """
        # Create the root (pelvis)
        pelvis_path = f"{self.robot_name}/pelvis"
        pelvis = prim_utils.create_prim(
            prim_path=pelvis_path,
            prim_type="Xform",
            position=[0.0, 0.0, 0.8]  # Start above ground for biped
        )

        # Add skeleton API for animation
        skel_api = UsdSkel.Skeleton.Define(self.stage, f"{self.robot_name}/Skeleton")

        # Define joint hierarchy
        joint_names = [
            "pelvis", "torso", "neck", "head",
            "left_hip", "left_knee", "left_ankle", "left_foot",
            "right_hip", "right_knee", "right_ankle", "right_foot",
            "left_shoulder", "left_elbow", "left_wrist", "left_hand",
            "right_shoulder", "right_elbow", "right_wrist", "right_hand"
        ]

        skel_api.CreateJointsAttr().Set(joint_names)

        # Define rest poses for joints
        rest_transforms = []
        for i, joint_name in enumerate(joint_names):
            # Default rest pose (identity transform)
            rest_transforms.append(Gf.Matrix4d(1))

        skel_api.CreateRestTransformsAttr().Set(rest_transforms)

        # Create physical links
        self._create_physical_links()

        # Create joints between links
        self._create_joints()

        print(f"Humanoid skeleton created at {self.robot_name}")
        return pelvis

    def _create_physical_links(self):
        """
        Create physical links for the humanoid robot
        """
        # Pelvis (torso base)
        pelvis_path = f"{self.robot_name}/pelvis"
        pelvis = prim_utils.create_prim(
            prim_path=pelvis_path,
            prim_type="Capsule",
            position=[0.0, 0.0, 0.8],
            attributes={"radius": 0.1, "height": 0.2}
        )

        # Torso
        torso_path = f"{self.robot_name}/torso"
        torso = prim_utils.create_prim(
            prim_path=torso_path,
            prim_type="Capsule",
            position=[0.0, 0.0, 1.1],
            attributes={"radius": 0.1, "height": 0.4}
        )

        # Head
        head_path = f"{self.robot_name}/head"
        head = prim_utils.create_prim(
            prim_path=head_path,
            prim_type="Sphere",
            position=[0.0, 0.0, 1.5],
            attributes={"radius": 0.15}
        )

        # Left leg components
        left_thigh = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/left_thigh",
            prim_type="Capsule",
            position=[-0.1, 0.0, 0.6],
            attributes={"radius": 0.08, "height": 0.35}
        )

        left_shin = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/left_shin",
            prim_type="Capsule",
            position=[-0.1, 0.0, 0.25],
            attributes={"radius": 0.07, "height": 0.35}
        )

        left_foot = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/left_foot",
            prim_type="Box",
            position=[-0.1, 0.0, 0.08],
            attributes={"size": 0.15}
        )

        # Right leg components (similar to left)
        right_thigh = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/right_thigh",
            prim_type="Capsule",
            position=[0.1, 0.0, 0.6],
            attributes={"radius": 0.08, "height": 0.35}
        )

        right_shin = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/right_shin",
            prim_type="Capsule",
            position=[0.1, 0.0, 0.25],
            attributes={"radius": 0.07, "height": 0.35}
        )

        right_foot = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/right_foot",
            prim_type="Box",
            position=[0.1, 0.0, 0.08],
            attributes={"size": 0.15}
        )

        # Left arm components
        left_upper_arm = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/left_upper_arm",
            prim_type="Capsule",
            position=[-0.2, 0.0, 1.2],
            attributes={"radius": 0.06, "height": 0.25}
        )

        left_lower_arm = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/left_lower_arm",
            prim_type="Capsule",
            position=[-0.35, 0.0, 1.2],
            attributes={"radius": 0.05, "height": 0.25}
        )

        left_hand = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/left_hand",
            prim_type="Sphere",
            position=[-0.45, 0.0, 1.2],
            attributes={"radius": 0.05}
        )

        # Right arm components (similar to left)
        right_upper_arm = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/right_upper_arm",
            prim_type="Capsule",
            position=[0.2, 0.0, 1.2],
            attributes={"radius": 0.06, "height": 0.25}
        )

        right_lower_arm = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/right_lower_arm",
            prim_type="Capsule",
            position=[0.35, 0.0, 1.2],
            attributes={"radius": 0.05, "height": 0.25}
        )

        right_hand = prim_utils.create_prim(
            prim_path=f"{self.robot_name}/right_hand",
            prim_type="Sphere",
            position=[0.45, 0.0, 1.2],
            attributes={"radius": 0.05}
        )

    def _create_joints(self):
        """
        Create joints connecting the humanoid links
        """
        # Create ball joints for shoulders and hips (3 DOF)
        self._create_ball_joint(
            parent_path=f"{self.robot_name}/pelvis",
            child_path=f"{self.robot_name}/torso",
            joint_name="torso_joint",
            position=[0.0, 0.0, 0.1]
        )

        # Hip joints (left and right)
        self._create_ball_joint(
            parent_path=f"{self.robot_name}/pelvis",
            child_path=f"{self.robot_name}/left_thigh",
            joint_name="left_hip",
            position=[-0.1, 0.0, 0.0]
        )

        self._create_ball_joint(
            parent_path=f"{self.robot_name}/pelvis",
            child_path=f"{self.robot_name}/right_thigh",
            joint_name="right_hip",
            position=[0.1, 0.0, 0.0]
        )

        # Knee joints (revolute, single DOF)
        self._create_revolute_joint(
            parent_path=f"{self.robot_name}/left_thigh",
            child_path=f"{self.robot_name}/left_shin",
            joint_name="left_knee",
            position=[-0.1, 0.0, 0.425],
            axis=[1, 0, 0]  # Rotate around X-axis
        )

        self._create_revolute_joint(
            parent_path=f"{self.robot_name}/right_thigh",
            child_path=f"{self.robot_name}/right_shin",
            joint_name="right_knee",
            position=[0.1, 0.0, 0.425],
            axis=[1, 0, 0]  # Rotate around X-axis
        )

        # Ankle joints
        self._create_ball_joint(
            parent_path=f"{self.robot_name}/left_shin",
            child_path=f"{self.robot_name}/left_foot",
            joint_name="left_ankle",
            position=[-0.1, 0.0, 0.075]
        )

        self._create_ball_joint(
            parent_path=f"{self.robot_name}/right_shin",
            child_path=f"{self.robot_name}/right_foot",
            joint_name="right_ankle",
            position=[0.1, 0.0, 0.075]
        )

        # Shoulder joints
        self._create_ball_joint(
            parent_path=f"{self.robot_name}/torso",
            child_path=f"{self.robot_name}/left_upper_arm",
            joint_name="left_shoulder",
            position=[-0.15, 0.0, 0.0]
        )

        self._create_ball_joint(
            parent_path=f"{self.robot_name}/torso",
            child_path=f"{self.robot_name}/right_upper_arm",
            joint_name="right_shoulder",
            position=[0.15, 0.0, 0.0]
        )

        # Elbow joints
        self._create_revolute_joint(
            parent_path=f"{self.robot_name}/left_upper_arm",
            child_path=f"{self.robot_name}/left_lower_arm",
            joint_name="left_elbow",
            position=[-0.275, 0.0, 0.0],
            axis=[0, 1, 0]  # Rotate around Y-axis
        )

        self._create_revolute_joint(
            parent_path=f"{self.robot_name}/right_upper_arm",
            child_path=f"{self.robot_name}/right_lower_arm",
            joint_name="right_elbow",
            position=[0.275, 0.0, 0.0],
            axis=[0, 1, 0]  # Rotate around Y-axis
        )

    def _create_ball_joint(self, parent_path, child_path, joint_name, position):
        """
        Create a ball joint (3 DOF spherical joint)
        """
        joint_path = f"{parent_path}/{joint_name}"

        # Create the joint prim
        joint_prim = prim_utils.create_prim(
            prim_path=joint_path,
            prim_type="Joint",
            position=position
        )

        # Apply joint API
        joint_api = UsdPhysics.Joint.Apply(
            self.stage.GetPrimAtPath(joint_path)
        )

        # Set joint type to ball (spherical)
        joint_api.CreateJointTypeAttr("spherical")

        # Set joint limits (for ball joints, this sets the range for each axis)
        limit_api = UsdPhysics.LimitAPI.Apply(joint_api.GetPrim())
        limit_api.CreateLowerLimitAttr(-1.57)  # -90 degrees
        limit_api.CreateUpperLimitAttr(1.57)   # 90 degrees

    def _create_revolute_joint(self, parent_path, child_path, joint_name, position, axis):
        """
        Create a revolute joint (1 DOF rotational joint)
        """
        joint_path = f"{parent_path}/{joint_name}"

        # Create the joint prim
        joint_prim = prim_utils.create_prim(
            prim_path=joint_path,
            prim_type="Joint",
            position=position
        )

        # Apply joint API
        joint_api = UsdPhysics.Joint.Apply(
            self.stage.GetPrimAtPath(joint_path)
        )

        # Set joint type to revolute
        joint_api.CreateJointTypeAttr("revolute")

        # Set the rotation axis
        joint_api.CreateAxisAttr(axis)

        # Set joint limits
        limit_api = UsdPhysics.LimitAPI.Apply(joint_api.GetPrim())
        limit_api.CreateLowerLimitAttr(-2.0)  # -114 degrees
        limit_api.CreateUpperLimitAttr(2.0)   # 114 degrees

# Usage example
def create_humanoid_robot():
    robot = HumanoidRobotModel("/World/MyHumanoid")
    robot.create_humanoid_skeleton()
    print("Humanoid robot model created successfully")
    return robot
```

### Physics Properties for Humanoid Models

```python
# Physics configuration for humanoid robots
import omni
from pxr import UsdPhysics, Gf
import omni.isaac.core.utils.prims as prim_utils

class HumanoidPhysicsConfig:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.stage = omni.usd.get_context().get_stage()

    def configure_mass_properties(self):
        """
        Configure realistic mass properties for humanoid links
        """
        # Define mass for each body part (approximate human values)
        mass_config = {
            "pelvis": 10.0,      # ~10kg
            "torso": 25.0,       # ~25kg
            "head": 5.0,         # ~5kg
            "left_thigh": 10.0,  # ~10kg
            "right_thigh": 10.0, # ~10kg
            "left_shin": 5.0,    # ~5kg
            "right_shin": 5.0,   # ~5kg
            "left_foot": 1.5,    # ~1.5kg
            "right_foot": 1.5,   # ~1.5kg
            "left_upper_arm": 2.5,   # ~2.5kg
            "right_upper_arm": 2.5,  # ~2.5kg
            "left_lower_arm": 1.5,   # ~1.5kg
            "right_lower_arm": 1.5,  # ~1.5kg
            "left_hand": 0.5,    # ~0.5kg
            "right_hand": 0.5    # ~0.5kg
        }

        # Apply mass properties to each link
        for link_name, mass in mass_config.items():
            link_path = f"{self.robot_name}/{link_name}"
            link_prim = self.stage.GetPrimAtPath(link_path)

            if link_prim.IsValid():
                # Apply rigid body API
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(link_prim)

                # Apply mass API
                mass_api = UsdPhysics.MassAPI.Apply(link_prim)
                mass_api.CreateMassAttr().Set(mass)

                # Calculate and set inertia tensor (simplified)
                self._set_inertia_tensor(link_prim, mass, link_name)

    def _set_inertia_tensor(self, prim, mass, link_name):
        """
        Set approximate inertia tensor for humanoid links
        """
        # Approximate inertia based on link geometry
        if "thigh" in link_name or "shin" in link_name:
            # Cylindrical approximation for legs
            inertia = Gf.Vec3f(0.1 * mass, 0.1 * mass, 0.05 * mass)
        elif "arm" in link_name:
            # Cylindrical approximation for arms
            inertia = Gf.Vec3f(0.05 * mass, 0.05 * mass, 0.02 * mass)
        elif link_name == "torso":
            # Box approximation for torso
            inertia = Gf.Vec3f(0.2 * mass, 0.2 * mass, 0.1 * mass)
        elif link_name == "head":
            # Spherical approximation for head
            inertia = Gf.Vec3f(0.05 * mass, 0.05 * mass, 0.05 * mass)
        else:
            # Default approximation
            inertia = Gf.Vec3f(0.05 * mass, 0.05 * mass, 0.05 * mass)

        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
        # Note: Isaac Sim handles inertia calculation automatically in many cases

    def configure_friction_properties(self):
        """
        Configure friction properties for realistic interaction
        """
        # Set friction for different parts
        friction_config = {
            "left_foot": {"static": 0.8, "dynamic": 0.6},
            "right_foot": {"static": 0.8, "dynamic": 0.6},
            "left_hand": {"static": 0.5, "dynamic": 0.4},
            "right_hand": {"static": 0.5, "dynamic": 0.4}
        }

        for link_name, friction_props in friction_config.items():
            link_path = f"{self.robot_name}/{link_name}"
            link_prim = self.stage.GetPrimAtPath(link_path)

            if link_prim.IsValid():
                # Apply friction API
                friction_api = UsdPhysics.MaterialAPI.Apply(link_prim)
                friction_api.CreateStaticFrictionAttr().Set(friction_props["static"])
                friction_api.CreateDynamicFrictionAttr().Set(friction_props["dynamic"])
                friction_api.CreateRestitutionAttr().Set(0.1)  # Low restitution for feet

# Usage example
def configure_humanoid_physics(robot_name):
    physics_config = HumanoidPhysicsConfig(robot_name)
    physics_config.configure_mass_properties()
    physics_config.configure_friction_properties()
    print("Humanoid physics properties configured successfully")
```

## Bipedal Locomotion Control Systems

### Zero Moment Point (ZMP) Control

```python
# ZMP-based balance control for humanoid robots
import numpy as np
from scipy import integrate
import math

class ZMPController:
    def __init__(self, robot_height=0.8, sampling_time=0.01):
        self.robot_height = robot_height  # Height of COM above ground
        self.sampling_time = sampling_time
        self.gravity = 9.81

        # Current state
        self.com_position = np.array([0.0, 0.0, robot_height])
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])

        # Desired ZMP (initialized to zero)
        self.desired_zmp = np.array([0.0, 0.0])

        # Support polygon (simplified as rectangle under feet)
        self.support_polygon = self._define_support_polygon()

        # Gains for ZMP control
        self.kp = 10.0  # Proportional gain
        self.kd = 2.0   # Derivative gain

    def _define_support_polygon(self):
        """
        Define the support polygon based on foot positions
        Simplified as a rectangle between feet
        """
        # For now, assume feet are at fixed positions
        # In a real implementation, this would come from robot state
        foot_separation = 0.2  # 20cm between feet
        foot_size = 0.15       # 15cm foot length

        # Define support polygon as rectangle
        support_polygon = {
            'min_x': -foot_size/2,
            'max_x': foot_size/2,
            'min_y': -foot_separation/2,
            'max_y': foot_separation/2
        }

        return support_polygon

    def update_state(self, com_pos, com_vel, com_acc):
        """
        Update center of mass state
        """
        self.com_position = np.array(com_pos)
        self.com_velocity = np.array(com_vel)
        self.com_acceleration = np.array(com_acc)

    def compute_zmp(self):
        """
        Compute actual ZMP from current state
        ZMP_x = CoM_x - (CoM_height * CoM_acc_x) / gravity
        ZMP_y = CoM_y - (CoM_height * CoM_acc_y) / gravity
        """
        zmp_x = self.com_position[0] - (self.robot_height * self.com_acceleration[0]) / self.gravity
        zmp_y = self.com_position[1] - (self.robot_height * self.com_acceleration[1]) / self.gravity

        return np.array([zmp_x, zmp_y])

    def is_stable(self, zmp=None):
        """
        Check if the current ZMP is within the support polygon
        """
        if zmp is None:
            zmp = self.compute_zmp()

        is_in_polygon = (
            self.support_polygon['min_x'] <= zmp[0] <= self.support_polygon['max_x'] and
            self.support_polygon['min_y'] <= zmp[1] <= self.support_polygon['max_y']
        )

        return is_in_polygon

    def compute_desired_com_trajectory(self, target_zmp):
        """
        Compute desired CoM trajectory to achieve target ZMP
        Uses inverted pendulum model
        """
        # Simplified approach: use PD control to drive ZMP error to zero
        current_zmp = self.compute_zmp()
        zmp_error = target_zmp - current_zmp

        # Desired CoM acceleration to correct ZMP error
        desired_com_acc_x = -self.gravity / self.robot_height * (zmp_error[0])
        desired_com_acc_y = -self.gravity / self.robot_height * (zmp_error[1])

        # Add PD control terms
        desired_com_acc_x += self.kp * zmp_error[0] + self.kd * (0 - self.com_velocity[0])
        desired_com_acc_y += self.kp * zmp_error[1] + self.kd * (0 - self.com_velocity[1])

        return np.array([desired_com_acc_x, desired_com_acc_y, 0.0])

    def generate_footstep_plan(self, walk_distance, step_length=0.3, step_width=0.2):
        """
        Generate a simple footstep plan for walking
        """
        num_steps = int(walk_distance / step_length)
        footsteps = []

        # Start with left foot forward
        for i in range(num_steps):
            # Left foot step
            if i % 2 == 0:
                x = (i + 1) * step_length
                y = step_width / 2
                footsteps.append(('left', x, y))
            # Right foot step
            else:
                x = (i + 1) * step_length
                y = -step_width / 2
                footsteps.append(('right', x, y))

        return footsteps

    def compute_foot_trajectory(self, start_pos, end_pos, step_height=0.1, steps=20):
        """
        Compute smooth trajectory for foot movement
        """
        trajectory = []

        for i in range(steps + 1):
            t = i / steps  # Normalized time (0 to 1)

            # Cubic interpolation for smooth movement
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])

            # Parabolic lift for foot
            z_offset = step_height * math.sin(math.pi * t)  # Lift foot in arc

            z = start_pos[2] + t * (end_pos[2] - start_pos[2]) + z_offset

            trajectory.append(np.array([x, y, z]))

        return trajectory

# Usage example
def setup_zmp_controller():
    zmp_controller = ZMPController(robot_height=0.8)
    print("ZMP controller initialized successfully")
    return zmp_controller
```

### Inverse Kinematics for Humanoid Control

```python
# Inverse kinematics for humanoid robot control
import numpy as np
from scipy.optimize import minimize
import math

class HumanoidIKSolver:
    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.link_lengths = self._get_link_lengths()

    def _get_link_lengths(self):
        """
        Get link lengths from robot configuration
        """
        # Simplified link lengths based on humanoid model
        lengths = {
            'upper_leg': 0.35,    # Thigh length
            'lower_leg': 0.35,    # Shin length
            'upper_arm': 0.25,    # Upper arm length
            'lower_arm': 0.25     # Lower arm length
        }
        return lengths

    def leg_ik_3dof(self, target_pos, leg_type='left', current_angles=None):
        """
        Solve 3-DOF leg inverse kinematics (hip, knee, ankle)
        target_pos: [x, y, z] in leg coordinate frame
        """
        # Leg coordinate frame: origin at hip, x forward, y lateral, z up
        x, y, z = target_pos

        # Calculate hip lateral angle (first joint)
        hip_yaw = math.atan2(y, z) if z != 0 else 0

        # Project onto sagittal plane
        z_proj = math.sqrt(y**2 + z**2)

        # Calculate distance from hip to target in sagittal plane
        dist = math.sqrt(x**2 + z_proj**2)

        # Check if target is reachable
        upper_leg = self.link_lengths['upper_leg']
        lower_leg = self.link_lengths['lower_leg']

        if dist > upper_leg + lower_leg:
            # Target too far, extend leg fully
            hip_pitch = math.atan2(x, z_proj)
            knee_angle = 0
            ankle_pitch = -hip_pitch
        elif dist < abs(upper_leg - lower_leg):
            # Target too close, return None or handle specially
            return None
        else:
            # Standard IK solution
            # Knee angle using law of cosines
            cos_knee = (upper_leg**2 + lower_leg**2 - dist**2) / (2 * upper_leg * lower_leg)
            cos_knee = max(-1, min(1, cos_knee))  # Clamp to valid range
            knee_angle = math.pi - math.acos(cos_knee)

            # Hip pitch
            cos_hip = (upper_leg**2 + dist**2 - lower_leg**2) / (2 * upper_leg * dist)
            cos_hip = max(-1, min(1, cos_hip))
            alpha = math.acos(cos_hip)
            beta = math.atan2(x, z_proj)
            hip_pitch = beta - alpha

            # Ankle pitch to maintain foot orientation
            ankle_pitch = -(hip_pitch + (math.pi - knee_angle))

        # Adjust for leg type (left/right)
        if leg_type == 'right':
            hip_yaw = -hip_yaw  # Mirror for right leg

        return np.array([hip_yaw, hip_pitch, knee_angle, ankle_pitch])

    def arm_ik_3dof(self, target_pos, arm_type='left', current_angles=None):
        """
        Solve 3-DOF arm inverse kinematics (shoulder, elbow)
        target_pos: [x, y, z] in arm coordinate frame
        """
        # Arm coordinate frame: origin at shoulder, x forward, y lateral, z up
        x, y, z = target_pos

        # Calculate shoulder angles
        shoulder_yaw = math.atan2(y, x) if x != 0 else 0
        shoulder_pitch = math.atan2(z, math.sqrt(x**2 + y**2))

        # Calculate distance from shoulder to target
        dist = math.sqrt(x**2 + y**2 + z**2)

        # Check reachability
        upper_arm = self.link_lengths['upper_arm']
        lower_arm = self.link_lengths['lower_arm']

        if dist > upper_arm + lower_arm:
            # Target too far, extend arm fully
            elbow_angle = 0
            shoulder_roll = 0
        elif dist < abs(upper_arm - lower_arm):
            # Target too close
            return None
        else:
            # Standard IK solution
            cos_elbow = (upper_arm**2 + lower_arm**2 - dist**2) / (2 * upper_arm * lower_arm)
            cos_elbow = max(-1, min(1, cos_elbow))
            elbow_angle = math.pi - math.acos(cos_elbow)

            # Calculate shoulder roll to achieve target
            cos_shoulder = (upper_arm**2 + dist**2 - lower_arm**2) / (2 * upper_arm * dist)
            cos_shoulder = max(-1, min(1, cos_shoulder))
            alpha = math.acos(cos_shoulder)
            beta = math.atan2(math.sqrt(x**2 + y**2), z)
            shoulder_roll = beta - alpha

        # Adjust for arm type (left/right)
        if arm_type == 'right':
            shoulder_yaw = -shoulder_yaw  # Mirror for right arm

        return np.array([shoulder_yaw, shoulder_pitch, elbow_angle, shoulder_roll])

    def full_body_ik(self, target_poses, weights=None):
        """
        Solve full-body inverse kinematics
        target_poses: dictionary with target poses for different body parts
        """
        if weights is None:
            weights = {'left_arm': 1.0, 'right_arm': 1.0, 'left_leg': 1.0, 'right_leg': 1.0}

        solutions = {}

        # Solve for each limb independently
        if 'left_hand' in target_poses:
            solutions['left_arm'] = self.arm_ik_3dof(
                target_poses['left_hand'], 'left'
            )

        if 'right_hand' in target_poses:
            solutions['right_arm'] = self.arm_ik_3dof(
                target_poses['right_hand'], 'right'
            )

        if 'left_foot' in target_poses:
            solutions['left_leg'] = self.leg_ik_3dof(
                target_poses['left_foot'], 'left'
            )

        if 'right_foot' in target_poses:
            solutions['right_leg'] = self.leg_ik_3dof(
                target_poses['right_foot'], 'right'
            )

        return solutions

    def optimize_joint_angles(self, target_poses, initial_angles=None):
        """
        Use optimization to find joint angles that best achieve target poses
        """
        def objective_function(joint_angles):
            # Forward kinematics to get current poses
            current_poses = self.forward_kinematics(joint_angles)

            # Calculate error for each target pose
            total_error = 0.0
            for body_part, target_pose in target_poses.items():
                if body_part in current_poses:
                    error = np.linalg.norm(current_poses[body_part] - target_pose)
                    total_error += error**2

            return total_error

        if initial_angles is None:
            initial_angles = np.zeros(20)  # 20 DOF humanoid

        # Optimize joint angles
        result = minimize(
            objective_function,
            initial_angles,
            method='BFGS',
            options={'disp': False}
        )

        return result.x if result.success else initial_angles

    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics to get end-effector positions
        This is a simplified implementation
        """
        # Simplified forward kinematics - in practice, this would use
        # proper transformation matrices and DH parameters
        poses = {}

        # Placeholder implementation
        # In a real system, this would compute actual FK based on joint angles
        poses['left_hand'] = np.array([0.3, -0.2, 1.0])  # Example position
        poses['right_hand'] = np.array([0.3, 0.2, 1.0])   # Example position
        poses['left_foot'] = np.array([0.0, -0.1, 0.05])  # Example position
        poses['right_foot'] = np.array([0.0, 0.1, 0.05])  # Example position

        return poses

# Usage example
def setup_humanoid_ik():
    robot_config = {}  # Would contain actual robot configuration
    ik_solver = HumanoidIKSolver(robot_config)
    print("Humanoid IK solver initialized successfully")
    return ik_solver
```

## AI Integration for Humanoid Perception and Control

### Perception Pipeline for Humanoid Robots

```python
# Perception pipeline for humanoid robots in Isaac Sim
import omni
import numpy as np
import cv2
from omni.isaac.core import World
from omni.isaac.core.sensors import Camera, ImuSensor
from omni.isaac.range_sensor import _range_sensor
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class HumanoidPerceptionPipeline:
    def __init__(self, world: World):
        self.world = world
        self.cameras = {}
        self.sensors = {}
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize neural networks
        self.object_detector = self._init_object_detector()
        self.pose_estimator = self._init_pose_estimator()
        self.depth_estimator = self._init_depth_estimator()

    def _init_object_detector(self):
        """
        Initialize object detection network
        """
        # Using a lightweight network for real-time performance
        class SimpleObjectDetector(nn.Module):
            def __init__(self, num_classes=10):
                super(SimpleObjectDetector, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        return SimpleObjectDetector()

    def _init_pose_estimator(self):
        """
        Initialize human pose estimation network
        """
        class SimplePoseEstimator(nn.Module):
            def __init__(self, num_keypoints=18):
                super(SimplePoseEstimator, self).__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.keypoint_head = nn.Conv2d(128, num_keypoints, kernel_size=1)

            def forward(self, x):
                x = self.backbone(x)
                keypoints = self.keypoint_head(x)
                return keypoints

        return SimplePoseEstimator()

    def _init_depth_estimator(self):
        """
        Initialize depth estimation network
        """
        class SimpleDepthEstimator(nn.Module):
            def __init__(self):
                super(SimpleDepthEstimator, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, kernel_size=3, padding=1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.encoder(x)
                depth = self.decoder(x)
                return depth

        return SimpleDepthEstimator()

    def add_camera_sensor(self, prim_path, position, resolution=(640, 480)):
        """
        Add a camera sensor to the humanoid robot
        """
        camera = self.world.scene.add(
            Camera(
                prim_path=prim_path,
                position=position,
                frequency=30,
                resolution=resolution
            )
        )

        # Add different sensor types to camera
        camera.add_distance_to_image_plane_to_frame()
        camera.add_instance_segmentation_to_frame()

        self.cameras[prim_path] = camera
        return camera

    def add_imu_sensor(self, prim_path, position):
        """
        Add an IMU sensor to the humanoid robot
        """
        imu = self.world.scene.add(
            ImuSensor(
                prim_path=prim_path,
                position=position,
                frequency=100
            )
        )

        self.sensors[prim_path] = imu
        return imu

    def process_camera_data(self, camera_path):
        """
        Process data from a camera sensor
        """
        camera = self.cameras[camera_path]

        # Get RGB image
        rgb_image = camera.get_rgb()

        # Get depth information
        depth_image = camera.get_depth()

        # Get segmentation
        segmentation = camera.get_segmentation()

        # Run through perception networks
        with torch.no_grad():
            # Preprocess image for neural networks
            img_tensor = self.transforms(rgb_image).unsqueeze(0)

            # Object detection
            obj_detections = self.object_detector(img_tensor)

            # Pose estimation
            pose_estimates = self.pose_estimator(img_tensor)

            # Depth estimation
            depth_est = self.depth_estimator(img_tensor)

        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'segmentation': segmentation,
            'objects': obj_detections,
            'poses': pose_estimates,
            'estimated_depth': depth_est
        }

    def process_sensor_fusion(self):
        """
        Fuse data from multiple sensors for comprehensive perception
        """
        sensor_data = {}

        # Process all cameras
        for cam_path, camera in self.cameras.items():
            sensor_data[cam_path] = self.process_camera_data(cam_path)

        # Process IMU data
        for sensor_path, sensor in self.sensors.items():
            if isinstance(sensor, ImuSensor):
                imu_data = {
                    'linear_acceleration': sensor.get_linear_acceleration(),
                    'angular_velocity': sensor.get_angular_velocity()
                }
                sensor_data[sensor_path] = imu_data

        # Implement sensor fusion logic
        fused_data = self._fuse_sensor_data(sensor_data)

        return fused_data

    def _fuse_sensor_data(self, raw_data):
        """
        Implement sensor fusion to combine data from multiple sensors
        """
        # This is a simplified fusion approach
        # In practice, you would use Kalman filters, particle filters, etc.

        fused_result = {
            'environment_map': self._create_environment_map(raw_data),
            'object_poses': self._estimate_object_poses(raw_data),
            'robot_state': self._estimate_robot_state(raw_data),
            'navigation_goals': self._identify_navigation_goals(raw_data)
        }

        return fused_result

    def _create_environment_map(self, sensor_data):
        """
        Create a map of the environment from sensor data
        """
        # Simplified environment mapping
        # In practice, this would use SLAM algorithms
        env_map = {
            'obstacles': [],
            'free_space': [],
            'landmarks': []
        }

        return env_map

    def _estimate_object_poses(self, sensor_data):
        """
        Estimate poses of objects in the environment
        """
        object_poses = []

        # Process each camera's object detections
        for cam_path, data in sensor_data.items():
            if 'objects' in data:
                # Convert 2D detections to 3D poses using depth
                for detection in data['objects']:
                    # Simplified 3D pose estimation
                    pose_3d = self._convert_2d_to_3d(
                        detection['bbox'],
                        data['depth'],
                        cam_path
                    )
                    object_poses.append(pose_3d)

        return object_poses

    def _estimate_robot_state(self, sensor_data):
        """
        Estimate the current state of the robot using sensor data
        """
        # Combine data from IMU and visual odometry
        robot_state = {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],  # quaternion
            'velocity': [0, 0, 0],
            'angular_velocity': [0, 0, 0]
        }

        # Process IMU data for orientation and acceleration
        for sensor_path, data in sensor_data.items():
            if 'linear_acceleration' in data:
                # Integrate IMU data to estimate position and velocity
                pass

        return robot_state

    def _convert_2d_to_3d(self, bbox_2d, depth_map, camera_path):
        """
        Convert 2D bounding box to 3D pose using depth information
        """
        # Simplified conversion
        # In practice, this would use camera intrinsic parameters
        center_x = (bbox_2d[0] + bbox_2d[2]) / 2
        center_y = (bbox_2d[1] + bbox_2d[3]) / 2

        depth = depth_map[int(center_y), int(center_x)]

        # Convert to 3D coordinates (simplified)
        pose_3d = {
            'position': [center_x * depth, center_y * depth, depth],
            'size': [abs(bbox_2d[2] - bbox_2d[0]) * depth,
                     abs(bbox_2d[3] - bbox_2d[1]) * depth,
                     depth * 0.5]
        }

        return pose_3d

    def identify_navigation_goals(self, fused_data):
        """
        Identify potential navigation goals from perception data
        """
        goals = []

        # Identify free space for navigation
        free_space = fused_data['environment_map']['free_space']

        # Identify interesting objects
        objects = fused_data['object_poses']

        # Combine to identify goals
        for obj in objects:
            if self._is_navigable_object(obj):
                goals.append({
                    'type': 'object_interaction',
                    'position': obj['position'],
                    'priority': self._calculate_goal_priority(obj)
                })

        return goals

    def _is_navigable_object(self, obj):
        """
        Determine if an object is a suitable navigation goal
        """
        # Simplified check - in practice, this would be more sophisticated
        return True

    def _calculate_goal_priority(self, obj):
        """
        Calculate priority for a navigation goal
        """
        # Simplified priority calculation
        return 1.0

# Usage example
def setup_humanoid_perception(world):
    perception_pipeline = HumanoidPerceptionPipeline(world)

    # Add sensors to robot
    perception_pipeline.add_camera_sensor(
        "/World/HumanoidRobot/head_camera",
        position=[0.0, 0.0, 0.05],
        resolution=(640, 480)
    )

    perception_pipeline.add_imu_sensor(
        "/World/HumanoidRobot/torso_imu",
        position=[0.0, 0.0, 0.1]
    )

    print("Humanoid perception pipeline configured successfully")
    return perception_pipeline
```

## Walking Pattern Generation and Control

### Gait Pattern Generation

```python
# Gait pattern generation for humanoid robots
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

class GaitPatternGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_duration=0.8):
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.dt = 0.01  # Time step for trajectory generation

        # Walking parameters
        self.stride_length = step_length
        self.step_width = 0.2  # Distance between feet
        self.com_height = 0.8  # Center of mass height

    def generate_walk_trajectory(self, num_steps, walking_speed=0.5):
        """
        Generate complete walking trajectory for multiple steps
        """
        total_time = num_steps * self.step_duration
        num_points = int(total_time / self.dt)

        # Initialize trajectory arrays
        time_vector = np.linspace(0, total_time, num_points)
        com_trajectory = np.zeros((num_points, 3))  # x, y, z
        foot_trajectories = {
            'left': np.zeros((num_points, 3)),
            'right': np.zeros((num_points, 3))
        }

        # Generate trajectories for each step
        for step_idx in range(num_steps):
            start_time = step_idx * self.step_duration
            end_time = (step_idx + 1) * self.step_duration

            # Determine which foot is swing foot for this step
            swing_foot = 'left' if step_idx % 2 == 0 else 'right'
            stance_foot = 'right' if step_idx % 2 == 0 else 'left'

            # Generate step trajectory
            step_indices = np.where((time_vector >= start_time) & (time_vector < end_time))[0]

            for idx in step_indices:
                t_in_step = time_vector[idx] - start_time

                # Update COM trajectory
                com_trajectory[idx, 0] = start_time * walking_speed + self._com_x_profile(t_in_step)
                com_trajectory[idx, 1] = self._com_y_profile(t_in_step, step_idx)
                com_trajectory[idx, 2] = self.com_height + self._com_z_profile(t_in_step)

                # Update foot trajectories
                foot_trajectories[swing_foot][idx, :] = self._swing_foot_trajectory(
                    t_in_step, step_idx, swing_foot
                )
                foot_trajectories[stance_foot][idx, :] = self._stance_foot_position(step_idx, stance_foot)

        return {
            'time': time_vector,
            'com': com_trajectory,
            'left_foot': foot_trajectories['left'],
            'right_foot': foot_trajectories['right'],
            'walking_speed': walking_speed
        }

    def _com_x_profile(self, t):
        """
        Generate COM x-position profile for a single step
        """
        # Smooth transition following the average of foot positions
        phase = t / self.step_duration
        return self.step_length * phase  # Simple linear progression

    def _com_y_profile(self, t, step_idx):
        """
        Generate COM y-position profile to maintain balance
        """
        # Move COM toward stance foot
        phase = t / self.step_duration
        if step_idx % 2 == 0:  # Left foot swing, COM moves right
            target_y = self.step_width / 2
        else:  # Right foot swing, COM moves left
            target_y = -self.step_width / 2

        # Smooth transition
        return target_y * (1 - math.cos(math.pi * phase)) / 2

    def _com_z_profile(self, t):
        """
        Generate COM z-position profile (small oscillations)
        """
        # Small vertical oscillations for natural walking
        phase = t / self.step_duration
        return 0.02 * math.sin(2 * math.pi * phase)  # 2cm oscillation

    def _swing_foot_trajectory(self, t, step_idx, foot_type):
        """
        Generate trajectory for swing foot
        """
        phase = t / self.step_duration

        # X position: move forward
        x = step_idx * self.step_length + self.step_length * phase

        # Y position: move to appropriate side
        if foot_type == 'left':
            y = self.step_width / 2
        else:
            y = -self.step_width / 2

        # Z position: lift and place foot
        z_lift = self.step_height * math.sin(math.pi * phase)  # Parabolic lift
        z = 0.05 + z_lift  # 5cm above ground at highest point

        return np.array([x, y, z])

    def _stance_foot_position(self, step_idx, foot_type):
        """
        Position of stance foot during step
        """
        # Stance foot remains in place during the step
        x = step_idx * self.step_length
        if foot_type == 'left':
            y = self.step_width / 2
        else:
            y = -self.step_width / 2
        z = 0.0  # On the ground

        return np.array([x, y, z])

    def generate_ankle_trajectories(self, walk_trajectory):
        """
        Generate ankle joint trajectories from foot trajectories
        """
        # Calculate required ankle positions to achieve foot trajectories
        # This would involve inverse kinematics in a real implementation
        ankle_trajectories = {}

        for foot_type in ['left', 'right']:
            foot_traj = walk_trajectory[f'{foot_type}_foot']
            ankle_traj = np.zeros_like(foot_traj)

            # In a real system, this would account for foot geometry and orientation
            # For now, assume ankle position is close to foot position
            ankle_traj = foot_traj.copy()

            ankle_trajectories[f'{foot_type}_ankle'] = ankle_traj

        return ankle_trajectories

    def generate_joint_trajectories(self, walk_trajectory):
        """
        Generate complete joint trajectories using inverse kinematics
        """
        # This would use the IK solver to convert Cartesian trajectories to joint space
        # For now, we'll return a placeholder
        num_points = len(walk_trajectory['time'])

        joint_trajectories = {
            'left_hip': np.zeros((num_points, 3)),   # yaw, pitch, roll
            'right_hip': np.zeros((num_points, 3)),
            'left_knee': np.zeros((num_points, 1)),  # flexion
            'right_knee': np.zeros((num_points, 1)),
            'left_ankle': np.zeros((num_points, 3)), # pitch, roll, yaw
            'right_ankle': np.zeros((num_points, 3))
        }

        # Placeholder implementation - in real system, use IK solver
        for joint_name in joint_trajectories:
            if len(joint_trajectories[joint_name].shape) == 2:
                for i in range(joint_trajectories[joint_name].shape[1]):
                    joint_trajectories[joint_name][:, i] = np.sin(
                        2 * np.pi * walk_trajectory['time'] * 0.5
                    ) * 0.1  # Small oscillation
            else:
                joint_trajectories[joint_name][:, 0] = np.sin(
                    2 * np.pi * walk_trajectory['time'] * 0.5
                ) * 0.2  # Larger oscillation for single DOF joints

        return joint_trajectories

    def generate_balance_adjustments(self, sensor_data):
        """
        Generate real-time balance adjustments based on sensor feedback
        """
        # This would process sensor data (IMU, force sensors, etc.) to adjust gait
        adjustments = {
            'com_offset': np.array([0.0, 0.0, 0.0]),
            'step_timing': 1.0,  # Multiplier for step timing
            'step_height': self.step_height,  # Adjusted step height
            'step_width': self.step_width    # Adjusted step width
        }

        # Example: adjust based on IMU data
        if 'imu' in sensor_data:
            imu_data = sensor_data['imu']
            roll = imu_data.get('roll', 0)
            pitch = imu_data.get('pitch', 0)

            # Adjust COM position to counteract tilt
            adjustments['com_offset'][0] = -pitch * 0.05  # Compensate for pitch
            adjustments['com_offset'][1] = -roll * 0.05   # Compensate for roll

            # Adjust step parameters for stability
            adjustments['step_height'] = max(0.02, self.step_height - abs(roll) * 0.1)
            adjustments['step_width'] = max(0.15, self.step_width + abs(roll) * 0.05)

        return adjustments

# Usage example
def setup_gait_generator():
    gait_gen = GaitPatternGenerator(
        step_length=0.3,
        step_height=0.05,
        step_duration=0.8
    )

    # Generate a 10-step walk
    walk_traj = gait_gen.generate_walk_trajectory(10, walking_speed=0.5)
    joint_traj = gait_gen.generate_joint_trajectories(walk_traj)

    print("Gait pattern generator configured successfully")
    return gait_gen, walk_traj, joint_traj
```

## Hands-on Exercise
Implement a complete humanoid robot in Isaac Sim with advanced control:

### Part 1: Humanoid Model Creation
1. Create a detailed humanoid robot model with proper kinematics
2. Configure physics properties for realistic movement
3. Add sensors (cameras, IMU) to the robot model

### Part 2: Control System Implementation
1. Implement ZMP-based balance control
2. Set up inverse kinematics for leg control
3. Create gait pattern generation for walking

### Part 3: AI Integration
1. Implement perception pipeline with neural networks
2. Create sensor fusion system
3. Test with synthetic data generation

### Part 4: Integration and Testing
1. Combine all components into a complete system
2. Test walking behavior in simulation
3. Validate balance control with perturbations

This exercise provides hands-on experience with humanoid robotics in Isaac Sim, covering modeling, control, and AI integration.

## Summary
Isaac Sim provides comprehensive tools for humanoid robotics development, including advanced modeling capabilities, sophisticated control systems, and AI integration. The combination of realistic physics simulation, sensor models, and neural network integration makes it ideal for developing and testing humanoid robots before deployment in the real world.

## Practice Questions
1. What are the key components of ZMP-based balance control?
   - *Answer: Key components include center of mass (CoM) tracking, zero moment point (ZMP) calculation, support polygon definition, and feedback control to maintain ZMP within the support polygon.*

2. How does inverse kinematics facilitate humanoid robot control?
   - *Answer: Inverse kinematics converts desired end-effector positions (like foot or hand positions) into required joint angles, enabling precise control of complex multi-DOF humanoid robots.*

3. What role does sensor fusion play in humanoid perception?
   - *Answer: Sensor fusion combines data from multiple sensors (cameras, IMU, LIDAR) to create a comprehensive understanding of the environment and robot state, improving reliability and accuracy.*

4. Explain the importance of gait pattern generation in humanoid locomotion.
   - *Answer: Gait pattern generation creates stable walking patterns by coordinating foot placement, center of mass movement, and balance control to achieve efficient and stable bipedal locomotion.*

5. How does Isaac Sim's USD framework benefit humanoid robotics simulation?
   - *Answer: USD provides efficient scene management, allows for complex kinematic structures, enables asset reusability, and supports collaboration between different tools in the robotics development pipeline.*

## Check Your Understanding
**Multiple Choice Questions:**

1. What does ZMP stand for in humanoid robotics?
   A) Zero Moment Point  *(Correct)*
   B) Zero Motion Position
   C) Zed Motor Processor
   D) Zone Management Protocol

   *Explanation: ZMP (Zero Moment Point) is a key concept in bipedal robotics representing the point where the ground reaction force would act to maintain balance.*

2. Which joint type is most appropriate for humanoid hip joints?
   A) Revolute joint only
   B) Prismatic joint only
   C) Ball (spherical) joint  *(Correct)*
   D) Fixed joint

   *Explanation: Ball joints provide 3 degrees of freedom needed for the complex movements of humanoid hip joints.*

3. What is the primary purpose of inverse kinematics in robotics?
   A) Calculate joint angles from end-effector positions  *(Correct)*
   B) Calculate end-effector positions from joint angles
   C) Control robot speed
   D) Generate sensor data

   *Explanation: Inverse kinematics calculates the required joint angles to achieve a desired end-effector position and orientation.*

4. Which neural network component is essential for object detection in humanoid perception?
   A) Convolutional layers  *(Correct)*
   B) Recurrent layers only
   C) Fully connected layers only
   D) Normalization layers only

   *Explanation: Convolutional layers are essential for processing visual input and detecting objects in images.*

5. What is the typical range of motion for a humanoid knee joint?
   A) 360 degrees
   B) 180 degrees
   C) 0 to 180 degrees (flexion only)  *(Correct)*
   D) -90 to 90 degrees

   *Explanation: Humanoid knee joints typically allow flexion from 0 to 180 degrees, similar to human knees.*

6. Which sensor is crucial for humanoid balance control?
   A) Camera only
   B) LIDAR only
   C) IMU (Inertial Measurement Unit)  *(Correct)*
   D) Force sensors only

   *Explanation: IMU sensors provide critical information about orientation, acceleration, and angular velocity for balance control.*

7. What is the main advantage of using USD for humanoid robot modeling?
   A) Better rendering only
   B) Efficient scene management and asset reusability  *(Correct)*
   C) Faster physics simulation
   D) Simplified programming

   *Explanation: USD provides efficient management of complex scenes and enables reusability of robot models across different applications.*

8. Which control approach is most effective for humanoid balance?
   A) Open-loop control only
   B) PID control only
   C) Feedback control with sensor integration  *(Correct)*
   D) Feedforward control only

   *Explanation: Feedback control using sensor data is essential for maintaining balance in dynamic humanoid systems.*

<!-- RAG_CHUNK_ID: isaac-sim-humanoid-chapter -->
<!-- URDU_TODO: Translate this chapter to Urdu -->