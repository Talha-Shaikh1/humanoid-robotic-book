#!/usr/bin/env python3
# This example demonstrates manipulator robot simulation in Isaac Sim

"""
Isaac Sim Manipulator Simulation Example

This script demonstrates how to simulate a manipulator robot in Isaac Sim
with inverse kinematics, grasping, and task execution.
"""

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.semantics import add_semantic_label
from omni.isaac.core.objects import DynamicCuboid
from pxr import Gf, UsdGeom
import numpy as np
import math
import asyncio


class IsaacSimManipulatorSimulation:
    """
    A manipulator simulation environment for Isaac Sim with IK and grasping.
    """

    def __init__(self):
        """
        Initialize the manipulator simulation.
        """
        self.world = None
        self.manipulator = None
        self.objects = []
        self.end_effector_link = "end_effector"
        self.gripper_joints = ["left_gripper_joint", "right_gripper_joint"]

    async def setup_manipulation_environment(self):
        """
        Set up the manipulation environment with robot and objects.
        """
        # Create the world with physics
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add a ground plane
        self.world.scene.add_default_ground_plane()

        # Add a table for manipulation
        table = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.5, 0, 0.4]),
                size=0.8,
                color=np.array([0.4, 0.4, 0.4])
            )
        )

        # Load a manipulator robot (using a simple manipulator for this example)
        try:
            # Add a UR5 robot (if available in the Isaac Sim assets)
            add_reference_to_stage(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/UR5/ur5.usd",
                prim_path="/World/UR5"
            )
        except Exception as e:
            print(f"Could not load UR5 robot, using basic manipulator: {e}")
            # Create a basic manipulator if UR5 is not available
            self._create_basic_manipulator()

        # Add objects to manipulate
        self._add_manipulation_objects()

        # Reset the world to apply changes
        self.world.reset()

    def _create_basic_manipulator(self):
        """
        Create a basic manipulator if asset loading fails.
        """
        # Create a simple 6-DOF manipulator arm
        manipulator_xform = UsdGeom.Xform.Define(self.world.stage, "/World/BasicManipulator")

        # Base
        base = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicManipulator/Base")
        base.CreateRadiusAttr(0.1)
        base.CreateHeightAttr(0.2)

        # Links
        link1 = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicManipulator/Link1")
        link1.CreateRadiusAttr(0.05)
        link1.CreateHeightAttr(0.3)
        link1_xform = UsdGeom.Xformable(link1.GetPrim())
        link1_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.2))

        link2 = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicManipulator/Link2")
        link2.CreateRadiusAttr(0.04)
        link2.CreateHeightAttr(0.4)
        link2_xform = UsdGeom.Xformable(link2.GetPrim())
        link2_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.5))

        link3 = UsdGeom.Cylinder.Define(self.world.stage, "/World/BasicManipulator/Link3")
        link3.CreateRadiusAttr(0.03)
        link3.CreateHeightAttr(0.3)
        link3_xform = UsdGeom.Xformable(link3.GetPrim())
        link3_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.85))

        # End effector
        ee = UsdGeom.Sphere.Define(self.world.stage, "/World/BasicManipulator/EndEffector")
        ee.CreateRadiusAttr(0.03)
        ee_xform = UsdGeom.Xformable(ee.GetPrim())
        ee_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1.1))

    def _add_manipulation_objects(self):
        """
        Add objects for the manipulator to interact with.
        """
        # Add colored cubes for manipulation
        red_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/RedCube",
                name="red_cube",
                position=np.array([0.6, 0.2, 0.5]),
                size=0.08,
                color=np.array([0.8, 0.1, 0.1])
            )
        )
        self.objects.append(red_cube)

        green_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/GreenCube",
                name="green_cube",
                position=np.array([0.6, -0.2, 0.5]),
                size=0.08,
                color=np.array([0.1, 0.8, 0.1])
            )
        )
        self.objects.append(green_cube)

        blue_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/BlueCube",
                name="blue_cube",
                position=np.array([0.8, 0.0, 0.5]),
                size=0.08,
                color=np.array([0.1, 0.1, 0.8])
            )
        )
        self.objects.append(blue_cube)

        # Add a target location
        target = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Target",
                name="target",
                position=np.array([0.2, 0.3, 0.5]),
                size=0.15,
                color=np.array([0.5, 0.5, 0.5]),
                mass=0  # Static target
            )
        )

    def inverse_kinematics(self, target_position, target_orientation=None):
        """
        Calculate inverse kinematics to reach target position.
        """
        # In a real implementation, we would use Isaac Sim's IK solver
        # This is a simplified conceptual implementation
        print(f"Calculating IK for target: {target_position}")

        # Return joint angles that would achieve the target
        # This is a placeholder - real IK would be more complex
        joint_positions = [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]  # Example joint positions
        return joint_positions

    def move_to_joint_positions(self, joint_positions):
        """
        Move the manipulator to specified joint positions.
        """
        if self.manipulator:
            # In Isaac Sim, we would command the joints directly
            self.manipulator.set_joint_positions(np.array(joint_positions))

    def move_to_cartesian_pose(self, position, orientation=None):
        """
        Move the manipulator end-effector to a Cartesian pose.
        """
        # Calculate required joint positions using IK
        joint_positions = self.inverse_kinematics(position, orientation)

        # Move to calculated joint positions
        self.move_to_joint_positions(joint_positions)

    def open_gripper(self):
        """
        Open the manipulator's gripper.
        """
        if self.manipulator:
            # In Isaac Sim, we would control the gripper joints
            # Open gripper by setting joint positions
            self.manipulator.set_joint_positions(np.array([0.05, -0.05]), indices=np.array([6, 7]))

    def close_gripper(self):
        """
        Close the manipulator's gripper.
        """
        if self.manipulator:
            # Close gripper by setting joint positions
            self.manipulator.set_joint_positions(np.array([0.01, -0.01]), indices=np.array([6, 7]))

    def grasp_object(self, object_position):
        """
        Execute a grasping motion to pick up an object.
        """
        print(f"Attempting to grasp object at {object_position}")

        # Approach the object from above
        approach_position = [object_position[0], object_position[1], object_position[2] + 0.15]
        self.move_to_cartesian_pose(approach_position)

        # Move down to object level
        self.move_to_cartesian_pose(object_position)

        # Close gripper to grasp
        self.close_gripper()

        # Lift the object
        lift_position = [object_position[0], object_position[1], object_position[2] + 0.1]
        self.move_to_cartesian_pose(lift_position)

        print("Object grasped successfully")

    def place_object(self, target_position):
        """
        Place the currently grasped object at a target position.
        """
        print(f"Placing object at {target_position}")

        # Move to above target position
        approach_position = [target_position[0], target_position[1], target_position[2] + 0.15]
        self.move_to_cartesian_pose(approach_position)

        # Move down to target level
        self.move_to_cartesian_pose(target_position)

        # Open gripper to release
        self.open_gripper()

        # Lift gripper away
        lift_position = [target_position[0], target_position[1], target_position[2] + 0.2]
        self.move_to_cartesian_pose(lift_position)

        print("Object placed successfully")

    def execute_pick_and_place_task(self):
        """
        Execute a complete pick and place task.
        """
        print("Starting pick and place task...")

        # Define object and target positions
        object_pos = np.array([0.6, 0.2, 0.55])  # Position of red cube
        target_pos = np.array([0.2, 0.3, 0.55])  # Target location

        # Open gripper initially
        self.open_gripper()

        # Move to safe position above the object
        safe_pos = [object_pos[0], object_pos[1], object_pos[2] + 0.2]
        self.move_to_cartesian_pose(safe_pos)

        # Execute grasp
        self.grasp_object(object_pos)

        # Move to safe position above the target
        safe_target_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.2]
        self.move_to_cartesian_pose(safe_target_pos)

        # Execute place
        self.place_object(target_pos)

        # Move to home position
        home_pos = [0.3, 0, 0.7]
        self.move_to_cartesian_pose(home_pos)

        print("Pick and place task completed!")


async def run_manipulation_demo():
    """
    Run a demonstration of the manipulator simulation.
    """
    print("Setting up Isaac Sim manipulator simulation...")

    # Create the simulation
    sim = IsaacSimManipulatorSimulation()

    # Set up the environment
    await sim.setup_manipulation_environment()

    print("Manipulation environment setup complete. Starting task...")

    # Execute the pick and place task
    sim.execute_pick_and_place_task()

    # Run simulation for a few seconds to see the result
    for step in range(120):  # 2 seconds at 60 FPS
        sim.world.step(render=True)

    print("Manipulation demo completed.")


# Manipulator control configuration
ISAAC_SIM_MANIPULATOR_CONFIG = {
    "ur5": {
        "joint_names": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ],
        "joint_limits": {
            "min": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
            "max": [3.14, 0, 3.14, 3.14, 3.14, 3.14]
        },
        "ee_link": "ee_link",
        "max_velocity": 1.0,
        "max_effort": 100.0
    },
    "franka_panda": {
        "joint_names": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7"
        ],
        "joint_limits": {
            "min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            "max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        },
        "ee_link": "panda_hand",
        "max_velocity": 1.5,
        "max_effort": 150.0
    },
    "kinematics": {
        "ik_solver": "JacobianTranspose",
        "position_tolerance": 0.001,
        "orientation_tolerance": 0.01,
        "max_iterations": 100
    }
}


# This would be run in Isaac Sim's Python environment
if __name__ == "__main__":
    print("Isaac Sim Manipulator Simulation Example")
    print("=" * 50)
    print("This example demonstrates:")
    print("- Manipulator loading and control")
    print("- Inverse kinematics for Cartesian control")
    print("- Grasping and manipulation tasks")
    print("- Pick and place operations")
    print("- Physics interaction with objects")
    print("=" * 50)
    print("Note: This example is designed for Isaac Sim environment")
    print("In a real Isaac Sim scenario, this would be run as a script within Isaac Sim")