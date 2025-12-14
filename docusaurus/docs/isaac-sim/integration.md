# Isaac Sim Integration in Textbook

This document integrates the Isaac Sim examples into the educational content of the AI-Native Humanoid Robotics textbook.

## Table of Contents
- [Introduction to Isaac Sim](#introduction-to-isaac-sim)
- [Perception Pipeline](#perception-pipeline)
- [Robot Simulation](#robot-simulation)
- [Humanoid Control](#humanoid-control)
- [AI Integration](#ai-integration)
- [Learning Outcomes](#learning-outcomes)
- [Exercises](#exercises)

## Introduction to Isaac Sim

Isaac Sim is NVIDIA's reference application and framework for robot simulation, based on NVIDIA Omniverse. It provides a photorealistic simulation environment with accurate physics, sensor simulation, and AI integration capabilities for robotics development and testing.

### Key Concepts
- NVIDIA Omniverse platform integration
- Physically accurate simulation with PhysX
- Photorealistic rendering with RTX
- GPU-accelerated simulation
- AI training environments
- ROS/ROS2 bridge integration

### Isaac Sim Architecture

Isaac Sim follows a modular architecture that allows for flexible simulation scenarios:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera

class IsaacSimEnvironment:
    def __init__(self):
        # Initialize the simulation world
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,
            rendering_dt=1.0/60.0
        )

    def setup_environment(self):
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        self._setup_lighting()

        # Add robot
        self._add_robot()

        # Add sensors
        self._add_sensors()

        # Reset the world
        self.world.reset()

    def _setup_lighting(self):
        # Add dome light for realistic illumination
        from pxr import UsdGeom, Gf
        dome_light = UsdGeom.DomeLight.Define(self.world.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(3000)

    def _add_robot(self):
        # Add a robot to the simulation
        add_reference_to_stage(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/Carter/carter_v2.usd",
            prim_path="/World/Carter"
        )

    def _add_sensors(self):
        # Add a camera to the robot
        camera = Camera(
            prim_path="/World/Carter/base_link/Camera",
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(camera)
```

### Key Learning Points
1. **Omniverse Integration**: Understanding how Isaac Sim leverages the Omniverse platform
2. **Physics Simulation**: Working with PhysX for accurate physics
3. **Sensor Simulation**: Implementing realistic sensor models
4. **GPU Acceleration**: Leveraging GPU computing for simulation
5. **Modular Design**: Building flexible simulation environments

## Perception Pipeline

The perception pipeline example demonstrates how to create a comprehensive perception system in Isaac Sim with multiple sensors and AI processing.

### Multi-Sensor Perception System

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

class IsaacSimPerceptionPipeline:
    def __init__(self):
        self.world = None
        self.rgb_camera = None
        self.depth_camera = None
        self.segmentation_model = None
        self.detection_model = None

        # Initialize image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    async def setup_sensors(self):
        """Set up various sensors for the perception pipeline."""
        # Set up RGB camera
        self.rgb_camera = Camera(
            prim_path="/World/Robot/base_link/rgb_camera",
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.rgb_camera)

        # Set up depth camera
        self.depth_camera = Camera(
            prim_path="/World/Robot/base_link/depth_camera",
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.depth_camera)

    def setup_ai_models(self):
        """Set up AI models for perception tasks."""
        # Example: Create dummy models for simulation
        self.detection_model = self.create_dummy_detection_model()
        self.segmentation_model = self.create_dummy_segmentation_model()

    def create_dummy_detection_model(self):
        """Create a dummy object detection model for simulation."""
        class DummyDetectionModel:
            def predict(self, image):
                height, width = image.shape[:2]
                detections = [
                    {
                        "label": "person",
                        "confidence": 0.92,
                        "bbox": [int(width*0.3), int(height*0.3), int(width*0.6), int(height*0.8)],
                        "center": [int(width*0.45), int(height*0.55)]
                    },
                    {
                        "label": "box",
                        "confidence": 0.85,
                        "bbox": [int(width*0.7), int(height*0.5), int(width*0.9), int(height*0.9)],
                        "center": [int(width*0.8), int(height*0.7)]
                    }
                ]
                return detections
        return DummyDetectionModel()

    def create_dummy_segmentation_model(self):
        """Create a dummy segmentation model for simulation."""
        class DummySegmentationModel:
            def predict(self, image):
                height, width = image.shape[:2]
                segmentation = np.zeros((height, width), dtype=np.uint8)

                # Create some segmented regions
                segmentation[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.5)] = 1  # Region 1
                segmentation[int(height*0.6):int(height*0.9), int(width*0.6):int(width*0.9)] = 2  # Region 2
                segmentation[:int(height*0.2), :] = 3  # Region 3 (sky)

                return segmentation
        return DummySegmentationModel()

    def process_rgb_data(self):
        """Process RGB camera data."""
        if self.rgb_camera:
            current_frame = self.rgb_camera.get_current_frame()
            rgb_data = current_frame["rgb"]
            return rgb_data
        return None

    def process_depth_data(self):
        """Process depth camera data."""
        if self.depth_camera:
            current_frame = self.depth_camera.get_current_frame()
            depth_data = current_frame["depth"]
            return depth_data
        return None

    def run_object_detection(self, image):
        """Run object detection on the image."""
        return self.detection_model.predict(image)

    def run_semantic_segmentation(self, image):
        """Run semantic segmentation on the image."""
        return self.segmentation_model.predict(image)

    def create_occupancy_grid(self, depth_data):
        """Create an occupancy grid from depth data."""
        if depth_data is not None:
            # Threshold depth to create binary occupancy
            occupancy_grid = (depth_data < 2.0).astype(np.uint8)  # Objects within 2m
            return occupancy_grid
        return np.zeros((480, 640), dtype=np.uint8)

    def visualize_perception_results(self, rgb_image, detections, segmentation, occupancy_grid):
        """Visualize all perception results together."""
        # Create visualization combining all results
        vis_image = rgb_image.copy()

        # Draw detections
        for detection in detections:
            bbox = detection["bbox"]
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(vis_image, f"{detection['label']}: {detection['confidence']:.2f}",
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Overlay segmentation (with transparency)
        seg_overlay = np.zeros_like(vis_image)
        seg_overlay[segmentation == 1] = [255, 0, 0]    # Blue for region 1
        seg_overlay[segmentation == 2] = [0, 255, 0]    # Green for region 2
        seg_overlay[segmentation == 3] = [0, 0, 255]    # Red for region 3
        vis_image = cv2.addWeighted(vis_image, 0.7, seg_overlay, 0.3, 0)

        return vis_image

    def get_3d_object_positions(self, detections, depth_data):
        """Estimate 3D positions of detected objects using depth data."""
        object_3d_positions = []

        for detection in detections:
            center_x, center_y = detection["center"]

            # Get depth at object center
            if center_y < depth_data.shape[0] and center_x < depth_data.shape[1]:
                depth_at_center = depth_data[center_y, center_x]

                # Convert pixel coordinates to 3D world coordinates (simplified)
                x_3d = (center_x - depth_data.shape[1]/2) * depth_at_center * 0.001
                y_3d = (center_y - depth_data.shape[0]/2) * depth_at_center * 0.001
                z_3d = depth_at_center

                object_3d_positions.append({
                    "label": detection["label"],
                    "position": [x_3d, y_3d, z_3d],
                    "confidence": detection["confidence"]
                })

        return object_3d_positions
```

### Perception Pipeline Configuration

Isaac Sim provides comprehensive configuration options for perception systems:

```python
ISAAC_SIM_SENSOR_CONFIG = {
    "rgb_camera": {
        "resolution": [640, 480],
        "frequency": 30,
        "focal_length": 24.0,
        "clipping_range": [0.1, 100.0],
        "projection_type": "perspective"
    },
    "depth_camera": {
        "resolution": [640, 480],
        "frequency": 30,
        "focal_length": 24.0,
        "clipping_range": [0.1, 10.0],
        "data_type": "depth",
        "min_depth": 0.1,
        "max_depth": 10.0
    },
    "lidar": {
        "rotation_frequency": 10,
        "channels": 16,
        "points_per_second": 400000,
        "laser_range": 25.0,
        "enable_semantics": True
    },
    "segmentation": {
        "enable_segmentation": True,
        "output_format": "class_id",
        "colorize_output": True
    }
}
```

## Robot Simulation

The robot simulation example demonstrates how to create realistic robot simulations with physics, control, and sensor integration in Isaac Sim.

### Mobile Robot Simulation

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path, set_attribute
from omni.isaac.core.robots import Robot
import numpy as np
import math

class IsaacSimRobotSimulation:
    def __init__(self):
        self.world = None
        self.robot = None
        self.simulation_steps = 0

    async def setup_environment(self):
        """Set up the simulation environment with robot and scene."""
        # Create the world with physics
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add a ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        self._setup_lighting()

        # Add a simple room environment
        self._setup_room_environment()

        # Load a robot asset
        try:
            add_reference_to_stage(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/Carter/carter_v2.usd",
                prim_path="/World/Carter"
            )
        except Exception as e:
            print(f"Could not load Carter robot, using basic robot: {e}")
            self._create_basic_robot()

        # Reset the world to apply changes
        self.world.reset()

    def _setup_lighting(self):
        """Set up lighting for the simulation environment."""
        from pxr import Gf, UsdGeom

        # Add dome light
        dome_light = UsdGeom.DomeLight.Define(self.world.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(3000)

        # Add key light
        key_light = UsdGeom.DistantLight.Define(self.world.stage, "/World/KeyLight")
        key_light.CreateIntensityAttr(3000)
        key_light.AddTranslateOp().Set(Gf.Vec3f(0, 0, 5))
        key_light.AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))

    def _setup_room_environment(self):
        """Set up a simple room environment with obstacles."""
        from pxr import Gf, UsdGeom

        # Add walls to create a room
        # North wall
        UsdGeom.Xform.Define(self.world.stage, "/World/WallNorth")
        wall_north = UsdGeom.Cube.Define(self.world.stage, "/World/WallNorth/Collision")
        wall_north.CreateSizeAttr(10.0)
        wall_xform = UsdGeom.Xformable(wall_north.GetPrim())
        wall_xform.AddTranslateOp().Set(Gf.Vec3f(0, 5, 1.5))
        wall_xform.AddScaleOp().Set(Gf.Vec3f(10, 0.2, 3))

        # South wall
        UsdGeom.Xform.Define(self.world.stage, "/World/WallSouth")
        wall_south = UsdGeom.Cube.Define(self.world.stage, "/World/WallSouth/Collision")
        wall_south.CreateSizeAttr(10.0)
        wall_xform = UsdGeom.Xformable(wall_south.GetPrim())
        wall_xform.AddTranslateOp().Set(Gf.Vec3f(0, -5, 1.5))
        wall_xform.AddScaleOp().Set(Gf.Vec3f(10, 0.2, 3))

        # East wall
        UsdGeom.Xform.Define(self.world.stage, "/World/WallEast")
        wall_east = UsdGeom.Cube.Define(self.world.stage, "/World/WallEast/Collision")
        wall_east.CreateSizeAttr(10.0)
        wall_xform = UsdGeom.Xformable(wall_east.GetPrim())
        wall_xform.AddTranslateOp().Set(Gf.Vec3f(5, 0, 1.5))
        wall_xform.AddScaleOp().Set(Gf.Vec3f(0.2, 10, 3))

        # West wall
        UsdGeom.Xform.Define(self.world.stage, "/World/WallWest")
        wall_west = UsdGeom.Cube.Define(self.world.stage, "/World/WallWest/Collision")
        wall_west.CreateSizeAttr(10.0)
        wall_xform = UsdGeom.Xformable(wall_west.GetPrim())
        wall_xform.AddTranslateOp().Set(Gf.Vec3f(-5, 0, 1.5))
        wall_xform.AddScaleOp().Set(Gf.Vec3f(0.2, 10, 3))

        # Add some furniture/obstacles
        # Table
        table = UsdGeom.Cube.Define(self.world.stage, "/World/Table")
        table.CreateSizeAttr(2.0)
        table_xform = UsdGeom.Xformable(table.GetPrim())
        table_xform.AddTranslateOp().Set(Gf.Vec3f(2, 2, 0.5))
        table_xform.AddScaleOp().Set(Gf.Vec3f(2, 1, 0.5))

    def move_robot(self, linear_velocity, angular_velocity):
        """Move the robot with specified linear and angular velocities."""
        if self.robot:
            # Get current position and orientation
            current_position, current_orientation = self.robot.get_world_pose()

            # Calculate new position based on velocities
            dt = 1.0/60.0  # Assuming 60 FPS
            new_x = current_position[0] + linear_velocity * math.cos(current_orientation[2]) * dt
            new_y = current_position[1] + linear_velocity * math.sin(current_orientation[2]) * dt
            new_orientation = current_orientation[2] + angular_velocity * dt

            # Set new pose
            self.robot.set_world_pose(
                position=np.array([new_x, new_y, current_position[2]]),
                orientation=np.array([0, 0, math.sin(new_orientation/2), math.cos(new_orientation/2)])
            )

    def get_robot_state(self):
        """Get the current state of the robot."""
        if self.robot:
            position, orientation = self.robot.get_world_pose()
            linear_vel, angular_vel = self.robot.get_linear_velocity(), self.robot.get_angular_velocity()
            return {
                "position": position,
                "orientation": orientation,
                "linear_velocity": linear_vel,
                "angular_velocity": angular_vel
            }
        return None

    def run_control_loop(self):
        """Run a simple control loop to move the robot."""
        # Define a simple trajectory for the robot to follow
        waypoints = [
            np.array([2.0, 0.0, 0.0]),
            np.array([2.0, 2.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        ]

        current_waypoint_idx = 0
        arrival_threshold = 0.3

        while current_waypoint_idx < len(waypoints):
            if self.robot:
                # Get current robot position
                current_pos, _ = self.robot.get_world_pose()
                current_pos_2d = np.array([current_pos[0], current_pos[1]])

                # Get current waypoint
                target_pos_2d = waypoints[current_waypoint_idx][:2]

                # Calculate distance to waypoint
                distance = np.linalg.norm(target_pos_2d - current_pos_2d)

                if distance < arrival_threshold:
                    # Move to next waypoint
                    current_waypoint_idx += 1
                    if current_waypoint_idx < len(waypoints):
                        print(f"Reached waypoint {current_waypoint_idx}, moving to next...")
                else:
                    # Calculate direction to waypoint
                    direction = target_pos_2d - current_pos_2d
                    direction_normalized = direction / np.linalg.norm(direction)

                    # Set robot velocity towards waypoint
                    linear_velocity = 0.5  # m/s
                    angular_velocity = 0.5  # rad/s to turn towards target

                    # Simple proportional controller for orientation
                    target_angle = math.atan2(direction[1], direction[0])
                    _, _, current_yaw = self.robot.get_world_orientation(orientation_type="euler")
                    angle_diff = target_angle - current_yaw

                    # Normalize angle difference
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi

                    angular_velocity = 0.5 * angle_diff  # Proportional control

                    # Apply control (conceptual - actual implementation would use joint control)
                    self.move_robot(linear_velocity, angular_velocity)

            # Step the simulation
            self.world.step(render=True)
```

### Manipulator Simulation

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import math

class IsaacSimManipulatorSimulation:
    def __init__(self):
        self.world = None
        self.manipulator = None
        self.objects = []
        self.end_effector_link = "end_effector"
        self.gripper_joints = ["left_gripper_joint", "right_gripper_joint"]

    async def setup_manipulation_environment(self):
        """Set up the manipulation environment with robot and objects."""
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

        # Load a manipulator robot
        try:
            # Add a UR5 robot (if available in the Isaac Sim assets)
            add_reference_to_stage(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/UR5/ur5.usd",
                prim_path="/World/UR5"
            )
        except Exception as e:
            print(f"Could not load UR5 robot, using basic manipulator: {e}")
            self._create_basic_manipulator()

        # Add objects to manipulate
        self._add_manipulation_objects()

        # Reset the world to apply changes
        self.world.reset()

    def _add_manipulation_objects(self):
        """Add objects for the manipulator to interact with."""
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

    def inverse_kinematics(self, target_position, target_orientation=None):
        """Calculate inverse kinematics to reach target position."""
        # In a real implementation, we would use Isaac Sim's IK solver
        # This is a simplified conceptual implementation
        print(f"Calculating IK for target: {target_position}")

        # Return joint angles that would achieve the target
        # This is a placeholder - real IK would be more complex
        joint_positions = [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]  # Example joint positions
        return joint_positions

    def move_to_joint_positions(self, joint_positions):
        """Move the manipulator to specified joint positions."""
        if self.manipulator:
            # In Isaac Sim, we would command the joints directly
            self.manipulator.set_joint_positions(np.array(joint_positions))

    def move_to_cartesian_pose(self, position, orientation=None):
        """Move the manipulator end-effector to a Cartesian pose."""
        # Calculate required joint positions using IK
        joint_positions = self.inverse_kinematics(position, orientation)

        # Move to calculated joint positions
        self.move_to_joint_positions(joint_positions)

    def open_gripper(self):
        """Open the manipulator's gripper."""
        if self.manipulator:
            # In Isaac Sim, we would control the gripper joints
            # Open gripper by setting joint positions
            self.manipulator.set_joint_positions(np.array([0.05, -0.05]), indices=np.array([6, 7]))

    def close_gripper(self):
        """Close the manipulator's gripper."""
        if self.manipulator:
            # Close gripper by setting joint positions
            self.manipulator.set_joint_positions(np.array([0.01, -0.01]), indices=np.array([6, 7]))

    def grasp_object(self, object_position):
        """Execute a grasping motion to pick up an object."""
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
        """Place the currently grasped object at a target position."""
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
        """Execute a complete pick and place task."""
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
```

## Humanoid Control

The humanoid control example demonstrates advanced locomotion and balance control for humanoid robots in Isaac Sim.

### Humanoid Robot Controller

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
import numpy as np
import math

class IsaacSimHumanoidControl:
    def __init__(self):
        self.world = None
        self.humanoid = None
        self.joint_names = []
        self.balance_controller = None
        self.walk_controller = None

    async def setup_humanoid_environment(self):
        """Set up the humanoid robot environment."""
        # Create the world with physics
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add a ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        self._setup_lighting()

        # Load a humanoid robot
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
        """Set up lighting for the humanoid simulation environment."""
        from pxr import Gf, UsdGeom

        # Add dome light
        dome_light = UsdGeom.DomeLight.Define(self.world.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(3000)

        # Add key light
        key_light = UsdGeom.DistantLight.Define(self.world.stage, "/World/KeyLight")
        key_light.CreateIntensityAttr(3000)
        key_light.AddTranslateOp().Set(Gf.Vec3f(0, 0, 5))
        key_light.AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))

    def initialize_balance_controller(self):
        """Initialize the balance controller for the humanoid."""
        # Implement a basic balance controller using center of mass and zero moment point
        self.balance_controller = {
            'kp': 100.0,  # Proportional gain for balance
            'kd': 10.0,   # Derivative gain for balance
            'com_target': np.array([0.0, 0.0, 0.0]),  # Target center of mass
            'support_polygon': []  # Support polygon for balance
        }

    def initialize_walk_controller(self):
        """Initialize the walking controller for the humanoid."""
        # Implement a basic walking controller using inverse kinematics
        self.walk_controller = {
            'step_length': 0.3,      # Length of each step
            'step_height': 0.1,      # Height of foot lift
            'step_duration': 0.8,    # Duration of each step
            'current_phase': 0.0,    # Current walking phase (0.0 to 1.0)
            'gait_pattern': 'walk'   # Current gait pattern
        }

    def calculate_balance_control(self, current_com, current_com_vel):
        """Calculate balance control commands to maintain stability."""
        # Calculate error from target center of mass
        com_error = self.balance_controller['com_target'] - current_com
        com_vel_error = -current_com_vel  # Target velocity is zero

        # Calculate control force using PD controller
        control_force = (self.balance_controller['kp'] * com_error +
                        self.balance_controller['kd'] * com_vel_error)

        return control_force

    def calculate_walk_trajectory(self, time):
        """Calculate the walking trajectory for the feet."""
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
        """Move the humanoid to specified joint positions."""
        if self.humanoid:
            # In Isaac Sim, we would command the joints directly
            if joint_velocities is not None:
                self.humanoid.set_joint_positions(np.array(joint_positions))
                self.humanoid.set_joint_velocities(np.array(joint_velocities))
            else:
                self.humanoid.set_joint_positions(np.array(joint_positions))

    def get_humanoid_state(self):
        """Get the current state of the humanoid robot."""
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
        """Execute a standing pose for the humanoid."""
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
        """Execute a simple walking motion."""
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

        print("Walking completed!")
```

## AI Integration

The AI integration example demonstrates how to incorporate machine learning models with Isaac Sim for perception, planning, and control.

### AI Model Integration

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np

class PerceptionModel(nn.Module):
    """A simple perception model for object detection in simulated images."""

    def __init__(self, num_classes=10):
        super(PerceptionModel, self).__init__()
        # Simple CNN for demonstration
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 20, 128)  # Assuming 120x160 input -> 30x40 after pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 15 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PlanningModel(nn.Module):
    """A simple planning model for path planning."""

    def __init__(self, input_size=100, output_size=2):  # 10x10 grid flattened, 2D action
        super(PlanningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ControlModel(nn.Module):
    """A simple control model for robot actuation."""

    def __init__(self, state_size=10, action_size=6):  # Example: 10-dim state, 6-dim action
        super(ControlModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Actions between -1 and 1
        return x

class IsaacSimAIIntegration:
    """AI integration system for Isaac Sim with perception, planning, and control."""

    def __init__(self):
        self.world = None
        self.camera = None
        self.robot = None

        # Initialize AI models
        self.perception_model = PerceptionModel()
        self.planning_model = PlanningModel()
        self.control_model = ControlModel()

        # Initialize transforms for image processing
        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Simulation state
        self.current_image = None
        self.detected_objects = []
        self.planned_path = []
        self.current_state = np.zeros(10)  # Example state vector

    async def setup_environment(self):
        """Set up the AI-integrated simulation environment."""
        # Create the world with physics
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add a ground plane
        self.world.scene.add_default_ground_plane()

        # Add a robot
        add_reference_to_stage(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/Carter/carter_v2.usd",
            prim_path="/World/Carter"
        )

        # Add a camera to the robot
        self.camera = Camera(
            prim_path="/World/Carter/base_link/Camera",
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

        # Reset the world to apply changes
        self.world.reset()

    def capture_and_process_image(self):
        """Capture image from simulation and process with AI model."""
        if self.camera:
            # In Isaac Sim, we would capture the image using the camera interface
            # This is a placeholder for the actual Isaac Sim API call
            # For simulation, we'll create a dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Convert to PIL Image and apply transforms
            pil_image = PILImage.fromarray(dummy_image)
            tensor_image = self.transform(pil_image).unsqueeze(0)  # Add batch dimension

            # Run perception model
            with torch.no_grad():
                output = self.perception_model(tensor_image)
                # For demonstration, we'll just return some dummy detections
                self.detected_objects = [
                    {"class": "obstacle", "confidence": 0.85, "bbox": [100, 100, 200, 200]},
                    {"class": "target", "confidence": 0.92, "bbox": [300, 300, 400, 400]}
                ]

            return dummy_image
        return None

    def plan_action(self, detected_objects, robot_state):
        """Plan actions based on detected objects and robot state."""
        # Create a simple occupancy grid representation for planning
        occupancy_grid = np.zeros(100)  # 10x10 grid flattened

        # Mark detected objects in the grid (simplified)
        for obj in detected_objects:
            if obj["class"] == "obstacle":
                # Mark as occupied
                grid_x, grid_y = 5, 5  # Simplified mapping
                grid_idx = grid_y * 10 + grid_x
                if 0 <= grid_idx < 100:
                    occupancy_grid[grid_idx] = 1

        # Convert to tensor and run planning model
        grid_tensor = torch.FloatTensor(occupancy_grid).unsqueeze(0)

        with torch.no_grad():
            action = self.planning_model(grid_tensor)

        # Convert to numpy for easier handling
        planned_action = action.numpy()[0]
        return planned_action

    def control_robot(self, planned_action, current_state):
        """Control the robot based on planned action and current state."""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)

        # Run control model
        with torch.no_grad():
            control_output = self.control_model(state_tensor)

        # Convert to numpy for easier handling
        control_commands = control_output.numpy()[0]

        # In Isaac Sim, we would apply these commands to the robot
        # This is a conceptual implementation
        print(f"Control commands: {control_commands}")

        # Update state based on control commands (simplified)
        self.current_state += control_commands * 0.1  # Integration with time step

        return control_commands

    def run_ai_pipeline(self):
        """Run the complete AI pipeline: perception -> planning -> control."""
        # Step 1: Perception
        image = self.capture_and_process_image()

        # Step 2: Planning
        planned_action = self.plan_action(self.detected_objects, self.current_state)

        # Step 3: Control
        control_commands = self.control_robot(planned_action, self.current_state)

        return image, self.detected_objects, planned_action, control_commands

    def simulate_with_ai(self):
        """Run simulation with AI integration loop."""
        print("Starting AI-integrated simulation...")

        for step in range(300):  # Run for 5 seconds at 60 FPS
            # Run the AI pipeline
            image, detections, planned_action, control_commands = self.run_ai_pipeline()

            # Print status periodically
            if step % 60 == 0:
                print(f"Step {step}: Detected {len(detections)} objects")
                print(f"  Planned action: [{planned_action[0]:.2f}, {planned_action[1]:.2f}]")
                print(f"  Control commands: {control_commands[:3]}...")  # Show first 3

            # Step the simulation
            self.world.step(render=True)

            # Update state (simplified)
            self.current_state[0] += control_commands[0] * 0.01  # Update x position
            self.current_state[1] += control_commands[1] * 0.01  # Update y position
```

### Reinforcement Learning Integration

```python
class ReinforcementLearningAgent:
    """A reinforcement learning agent for robot learning in Isaac Sim."""

    def __init__(self, state_size=10, action_size=6, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = ControlModel(state_size, action_size)
        self.target_network = ControlModel(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)

        for i in batch:
            state, action, reward, next_state, done = self.memory[i]

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_network(next_state_tensor).cpu().data.numpy())

            target_f = self.q_network(state_tensor)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(self.q_network(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# AI integration configuration
ISAAC_SIM_AI_CONFIG = {
    "perception": {
        "models": {
            "object_detection": "yolo_v5",
            "semantic_segmentation": "deep_lab",
            "depth_estimation": "monodepth2"
        },
        "input_resolution": [640, 480],
        "confidence_threshold": 0.5
    },
    "planning": {
        "algorithms": {
            "path_planning": "rrt_star",
            "motion_planning": "chomp",
            "task_planning": "pddl"
        },
        "planning_horizon": 5.0,
        "collision_threshold": 0.1
    },
    "control": {
        "algorithms": {
            "feedback_control": "pid",
            "optimal_control": "ilqr",
            "adaptive_control": "model_reference"
        },
        "control_frequency": 100,
        "tracking_accuracy": 0.01
    },
    "learning": {
        "rl_algorithms": ["ppo", "sac", "ddpg"],
        "imitation_learning": True,
        "sim_to_real": True,
        "training_frequency": 10  # Hz
    }
}
```

## Learning Outcomes

After completing the Isaac Sim examples, students should be able to:

1. **Understand Omniverse**: Comprehend the NVIDIA Omniverse platform and its integration with Isaac Sim
2. **Create Photorealistic Simulations**: Build realistic simulation environments with accurate physics
3. **Implement Perception Systems**: Develop multi-sensor perception pipelines with AI integration
4. **Design Robot Controllers**: Create controllers for various robot types including mobile robots and manipulators
5. **Develop Humanoid Behaviors**: Implement balance control and locomotion for humanoid robots
6. **Integrate AI Models**: Incorporate machine learning models for perception, planning, and control
7. **Train RL Agents**: Develop reinforcement learning systems in simulation environments

## Exercises

1. **Enhance Perception**: Add semantic segmentation to the perception pipeline and visualize the results
2. **Improve Navigation**: Implement a more sophisticated path planning algorithm using the occupancy grid
3. **Humanoid Walking**: Extend the humanoid controller to implement a stable walking gait using ZMP control
4. **Multi-Robot Coordination**: Create a simulation with multiple robots coordinating to achieve a task
5. **AI Model Training**: Train a reinforcement learning agent to perform a specific task in Isaac Sim
6. **Sensor Fusion**: Combine data from multiple sensors (camera, LIDAR, IMU) for more robust perception

## Summary

The Isaac Sim examples provide a comprehensive introduction to advanced robotics simulation with photorealistic rendering, accurate physics, and AI integration. These examples demonstrate how to leverage NVIDIA's Omniverse platform for developing and testing complex robotics systems before deployment on real hardware. The integration of AI models with simulation enables the development of intelligent robotic systems with perception, planning, and control capabilities.