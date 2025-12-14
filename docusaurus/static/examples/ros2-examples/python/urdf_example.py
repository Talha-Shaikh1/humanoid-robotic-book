#!/usr/bin/env python3

"""
ROS2 URDF Example

This example demonstrates how to work with URDF (Unified Robot Description Format)
models in ROS2, including loading, parsing, and using robot models.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.parameter import Parameter
from urdf_parser_py.urdf import URDF
import os


class URDFLoader(Node):
    """
    A ROS2 node that demonstrates loading and working with URDF models.
    """

    def __init__(self):
        """
        Initialize the URDF loader node.
        """
        super().__init__('urdf_loader')

        # Declare a parameter for the URDF file path
        self.declare_parameter(
            'robot_description_file',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to the URDF file for the robot model'
            )
        )

        # Get the URDF file path from parameters
        self.urdf_path = self.get_parameter('robot_description_file').value

        # If no URDF path was provided, use a default
        if not self.urdf_path:
            # Create a simple example URDF string
            self.default_urdf = self.create_example_urdf()
            self.get_logger().info('Using default example URDF model')

            # Load the example URDF
            self.robot_model = URDF.from_xml_string(self.default_urdf)
        else:
            # Load URDF from file
            self.robot_model = self.load_urdf_from_file(self.urdf_path)

        # Print information about the loaded robot model
        self.print_robot_info()

    def create_example_urdf(self):
        """
        Create an example URDF string for demonstration purposes.
        """
        example_urdf = '''<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Arm link -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <link name="arm_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Gripper link -->
  <joint name="arm_to_gripper" type="fixed">
    <parent link="arm_link"/>
    <child link="gripper_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="gripper_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>'''

        return example_urdf

    def load_urdf_from_file(self, file_path):
        """
        Load a URDF model from a file.
        """
        try:
            # Expand user home directory if needed
            expanded_path = os.path.expanduser(file_path)

            # Check if file exists
            if not os.path.exists(expanded_path):
                self.get_logger().error(f'URDF file does not exist: {expanded_path}')
                return None

            # Load the URDF from file
            robot = URDF.from_xml_file(expanded_path)
            self.get_logger().info(f'Loaded URDF from: {expanded_path}')
            return robot

        except Exception as e:
            self.get_logger().error(f'Failed to load URDF from {file_path}: {str(e)}')
            return None

    def print_robot_info(self):
        """
        Print information about the loaded robot model.
        """
        if self.robot_model is None:
            self.get_logger().error('No robot model loaded')
            return

        self.get_logger().info(f'Robot Name: {self.robot_model.name}')
        self.get_logger().info(f'Number of Links: {len(self.robot_model.links)}')
        self.get_logger().info(f'Number of Joints: {len(self.robot_model.joints)}')

        # Print link information
        for link in self.robot_model.links:
            self.get_logger().info(f'  Link: {link.name}')
            if link.inertial:
                self.get_logger().info(f'    Mass: {link.inertial.mass}')
            if link.visual:
                geom = link.visual.geometry
                if hasattr(geom, 'size'):
                    self.get_logger().info(f'    Visual Size: {geom.size}')
                elif hasattr(geom, 'radius'):
                    self.get_logger().info(f'    Visual Radius: {geom.radius}')
                if link.visual.material:
                    self.get_logger().info(f'    Material: {link.visual.material.name}')

        # Print joint information
        for joint in self.robot_model.joints:
            self.get_logger().info(f'  Joint: {joint.name} (Type: {joint.type})')
            self.get_logger().info(f'    Parent: {joint.parent}')
            self.get_logger().info(f'    Child: {joint.child}')

    def get_link_by_name(self, link_name):
        """
        Get a link by its name.
        """
        for link in self.robot_model.links:
            if link.name == link_name:
                return link
        return None

    def get_joint_by_name(self, joint_name):
        """
        Get a joint by its name.
        """
        for joint in self.robot_model.joints:
            if joint.name == joint_name:
                return joint
        return None


class URDFAnalyzer(Node):
    """
    A ROS2 node that analyzes URDF models for various properties.
    """

    def __init__(self):
        """
        Initialize the URDF analyzer node.
        """
        super().__init__('urdf_analyzer')

        # Declare parameter for URDF file
        self.declare_parameter('robot_description_file', '')

        # Get URDF file path
        urdf_path = self.get_parameter('robot_description_file').value

        # Create example URDF for demonstration
        example_urdf = self.create_example_urdf()
        self.robot_model = URDF.from_xml_string(example_urdf)

        # Analyze the robot model
        self.analyze_robot_model()

    def create_example_urdf(self):
        """
        Create a more complex example URDF for analysis.
        """
        complex_urdf = '''<?xml version="1.0"?>
<robot name="analyzed_robot">
  <link name="base_footprint"/>

  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0" ixz="0" iyy="0.4" iyz="0" izz="0.2"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left_link"/>
    <origin xyz="0 0.2 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_left_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right_link"/>
    <origin xyz="0 -0.2 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_right_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
  </link>
</robot>'''

        return complex_urdf

    def analyze_robot_model(self):
        """
        Analyze the robot model for various properties.
        """
        if self.robot_model is None:
            self.get_logger().error('No robot model to analyze')
            return

        self.get_logger().info('=== URDF Model Analysis ===')

        # Check for common issues
        self.check_common_issues()

        # Calculate total mass
        total_mass = self.calculate_total_mass()
        self.get_logger().info(f'Total Robot Mass: {total_mass:.2f} kg')

        # Find chains and degrees of freedom
        dof = self.calculate_degrees_of_freedom()
        self.get_logger().info(f'Degrees of Freedom: {dof}')

        # Check for fixed joints
        fixed_joints = [j for j in self.robot_model.joints if j.type == 'fixed']
        self.get_logger().info(f'Fixed Joints: {len(fixed_joints)}')

    def check_common_issues(self):
        """
        Check for common URDF issues.
        """
        issues = []

        # Check for links without visuals
        no_visual_links = [l for l in self.robot_model.links if not l.visual]
        if no_visual_links:
            issues.append(f'Links without visuals: {[l.name for l in no_visual_links]}')

        # Check for links without collisions
        no_collision_links = [l for l in self.robot_model.links if not l.collision]
        if no_collision_links:
            issues.append(f'Links without collisions: {[l.name for l in no_collision_links]}')

        # Check for links without inertials
        no_inertial_links = [l for l in self.robot_model.links if not l.inertial]
        if no_inertial_links:
            issues.append(f'Links without inertials: {[l.name for l in no_inertial_links]}')

        if issues:
            for issue in issues:
                self.get_logger().warn(f'URDF Issue: {issue}')
        else:
            self.get_logger().info('No common URDF issues detected')

    def calculate_total_mass(self):
        """
        Calculate the total mass of the robot.
        """
        total_mass = 0.0
        for link in self.robot_model.links:
            if link.inertial and link.inertial.mass:
                total_mass += link.inertial.mass
        return total_mass

    def calculate_degrees_of_freedom(self):
        """
        Calculate the degrees of freedom of the robot.
        """
        dof = 0
        for joint in self.robot_model.joints:
            if joint.type in ['revolute', 'prismatic', 'continuous']:
                dof += 1
            elif joint.type == 'planar':
                dof += 3
            elif joint.type == 'floating':
                dof += 6
        return dof


def main_loader(args=None):
    """
    Main function for the URDF loader node.
    """
    rclpy.init(args=args)

    urdf_loader = URDFLoader()

    try:
        rclpy.spin(urdf_loader)
    except KeyboardInterrupt:
        pass
    finally:
        urdf_loader.destroy_node()
        rclpy.shutdown()


def main_analyzer(args=None):
    """
    Main function for the URDF analyzer node.
    """
    rclpy.init(args=args)

    urdf_analyzer = URDFAnalyzer()

    try:
        rclpy.spin(urdf_analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        urdf_analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # This file contains multiple main functions for different node types
    # Run one of them depending on the desired functionality
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'loader':
            main_loader()
        elif sys.argv[1] == 'analyzer':
            main_analyzer()
        else:
            print("Usage: python urdf_example.py [loader|analyzer]")
    else:
        # Default to loader if no argument provided
        main_loader()