---
title: URDF Modeling for Robots
description: Creating and understanding robot models using URDF format
sidebar_position: 2
learning_outcomes:
  - Create complete robot models using URDF format
  - Understand the structure and components of URDF files
  - Implement joints, links, and transmissions in robot models
  - Validate and visualize robot models in simulation
---

# URDF Modeling for Robots: Creating Digital Robot Representations

## Purpose
This chapter covers URDF (Unified Robot Description Format), the standard format for describing robot models in ROS. You'll learn how to create complete robot models that can be used in simulation and real-world applications, including proper kinematic and dynamic properties.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Create complete robot models using URDF format
- Understand the structure and components of URDF files
- Implement joints, links, and transmissions in robot models
- Validate and visualize robot models in simulation

## Understanding URDF Structure

### URDF Overview
URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines:
- **Links**: Rigid bodies with mass, inertia, and geometry
- **Joints**: Connections between links with kinematic constraints
- **Materials**: Visual properties for rendering
- **Transmissions**: Motor and actuator interfaces
- **Gazebo plugins**: Simulation-specific properties

<!-- RAG_CHUNK_ID: urdf-overview-structure -->

### Basic URDF Template

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.5 0" rpy="0 0 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

<!-- RAG_CHUNK_ID: urdf-basic-template -->

## Links: The Building Blocks

### Link Components
Each link in a URDF model contains three main elements:

1. **Visual**: Defines how the link appears in visualization
2. **Collision**: Defines the collision geometry for physics simulation
3. **Inertial**: Defines mass properties for dynamics simulation

```xml
<link name="example_link">
  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_robot/meshes/link_mesh.dae"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>

  <!-- Collision properties for physics -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.5 0.5 0.5"/>
    </geometry>
  </collision>

  <!-- Inertial properties for dynamics -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="2.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

<!-- RAG_CHUNK_ID: urdf-link-components -->

### Geometry Types
URDF supports several geometry types:

- **Box**: Defined by width, height, and depth
- **Cylinder**: Defined by radius and length
- **Sphere**: Defined by radius
- **Mesh**: Imported 3D model files (STL, DAE, OBJ)

```xml
<!-- Different geometry examples -->
<geometry>
  <!-- Box: width, depth, height -->
  <box size="0.5 0.3 0.2"/>
</geometry>

<geometry>
  <!-- Cylinder: radius and length -->
  <cylinder radius="0.1" length="0.5"/>
</geometry>

<geometry>
  <!-- Sphere: radius -->
  <sphere radius="0.2"/>
</geometry>

<geometry>
  <!-- Mesh: external file -->
  <mesh filename="package://my_robot/meshes/complex_part.stl"/>
</geometry>
```

<!-- RAG_CHUNK_ID: urdf-geometry-types -->

## Joints: Connecting Links

### Joint Types
URDF supports several joint types that define how links can move relative to each other:

- **Fixed**: No movement allowed (welded connection)
- **Revolute**: Single axis rotation with limits
- **Continuous**: Single axis rotation without limits
- **Prismatic**: Single axis translation with limits
- **Planar**: Motion on a plane
- **Floating**: 6-DOF motion

```xml
<!-- Revolute joint (hinge) -->
<joint name="hinge_joint" type="revolute">
  <parent link="base_link"/>
  <child link="arm_link"/>
  <origin xyz="0 0 1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
</joint>

<!-- Continuous joint (wheel) -->
<joint name="wheel_joint" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0.5 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
</joint>

<!-- Prismatic joint (slider) -->
<joint name="slider_joint" type="prismatic">
  <parent link="base_link"/>
  <child link="slider_link"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="50" velocity="0.5"/>
</joint>

<!-- Fixed joint (permanent connection) -->
<joint name="fixed_joint" type="fixed">
  <parent link="base_link"/>
  <child link="sensor_link"/>
  <origin xyz="0.1 0.1 0.1" rpy="0 0 0"/>
</joint>
```

<!-- RAG_CHUNK_ID: urdf-joint-types -->

### Joint Properties
Important properties for joints include:

- **Origin**: Position and orientation relative to parent
- **Axis**: Direction of motion (for revolute/prismatic joints)
- **Limits**: Range of motion and physical constraints
- **Dynamics**: Damping and friction coefficients

```xml
<joint name="example_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>

  <!-- Position and orientation relative to parent -->
  <origin xyz="0.1 0.2 0.3" rpy="0.1 0.2 0.3"/>

  <!-- Axis of rotation -->
  <axis xyz="0 0 1"/>

  <!-- Motion limits -->
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>

  <!-- Dynamic properties -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

<!-- RAG_CHUNK_ID: urdf-joint-properties -->

## Transmissions: Motor Interfaces

### Transmission Components
Transmissions define how actuators (motors) connect to joints:

```xml
<transmission name="wheel_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

<!-- RAG_CHUNK_ID: urdf-transmission-components -->

### Hardware Interfaces
Common hardware interfaces include:

- **PositionJointInterface**: Position control
- **VelocityJointInterface**: Velocity control
- **EffortJointInterface**: Torque/force control

<!-- RAG_CHUNK_ID: urdf-hardware-interfaces -->

## Complete Robot Model Example

### Differential Drive Robot

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel_link">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel_link">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel_link"/>
    <origin xyz="0 -0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Transmissions -->
  <transmission name="left_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/diff_drive_robot</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

<!-- RAG_CHUNK_ID: urdf-complete-robot-example -->

## Xacro: URDF Macros

### Xacro Benefits
Xacro (XML Macros) extends URDF with:
- Variables and constants
- Macros for repeated elements
- Mathematical expressions
- File inclusion

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">

  <!-- Define constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="prefix parent x y z">
    <link name="${prefix}_wheel_link">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel_link"/>
      <origin xyz="${x} ${y} ${z}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create wheels -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Create wheels using macro -->
  <xacro:wheel prefix="left" parent="base_link" x="0" y="${base_width/2}" z="0"/>
  <xacro:wheel prefix="right" parent="base_link" x="0" y="${-base_width/2}" z="0"/>

</robot>
```

<!-- RAG_CHUNK_ID: urdf-xacro-macros -->

## Validation and Visualization

### URDF Validation Tools
Several tools help validate URDF models:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Show model information
urdf_to_graphiz /path/to/robot.urdf

# Visualize the robot model
ros2 run rviz2 rviz2
# Then add RobotModel display and specify the URDF file
```

<!-- RAG_CHUNK_ID: urdf-validation-tools -->

### Common Validation Issues
- Missing mass or inertia properties
- Invalid joint limits
- Self-collisions
- Kinematic loops
- Incorrect parent-child relationships

<!-- RAG_CHUNK_ID: urdf-validation-issues -->

## Hands-on Exercise
Create a complete URDF model for a simple 2-DOF manipulator robot with gripper:

### Part 1: Package Setup
1. Create a new robot description package: `ros2 pkg create --build-type ament_cmake manipulator_description`
2. Navigate to the package: `cd ~/ros2_ws/src/manipulator_description`
3. Create directories: `mkdir urdf meshes launch`
4. Create the main URDF file: `urdf/manipulator.urdf`

### Part 2: Complete URDF Model
Create a comprehensive manipulator model in `urdf/manipulator.urdf`:
```xml
<?xml version="1.0"?>
<robot name="simple_manipulator" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
      <origin xyz="0 0 0.05"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0.05"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Base joint (rotation around z-axis) -->
  <joint name="base_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_1_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- First arm link -->
  <link name="arm_1_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
      <origin xyz="0 0 0.15"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <origin xyz="0 0 0.15"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Elbow joint (rotation around y-axis) -->
  <joint name="elbow_joint" type="revolute">
    <parent link="arm_1_link"/>
    <child link="arm_2_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Second arm link -->
  <link name="arm_2_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
      <origin xyz="0 0 0.1"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <origin xyz="0 0 0.1"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Wrist joint -->
  <joint name="wrist_joint" type="revolute">
    <parent link="arm_2_link"/>
    <child link="wrist_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="1.0"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>

  <!-- Wrist link -->
  <link name="wrist_link">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.05"/>
      </geometry>
      <material name="yellow">
        <color rgba="1 1 0 1"/>
      </material>
      <origin xyz="0 0 0.025"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0.025"/>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Gripper base joint -->
  <joint name="gripper_base_joint" type="fixed">
    <parent link="wrist_link"/>
    <child link="gripper_base"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gripper base -->
  <link name="gripper_base">
    <visual>
      <geometry>
        <box size="0.04 0.08 0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.04 0.08 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Left finger joint -->
  <joint name="left_finger_joint" type="prismatic">
    <parent link="gripper_base"/>
    <child link="left_finger"/>
    <origin xyz="0 0.04 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.03" effort="20" velocity="0.1"/>
  </joint>

  <!-- Left finger -->
  <link name="left_finger">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.04"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
      <origin xyz="0 0.01 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.04"/>
      </geometry>
      <origin xyz="0 0.01 0"/>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- Right finger joint -->
  <joint name="right_finger_joint" type="prismatic">
    <parent link="gripper_base"/>
    <child link="right_finger"/>
    <origin xyz="0 -0.04 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.03" effort="20" velocity="0.1"/>
  </joint>

  <!-- Right finger -->
  <link name="right_finger">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.04"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
      <origin xyz="0 -0.01 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.04"/>
      </geometry>
      <origin xyz="0 -0.01 0"/>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- Transmissions for ROS control -->
  <transmission name="base_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="base_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="base_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="elbow_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="elbow_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="wrist_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="wrist_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="wrist_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="gripper_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_finger_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="gripper_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

### Part 3: Xacro Version (Optional Enhancement)
Create a more maintainable version using Xacro in `urdf/manipulator.xacro`:
```xml
<?xml version="1.0"?>
<robot name="simple_manipulator" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Define constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_radius" value="0.1" />
  <xacro:property name="base_length" value="0.1" />
  <xacro:property name="arm1_length" value="0.3" />
  <xacro:property name="arm2_length" value="0.2" />

  <!-- Base macro -->
  <xacro:macro name="base_link_macro" params="name xyz">
    <link name="${name}">
      <visual>
        <geometry>
          <cylinder radius="${base_radius}" length="${base_length}"/>
        </geometry>
        <material name="grey">
          <color rgba="0.5 0.5 0.5 1.0"/>
        </material>
        <origin xyz="${xyz}"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${base_radius}" length="${base_length}"/>
        </geometry>
        <origin xyz="${xyz}"/>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Include the base -->
  <xacro:base_link_macro name="base_link" xyz="0 0 0.05" />

  <!-- Rest of the robot definition continues as in URDF... -->
  <!-- (Similar content as above but using xacro properties) -->

</robot>
```

### Part 4: Launch and Visualization
Create a launch file `launch/display_manipulator.launch.py`:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('manipulator_description')

    urdf_file = os.path.join(pkg_share, 'urdf', 'manipulator.urdf')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    params = {'robot_description': robot_desc}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[params]
        ),

        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_share, 'config', 'manipulator.rviz')],
            output='screen'
        )
    ])
```

### Part 5: Validation and Testing
1. Build your workspace: `cd ~/ros2_ws && colcon build --packages-select manipulator_description`
2. Source the workspace: `source install/setup.bash`
3. Validate the URDF: `check_urdf $(ros2 pkg prefix manipulator_description)/share/manipulator_description/urdf/manipulator.urdf`
4. Display the robot model: `ros2 launch manipulator_description display_manipulator.launch.py`
5. Test the kinematics by moving the joints in the joint state publisher GUI
6. Verify that all links, joints, and transmissions are properly defined

### Expected Results
- Robot model displays correctly in RViz
- All joints move properly in the visualization
- URDF validation passes without errors
- Transmissions are properly defined for ROS control

<!-- RAG_CHUNK_ID: urdf-hands-on-exercise -->

## Summary
URDF is the standard format for describing robot models in ROS, defining links, joints, and their physical properties. Understanding URDF structure, components, and best practices is essential for creating accurate robot models that can be used in both simulation and real-world applications. Xacro extends URDF capabilities with macros and variables for more complex models.

## Further Reading
- [URDF Documentation](http://wiki.ros.org/urdf)
- [Xacro Documentation](http://wiki.ros.org/xacro)
- [URDF Tutorials](http://gazebosim.org/tutorials?tut=ros_urdf)

## Summary
URDF is the standard format for describing robot models in ROS, defining links, joints, and their physical properties. Understanding URDF structure, components, and best practices is essential for creating accurate robot models that can be used in both simulation and real-world applications. Xacro extends URDF capabilities with macros and variables for more complex models.

## Practice Questions
1. What are the three main components of a URDF link?
   - *Answer: The three main components are visual (for rendering appearance), collision (for physics simulation), and inertial (for mass properties and dynamics).*

2. What is the difference between visual and collision geometry?
   - *Answer: Visual geometry defines how the link appears in visualizations, while collision geometry defines the shape used for physics simulation and collision detection. They can be different shapes for performance reasons.*

3. How do transmissions connect actuators to joints?
   - *Answer: Transmissions define the mapping between actuators (motors) and joints, specifying how actuator properties (like effort, velocity) are translated to joint properties.*

## Check Your Understanding
**Multiple Choice Questions:**

1. What does URDF stand for?
   A) Unified Robot Design Format
   B) Universal Robot Description Format
   C) Unified Robot Description Format  *(Correct)*
   D) Universal Robot Design Framework

   *Explanation: URDF stands for Unified Robot Description Format, which is the standard XML format for describing robot models in ROS.*

2. Which of the following is NOT a joint type in URDF?
   A) Revolute
   B) Continuous
   C) Spherical  *(Correct)*
   D) Fixed

   *Explanation: While URDF supports several joint types, Spherical is not a valid joint type. The valid types include revolute, continuous, prismatic, fixed, floating, and planar.*

3. What is the purpose of the `<inertial>` tag in a URDF link?
   A) Defines visual appearance
   B) Defines collision geometry
   C) Defines mass properties for dynamics simulation  *(Correct)*
   D) Defines joint limits

   *Explanation: The `<inertial>` tag defines the mass, center of mass, and inertia matrix for a link, which are used in dynamics simulation.*

<!-- RAG_CHUNK_ID: urdf-modeling-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->