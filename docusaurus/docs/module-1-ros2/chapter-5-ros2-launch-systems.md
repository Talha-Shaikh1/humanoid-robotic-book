---
title: ROS 2 Launch Systems
description: Managing and orchestrating multiple nodes with ROS 2 launch files
sidebar_position: 5
learning_outcomes:
  - Understand the ROS 2 launch system architecture
  - Create launch files for complex multi-node systems
  - Use launch arguments and conditional execution
  - Implement advanced launch features like event handlers
---

# ROS 2 Launch Systems: Orchestrating Multi-Node Systems

## Purpose
This chapter covers the ROS 2 launch system, which provides a powerful way to manage and orchestrate multiple nodes in complex robotic systems. You'll learn how to create launch files that can start, stop, and manage multiple nodes with configuration, parameters, and sophisticated control flow.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Understand the ROS 2 launch system architecture
- Create launch files for complex multi-node systems
- Use launch arguments and conditional execution
- Implement advanced launch features like event handlers

## Introduction to ROS 2 Launch System

### Why Launch Systems?
In real robotic applications, you rarely run just a single node. Instead, you typically need to:
- Start multiple nodes simultaneously
- Configure nodes with specific parameters
- Control the order of node startup
- Manage node lifecycles and dependencies
- Handle node failures gracefully

The ROS 2 launch system addresses all these needs with a flexible, Python-based framework.

<!-- RAG_CHUNK_ID: ros2-launch-system-introduction -->

### Launch Architecture
The ROS 2 launch system is built on:
- **LaunchDescription**: The main container for all launch entities
- **LaunchActions**: Actions that perform operations (e.g., start nodes)
- **LaunchConditions**: Conditional execution of actions
- **LaunchSubstitutions**: Dynamic value resolution
- **EventHandlers**: React to system events

<!-- RAG_CHUNK_ID: ros2-launch-architecture-components -->

## Basic Launch Files

### Simple Launch File

```python
# launch/simple_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop'
        )
    ])
```

To run this launch file:
```bash
ros2 launch my_robot_pkg simple_launch.py
```

<!-- RAG_CHUNK_ID: ros2-simple-launch-file -->

### Launch File with Parameters

```python
# launch/parameterized_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_pkg',
            executable='parameter_node',
            name='configurable_node',
            parameters=[
                {'robot_name': 'launch_robot'},
                {'max_velocity': 1.5},
                {'debug_mode': True}
            ]
        )
    ])
```

<!-- RAG_CHUNK_ID: ros2-parameterized-launch-file -->

## Launch Arguments

### Using Launch Arguments
Launch arguments allow you to customize launch behavior without modifying the launch file:

```python
# launch/argument_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='default_robot',
        description='Name of the robot'
    )

    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode'
    )

    # Use launch arguments in node configuration
    robot_name = LaunchConfiguration('robot_name')
    debug_mode = LaunchConfiguration('debug_mode')

    return LaunchDescription([
        robot_name_arg,
        debug_mode_arg,

        Node(
            package='my_robot_pkg',
            executable='parameter_node',
            name='configurable_node',
            parameters=[
                {'robot_name': robot_name},
                {'debug_mode': [debug_mode, 'false']}
            ]
        )
    ])
```

To run with arguments:
```bash
ros2 launch my_robot_pkg argument_launch.py robot_name:=husky_robot debug_mode:=true
```

<!-- RAG_CHUNK_ID: ros2-launch-arguments-implementation -->

### Conditional Launch Actions
Launch conditions allow you to start nodes based on certain conditions:

```python
# launch/conditional_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug nodes'
    )

    gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Use GUI components'
    )

    debug = LaunchConfiguration('debug')
    use_gui = LaunchConfiguration('use_gui')

    return LaunchDescription([
        debug_arg,
        gui_arg,

        # Start GUI node only if use_gui is true
        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='rqt_gui',
            condition=IfCondition(use_gui)
        ),

        # Start debug nodes only if debug is true
        Node(
            package='rqt_plot',
            executable='rqt_plot',
            name='debug_plot',
            condition=IfCondition(debug)
        ),

        # Log a message based on condition
        LogInfo(
            msg=['Debug mode is enabled: ', debug],
            condition=IfCondition(debug)
        )
    ])
```

<!-- RAG_CHUNK_ID: ros2-conditional-launch-actions -->

## Advanced Launch Features

### Launch Substitutions
Substitutions allow dynamic value resolution in launch files:

```python
# launch/substitution_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, TextSubstitution, EnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='robot1',
        description='Robot namespace'
    )

    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        namespace_arg,

        # Set environment variable
        SetEnvironmentVariable(
            name='MY_ROBOT_NAMESPACE',
            value=namespace
        ),

        # Use substitution in node name and namespace
        Node(
            package='my_robot_pkg',
            executable='parameter_node',
            name='control_node',
            namespace=namespace,  # Use namespace from argument
            parameters=[
                {'robot_namespace': namespace}
            ]
        )
    ])
```

<!-- RAG_CHUNK_ID: ros2-launch-substitutions -->

### Event Handling in Launch Files
Event handlers allow you to react to system events:

```python
# launch/event_handling_launch.py
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    # Define nodes
    talker_node = Node(
        package='demo_nodes_cpp',
        executable='talker',
        name='talker'
    )

    listener_node = Node(
        package='demo_nodes_cpp',
        executable='listener',
        name='listener'
    )

    return LaunchDescription([
        # Register event handler for when talker starts
        RegisterEventHandler(
            OnProcessStart(
                target_action=talker_node,
                on_start=[
                    LogInfo(msg='Talker node started, now starting listener'),
                    listener_node  # Start listener after talker
                ]
            )
        ),

        # Register event handler for when talker exits
        RegisterEventHandler(
            OnProcessExit(
                target_action=talker_node,
                on_exit=[
                    LogInfo(msg='Talker node exited, shutting down listener'),
                    # Could add shutdown actions here
                ]
            )
        ),

        # Start the talker node
        talker_node
    ])
```

<!-- RAG_CHUNK_ID: ros2-event-handling-launch -->

## Complex Multi-Node Launch Example

### Complete Robot System Launch
Here's a more complex example that demonstrates many launch features together:

```python
# launch/robot_system_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='robot1',
        description='Robot namespace'
    )

    enable_viz_arg = DeclareLaunchArgument(
        'enable_viz',
        default_value='true',
        description='Enable visualization'
    )

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_namespace = LaunchConfiguration('robot_namespace')
    enable_viz = LaunchConfiguration('enable_viz')

    # Define nodes for robot system
    robot_nodes = GroupAction(
        actions=[
            # Push namespace for all nodes in this group
            PushRosNamespace(namespace=robot_namespace),

            # Robot controller node
            Node(
                package='my_robot_pkg',
                executable='robot_controller',
                name='controller',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_namespace': robot_namespace}
                ],
                respawn=True,
                respawn_delay=2.0
            ),

            # Sensor processing node
            Node(
                package='my_robot_pkg',
                executable='sensor_processor',
                name='sensor_processor',
                parameters=[
                    {'use_sim_time': use_sim_time}
                ]
            ),

            # Navigation node
            Node(
                package='nav2_bringup',
                executable='nav2_launch',
                name='navigator',
                parameters=[
                    {'use_sim_time': use_sim_time}
                ],
                condition=IfCondition(use_sim_time)  # Only if using sim time
            )
        ]
    )

    # Visualization nodes (only if enabled)
    viz_nodes = GroupAction(
        actions=[
            PushRosNamespace(namespace=robot_namespace),

            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', [FindPackageShare('my_robot_pkg'), '/rviz/robot_config.rviz']],
                condition=IfCondition(enable_viz)
            )
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_namespace_arg,
        enable_viz_arg,

        robot_nodes,
        viz_nodes
    ])
```

**Instructions:**
1. Create the launch file in your package: `my_robot_pkg/launch/robot_system_launch.py`
2. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
3. Source the setup file: `source install/setup.bash`
4. Run the launch file: `ros2 launch my_robot_pkg robot_system_launch.py robot_namespace:=my_robot enable_viz:=true`

<!-- RAG_CHUNK_ID: ros2-complex-launch-example -->

## Launch File Best Practices

### Organizing Launch Files
1. **Separate concerns**: Create different launch files for different subsystems
2. **Use includes**: Include common launch files to avoid duplication
3. **Parameter files**: Separate parameters into YAML files for better organization
4. **Naming conventions**: Use clear, descriptive names for launch files

### Performance Considerations
1. **Minimize startup time**: Start only necessary nodes initially
2. **Conditional launching**: Use conditions to avoid starting unused nodes
3. **Resource management**: Consider memory and CPU usage of launched nodes
4. **Error handling**: Implement proper error handling and recovery

<!-- RAG_CHUNK_ID: ros2-launch-best-practices -->

## Hands-on Exercise
Create a complete launch system for a mobile robot with advanced features:

### Part 1: Package Setup
1. Create a new package: `ros2 pkg create --build-type ament_python mobile_robot_launch`
2. Navigate to the package: `cd ~/ros2_ws/src/mobile_robot_launch`
3. Create a launch directory: `mkdir launch`
4. Create a config directory: `mkdir config`

### Part 2: Robot Configuration File
Create a configuration file `config/robot_config.yaml`:
```yaml
# Robot configuration parameters
mobile_robot_simulator:
  ros__parameters:
    robot_type: "turtlebot4"  # Default robot type
    max_velocity: 0.5
    max_angular_velocity: 1.0
    sensor_range: 3.0
    update_rate: 50.0

mobile_robot_controller:
  ros__parameters:
    linear_velocity_scale: 1.0
    angular_velocity_scale: 1.0
    safety_distance: 0.5

sensor_processor:
  ros__parameters:
    detection_threshold: 0.7
    max_detection_range: 5.0
```

### Part 3: Launch File Implementation
Create a comprehensive launch file `launch/mobile_robot_system.launch.py`:
```python
# mobile_robot_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    """Function to set up the launch description with conditional logic."""

    # Get launch arguments
    robot_type = LaunchConfiguration('robot_type')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_rviz = LaunchConfiguration('enable_rviz')
    robot_name = LaunchConfiguration('robot_name')

    # Set global parameters
    set_use_sim_time = SetParameter(name='use_sim_time', value=use_sim_time)

    # Robot simulator node
    robot_simulator_node = Node(
        package='mobile_robot_launch',  # Replace with actual package
        executable='mobile_robot_simulator',  # Replace with actual executable
        name=[robot_name, '_simulator'],
        parameters=[
            PathJoinSubstitution([FindPackageShare('mobile_robot_launch'), 'config', 'robot_config.yaml']),
            {'robot_type': robot_type},
            {'use_sim_time': use_sim_time}
        ],
        condition=UnlessCondition(LaunchConfiguration('use_gazebo', default='false')),
        output='screen'
    )

    # Controller node
    controller_node = Node(
        package='mobile_robot_launch',  # Replace with actual package
        executable='mobile_robot_controller',  # Replace with actual executable
        name=[robot_name, '_controller'],
        parameters=[
            PathJoinSubstitution([FindPackageShare('mobile_robot_launch'), 'config', 'robot_config.yaml']),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Sensor processing node
    sensor_processor_node = Node(
        package='mobile_robot_launch',  # Replace with actual package
        executable='sensor_processor',  # Replace with actual executable
        name=[robot_name, '_sensor_processor'],
        parameters=[
            PathJoinSubstitution([FindPackageShare('mobile_robot_launch'), 'config', 'robot_config.yaml']),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # RViz2 visualization (conditionally launched)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([FindPackageShare('mobile_robot_launch'), 'config', 'robot_config.rviz'])],
        condition=IfCondition(enable_rviz),
        output='screen'
    )

    # Return all launch actions
    return [
        set_use_sim_time,
        robot_simulator_node,
        controller_node,
        sensor_processor_node,
        rviz_node
    ]

def generate_launch_description():
    """Generate the launch description with arguments."""

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'robot_type',
            default_value='turtlebot4',
            description='Type of robot to simulate (turtlebot4, diff_drive, etc.)'
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for all nodes'
        ),

        DeclareLaunchArgument(
            'enable_rviz',
            default_value='true',
            description='Enable RViz2 visualization'
        ),

        DeclareLaunchArgument(
            'robot_name',
            default_value='mobile_robot',
            description='Name of the robot instance'
        ),

        DeclareLaunchArgument(
            'use_gazebo',
            default_value='false',
            description='Use Gazebo instead of simple simulator'
        ),

        # Opaque function that contains the main launch logic
        OpaqueFunction(function=launch_setup)
    ])
```

### Part 4: Advanced Launch with Event Handling
Create an advanced launch file `launch/advanced_mobile_robot_system.launch.py` with event handling:
```python
# advanced_mobile_robot_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import lifecycle_msgs.msg

def generate_launch_description():
    """Generate launch description with event handling."""

    # Launch arguments
    robot_name = LaunchConfiguration('robot_name', default='mobile_robot')

    # Robot simulator node
    robot_simulator_node = Node(
        package='mobile_robot_launch',
        executable='mobile_robot_simulator',
        name=[robot_name, '_simulator'],
        output='screen'
    )

    # Controller node that should start after simulator
    controller_node = Node(
        package='mobile_robot_launch',
        executable='mobile_robot_controller',
        name=[robot_name, '_controller'],
        output='screen'
    )

    # Event handler to start controller after simulator is running
    controller_start_event = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_simulator_node,
            on_start=[
                controller_node,
            ],
        )
    )

    # Sensor processor node
    sensor_processor_node = Node(
        package='mobile_robot_launch',
        executable='sensor_processor',
        name=[robot_name, '_sensor_processor'],
        output='screen'
    )

    # Event handler to start sensor processor after controller
    sensor_processor_start_event = RegisterEventHandler(
        OnProcessStart(
            target_action=controller_node,
            on_start=[
                sensor_processor_node,
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_name',
            default_value='mobile_robot',
            description='Name of the robot instance'
        ),

        robot_simulator_node,
        controller_start_event,
        sensor_processor_start_event,
    ])
```

### Part 5: Testing
1. Build your workspace: `cd ~/ros2_ws && colcon build --packages-select mobile_robot_launch`
2. Source the workspace: `source install/setup.bash`
3. Test basic launch: `ros2 launch mobile_robot_launch mobile_robot_system.launch.py enable_rviz:=false`
4. Test with different robot type: `ros2 launch mobile_robot_launch mobile_robot_system.launch.py robot_type:=diff_drive`
5. Test with visualization: `ros2 launch mobile_robot_launch mobile_robot_system.launch.py enable_rviz:=true robot_name:=test_robot`
6. Test advanced launch: `ros2 launch mobile_robot_launch advanced_mobile_robot_system.launch.py`

### Expected Results
- Launch files start nodes in the correct order
- Launch arguments properly configure the system
- Visualization tools are conditionally launched
- Event handlers ensure proper startup sequence
- The system works with different robot configurations

<!-- RAG_CHUNK_ID: ros2-hands-on-exercise-launch-systems -->

## Summary
The ROS 2 launch system provides a powerful and flexible way to manage complex multi-node robotic systems. With launch arguments, conditions, substitutions, and event handling, you can create sophisticated launch configurations that adapt to different environments and requirements. Understanding launch systems is essential for deploying real-world robotic applications.

## Further Reading
- [ROS 2 Launch System Documentation](https://docs.ros.org/en/humble/How-To-Guides/Launch-system.html)
- [Launch File Best Practices](https://docs.ros.org/en/humble/How-To-Guides/Using-Launch-Files.html)
- [Launch Arguments Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Using-Launch-Arguments.html)

## Summary
The ROS 2 launch system provides a powerful framework for managing and orchestrating complex multi-node robotic systems. With launch files, you can start multiple nodes simultaneously, configure them with parameters, control startup order, handle dependencies, and respond to system events. This enables the development of sophisticated robotic applications with proper system management.

## Practice Questions
1. What is the purpose of launch arguments in ROS 2?
   - *Answer: Launch arguments allow you to parameterize launch files, making them flexible and reusable. They enable different configurations without modifying the launch file itself.*

2. How do you conditionally launch nodes based on parameters?
   - *Answer: You can use launch conditions like `IfCondition` and `UnlessCondition` to conditionally execute launch actions based on the value of launch arguments or other conditions.*

3. What are event handlers in the ROS 2 launch system?
   - *Answer: Event handlers are callbacks that respond to system events like node shutdowns, errors, or other launch-related events. They allow you to implement sophisticated launch logic and error handling.*

## Check Your Understanding
**Multiple Choice Questions:**

1. What is the main purpose of ROS 2 launch files?
   A) To store robot data permanently
   B) To manage and orchestrate multiple nodes  *(Correct)*
   C) To create custom message types
   D) To handle hardware interfaces

   *Explanation: Launch files are designed to start, stop, and manage multiple nodes simultaneously with proper configuration and coordination.*

2. How do you pass arguments to a launch file?
   A) Through environment variables only
   B) Through the command line with ros2 launch  *(Correct)*
   C) By modifying the source code
   D) Through service calls

   *Explanation: Launch arguments are passed via the command line when running `ros2 launch`, allowing for flexible configuration of launch files.*

3. What does the condition `IfCondition` do in a launch file?
   A) Always executes the action
   B) Executes the action only if the condition is true  *(Correct)*
   C) Executes the action only if the condition is false
   D) Delays execution of the action

   *Explanation: `IfCondition` is a launch condition that executes the associated action only when the specified condition evaluates to true.*

<!-- RAG_CHUNK_ID: ros2-launch-systems-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->