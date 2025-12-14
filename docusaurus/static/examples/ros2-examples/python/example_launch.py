"""
ROS2 Launch File Example

This example demonstrates how to create a launch file for ROS2 nodes.
Launch files allow you to start multiple nodes with specific configurations.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate a launch description that defines the nodes to launch.
    """
    # Declare launch arguments
    # These can be overridden when running the launch file
    example_arg = DeclareLaunchArgument(
        'example_param',
        default_value='example_value',
        description='Example launch parameter'
    )

    # Get launch configurations
    example_param = LaunchConfiguration('example_param')

    # Define nodes to launch
    talker_node = Node(
        package='my_package',  # Replace with your package name
        executable='talker_example',  # Replace with your executable name
        name='talker_node',  # Custom name for the node
        parameters=[
            {'example_param': example_param},  # Pass parameters to the node
        ],
        remappings=[
            ('chatter', 'custom_chatter_topic')  # Remap topic names
        ],
        arguments=['arg1', 'arg2'],  # Command line arguments for the node
        output='screen'  # Output to screen for debugging
    )

    listener_node = Node(
        package='my_package',  # Replace with your package name
        executable='listener_example',  # Replace with your executable name
        name='listener_node',  # Custom name for the node
        parameters=[
            {'example_param': example_param},  # Pass parameters to the node
        ],
        remappings=[
            ('chatter', 'custom_chatter_topic')  # Remap topic names
        ],
        output='screen'  # Output to screen for debugging
    )

    # Return the launch description
    return LaunchDescription([
        example_arg,  # Include the declared launch argument
        talker_node,  # Include the talker node
        listener_node,  # Include the listener node
    ])


"""
Advanced Launch Example with Conditions

This example shows more advanced launch file features including conditions,
groups, and other launch actions.
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    LogInfo,
    RegisterEventHandler
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_advanced_launch_description():
    """
    Generate an advanced launch description with conditions and groups.
    """
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode'
    )

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    debug = LaunchConfiguration('debug')

    # Group of nodes that will share a namespace
    robot_group = GroupAction(
        actions=[
            # Push a namespace for all nodes in this group
            PushRosNamespace('robot1'),

            # Robot controller node
            Node(
                package='my_robot_package',
                executable='robot_controller',
                name='controller',
                parameters=[
                    {'use_sim_time': use_sim_time}
                ],
                condition=IfCondition(debug)  # Only start if debug is true
            ),

            # Robot sensor processor
            Node(
                package='my_robot_package',
                executable='sensor_processor',
                name='sensor_processor',
                parameters=[
                    {'use_sim_time': use_sim_time}
                ]
            )
        ]
    )

    # Event handler example
    event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=None,  # Would specify a particular action
            on_start=[
                LogInfo(msg='Robot nodes started successfully')
            ]
        )
    )

    return LaunchDescription([
        use_sim_time_arg,
        debug_arg,
        robot_group,
        # event_handler,  # Uncomment when target_action is specified
    ])


"""
Parameter File Launch Example

This example shows how to load parameters from a YAML file in a launch file.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_param_launch_description():
    """
    Generate a launch description that loads parameters from a YAML file.
    """
    # Node that loads parameters from a file
    param_node = Node(
        package='my_package',
        executable='parameter_example',
        name='param_node',
        parameters=[
            # Load parameters from a YAML file
            'path/to/params.yaml',

            # Override specific parameters
            {'override_param': 'new_value'},

            # Use launch configurations as parameter values
            # {'param_from_launch': LaunchConfiguration('some_param')}
        ],
        output='screen'
    )

    return LaunchDescription([
        param_node
    ])