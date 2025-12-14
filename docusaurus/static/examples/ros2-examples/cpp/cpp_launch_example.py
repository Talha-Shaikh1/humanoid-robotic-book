"""
C++ ROS2 Launch File Example

This example demonstrates how to create a launch file that runs C++ nodes.
The same launch file structure works for both Python and C++ nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate a launch description that defines the C++ nodes to launch.
    """
    # Declare launch arguments
    example_arg = DeclareLaunchArgument(
        'example_param',
        default_value='example_value',
        description='Example launch parameter for C++ nodes'
    )

    # Get launch configurations
    example_param = LaunchConfiguration('example_param')

    # Define C++ nodes to launch
    # Note: The executable name corresponds to the one defined in CMakeLists.txt
    cpp_publisher_node = Node(
        package='my_cpp_package',  # Replace with your package name
        executable='minimal_publisher',  # This matches the executable name in CMakeLists.txt
        name='cpp_publisher_node',  # Custom name for the node
        parameters=[
            {'example_param': example_param},  # Pass parameters to the node
        ],
        remappings=[
            ('topic', 'cpp_topic')  # Remap topic names if needed
        ],
        arguments=['arg1', 'arg2'],  # Command line arguments for the node
        output='screen'  # Output to screen for debugging
    )

    cpp_subscriber_node = Node(
        package='my_cpp_package',  # Replace with your package name
        executable='minimal_subscriber',  # This matches the executable name in CMakeLists.txt
        name='cpp_subscriber_node',  # Custom name for the node
        parameters=[
            {'example_param': example_param},  # Pass parameters to the node
        ],
        remappings=[
            ('topic', 'cpp_topic')  # Remap topic names to match publisher
        ],
        output='screen'  # Output to screen for debugging
    )

    cpp_service_node = Node(
        package='my_cpp_package',  # Replace with your package name
        executable='minimal_service',  # This matches the executable name in CMakeLists.txt
        name='cpp_service_node',  # Custom name for the node
        parameters=[
            {'example_param': example_param},  # Pass parameters to the node
        ],
        output='screen'  # Output to screen for debugging
    )

    cpp_action_server_node = Node(
        package='my_cpp_package',  # Replace with your package name
        executable='fibonacci_action_server',  # This matches the executable name in CMakeLists.txt
        name='cpp_action_server_node',  # Custom name for the node
        parameters=[
            {'example_param': example_param},  # Pass parameters to the node
        ],
        output='screen'  # Output to screen for debugging
    )

    # Return the launch description
    return LaunchDescription([
        example_arg,  # Include the declared launch argument
        cpp_publisher_node,  # Include the C++ publisher node
        cpp_subscriber_node,  # Include the C++ subscriber node
        cpp_service_node,  # Include the C++ service node
        cpp_action_server_node,  # Include the C++ action server node
    ])


"""
Advanced C++ Launch Example with Conditional Launch

This example shows how to conditionally launch C++ nodes based on launch arguments.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_advanced_cpp_launch_description():
    """
    Generate an advanced launch description with conditional C++ node launching.
    """
    # Declare launch arguments
    use_cpp_nodes_arg = DeclareLaunchArgument(
        'use_cpp_nodes',
        default_value='true',
        description='Whether to launch C++ nodes'
    )

    debug_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode for C++ nodes'
    )

    # Get launch configurations
    use_cpp_nodes = LaunchConfiguration('use_cpp_nodes')
    debug_mode = LaunchConfiguration('debug_mode')

    # Define C++ nodes with conditional launch
    conditional_cpp_publisher = Node(
        package='my_cpp_package',
        executable='minimal_publisher',
        name='conditional_cpp_publisher',
        condition=IfCondition(use_cpp_nodes),  # Only launch if use_cpp_nodes is true
        parameters=[
            {'debug_mode': debug_mode}
        ],
        output='screen'
    )

    conditional_cpp_subscriber = Node(
        package='my_cpp_package',
        executable='minimal_subscriber',
        name='conditional_cpp_subscriber',
        condition=IfCondition(use_cpp_nodes),  # Only launch if use_cpp_nodes is true
        parameters=[
            {'debug_mode': debug_mode}
        ],
        output='screen'
    )

    return LaunchDescription([
        use_cpp_nodes_arg,
        debug_arg,
        conditional_cpp_publisher,
        conditional_cpp_subscriber,
    ])


"""
C++ Node Testing Launch Example

This example shows how to create a launch file for testing C++ nodes.
"""
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node


def generate_cpp_testing_launch_description():
    """
    Generate a launch description for testing C++ nodes.
    """
    # Define test nodes
    test_cpp_publisher = Node(
        package='my_cpp_package',
        executable='minimal_publisher',
        name='test_cpp_publisher',
        output='screen'
    )

    test_cpp_subscriber = Node(
        package='my_cpp_package',
        executable='minimal_subscriber',
        name='test_cpp_subscriber',
        output='screen'
    )

    # Event handlers for testing
    on_publisher_start = RegisterEventHandler(
        OnProcessStart(
            target_action=test_cpp_publisher,
            on_start=[
                # Actions to perform when publisher starts
            ]
        )
    )

    on_subscriber_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=test_cpp_subscriber,
            on_exit=[
                # Actions to perform when subscriber exits (e.g., record test results)
            ]
        )
    )

    return LaunchDescription([
        test_cpp_publisher,
        test_cpp_subscriber,
        on_publisher_start,
        on_subscriber_exit,
    ])