"""Robot Simulation Launch File

This launch file demonstrates how to launch a robot simulation in Gazebo with RViz for visualization.
It loads the robot model, spawns it in Gazebo, and starts RViz for visualization.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for robot simulation."""

    # Declare launch arguments
    model_arg = DeclareLaunchArgument(
        name='model',
        default_value='simple_humanoid.urdf',
        description='URDF file for the robot model'
    )

    world_arg = DeclareLaunchArgument(
        name='world',
        default_value='simple_room.world',
        description='World file to load in Gazebo'
    )

    use_rviz_arg = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    # Get launch configurations
    model = LaunchConfiguration('model')
    world = LaunchConfiguration('world')
    use_rviz = LaunchConfiguration('use_rviz')

    # Find the robot model path
    robot_model_path = PathJoinSubstitution([
        FindPackageShare('humanoid_robot_description'),
        'urdf',
        model
    ])

    # Find the world file path
    world_path = PathJoinSubstitution([
        FindPackageShare('humanoid_gazebo_worlds'),
        'worlds',
        world
    ])

    # Launch Gazebo with the specified world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_path,
            'gui': 'true',
            'verbose': 'false'
        }.items()
    )

    # Robot State Publisher node to publish robot state
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': [
                PathJoinSubstitution([
                    FindPackageShare('humanoid_robot_description'),
                    'urdf',
                    model
                ])
            ],
            'publish_frequency': 50.0
        }]
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Launch RViz if requested
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', [
                PathJoinSubstitution([
                    FindPackageShare('humanoid_bringup'),
                    'rviz',
                    'robot_view.rviz'
                ])
            ]
        ],
        output='screen'
    )

    # Joint State Publisher GUI for manual joint control (for testing)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # Create and return the launch description
    return LaunchDescription([
        model_arg,
        world_arg,
        use_rviz_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        rviz,
        joint_state_publisher_gui
    ])


"""Advanced Simulation Launch Example

This example demonstrates a more complex launch setup with multiple robots and custom configurations.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_advanced_simulation_launch():
    """Generate an advanced launch description with multiple robots and custom configurations."""

    # Declare launch arguments
    launch_multiple_robots_arg = DeclareLaunchArgument(
        'launch_multiple_robots',
        default_value='false',
        description='Whether to launch multiple robots'
    )

    robot_count_arg = DeclareLaunchArgument(
        'robot_count',
        default_value='2',
        description='Number of robots to launch if multiple robots enabled'
    )

    enable_logging_arg = DeclareLaunchArgument(
        'enable_logging',
        default_value='true',
        description='Enable detailed logging for debugging'
    )

    # Get launch configurations
    launch_multiple_robots = LaunchConfiguration('launch_multiple_robots')
    robot_count = LaunchConfiguration('robot_count')
    enable_logging = LaunchConfiguration('enable_logging')

    # Basic simulation components (same as simple launch)
    model_arg = DeclareLaunchArgument(
        name='model',
        default_value='simple_humanoid.urdf',
        description='URDF file for the robot model'
    )

    world_arg = DeclareLaunchArgument(
        name='world',
        default_value='simple_room.world',
        description='World file to load in Gazebo'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_gazebo_worlds'),
                'worlds',
                LaunchConfiguration('world')
            ]),
            'gui': 'true',
            'verbose': IfCondition(enable_logging)
        }.items()
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': [
                PathJoinSubstitution([
                    FindPackageShare('humanoid_robot_description'),
                    'urdf',
                    LaunchConfiguration('model')
                ])
            ],
            'publish_frequency': 50.0
        }]
    )

    # Define a group for single robot launch
    single_robot_group = GroupAction(
        condition=UnlessCondition(launch_multiple_robots),
        actions=[
            # Spawn single robot
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic', 'robot_description',
                    '-entity', 'simple_humanoid',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '1.0'
                ],
                output='screen'
            ),

            # Single robot controller manager
            Node(
                package='controller_manager',
                executable='ros2_control_node',
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare('humanoid_control'),
                        'config',
                        'simple_humanoid_controllers.yaml'
                    ])
                ],
                output='screen'
            )
        ]
    )

    # Define a group for multiple robots launch
    multiple_robots_group = GroupAction(
        condition=IfCondition(launch_multiple_robots),
        actions=[]
    )

    # Dynamically add multiple robots based on robot_count
    # This would typically be done with a loop in a real implementation
    for i in range(2):  # Using fixed 2 robots for this example
        # Add robot with unique namespace
        robot_group = GroupAction(
            actions=[
                PushRosNamespace(f'robot{i+1}'),

                # Robot State Publisher for each robot
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name=f'robot_state_publisher_{i+1}',
                    parameters=[{
                        'robot_description': [
                            PathJoinSubstitution([
                                FindPackageShare('humanoid_robot_description'),
                                'urdf',
                                LaunchConfiguration('model')
                            ])
                        ],
                        'publish_frequency': 50.0
                    }],
                    remappings=[
                        ('joint_states', 'joint_states'),
                        ('robot_description', 'robot_description')
                    ]
                ),

                # Spawn robot in Gazebo with unique name and position
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-topic', f'/robot{i+1}/robot_description',
                        '-entity', f'humanoid_robot_{i+1}',
                        '-x', f'{i * 2.0}',  # Space robots apart
                        '-y', '0.0',
                        '-z', '1.0'
                    ],
                    output='screen'
                )
            ]
        )

        # Add the robot group to the multiple robots group
        multiple_robots_group.actions.append(robot_group)

    # RViz node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', [
                PathJoinSubstitution([
                    FindPackageShare('humanoid_bringup'),
                    'rviz',
                    'multi_robot_view.rviz'
                ])
            ]
        ],
        output='screen'
    )

    # Add event handlers for advanced functionality
    # Example: Start controller after robot is spawned
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    spawn_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=Node(  # This would target the spawn_entity node
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic', 'robot_description',
                    '-entity', 'simple_humanoid',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '1.0'
                ]
            ),
            on_start=[controller_spawner]
        )
    )

    # Return the complete launch description
    return LaunchDescription([
        # Launch arguments
        launch_multiple_robots_arg,
        robot_count_arg,
        enable_logging_arg,
        model_arg,
        world_arg,

        # Basic components
        gazebo,
        robot_state_publisher,

        # Conditional robot groups
        single_robot_group,
        multiple_robots_group,

        # Visualization
        rviz,

        # Event handlers
        spawn_event_handler
    ])


"""Simulation Testing Launch Example

This example shows how to create a launch file specifically for testing robot simulations.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_simulation_testing_launch():
    """Generate a launch description for testing robot simulations."""

    # Launch Gazebo without GUI for testing
    gazebo_test = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_gazebo_worlds'),
                'worlds',
                'test_world.world'
            ]),
            'gui': 'false',  # No GUI for testing
            'verbose': 'false'
        }.items()
    )

    # Robot State Publisher for test environment
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': [
                PathJoinSubstitution([
                    FindPackageShare('humanoid_robot_description'),
                    'urdf',
                    'simple_humanoid_test.urdf'
                ])
            ],
            'publish_frequency': 100.0  # Higher frequency for testing
        }],
        output='screen'
    )

    # Test-specific robot controller
    test_controller = Node(
        package='humanoid_control',
        executable='test_controller',
        name='test_controller',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'config',
                'test_config.yaml'
            ])
        ],
        output='screen'
    )

    # Robot pose publisher for testing specific poses
    pose_publisher = Node(
        package='humanoid_utils',
        executable='pose_publisher',
        name='pose_publisher',
        parameters=[
            {'test_poses_file': 'test_poses.yaml'}
        ],
        output='screen'
    )

    # Performance monitoring node
    performance_monitor = Node(
        package='humanoid_utils',
        executable='performance_monitor',
        name='performance_monitor',
        output='screen'
    )

    # Test result publisher
    test_results = Node(
        package='humanoid_test',
        executable='test_results_publisher',
        name='test_results_publisher',
        output='screen'
    )

    # Timer to stop the test after a certain duration
    stop_timer = TimerAction(
        period=60.0,  # Run test for 60 seconds
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'lifecycle', 'set', '/test_controller', 'deactivate'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo_test,
        robot_state_publisher,
        test_controller,
        pose_publisher,
        performance_monitor,
        test_results,
        stop_timer
    ])


if __name__ == '__main__':
    # This would normally be run as part of a launch system
    # The functions above return LaunchDescription objects
    # which are processed by the ROS2 launch system
    pass