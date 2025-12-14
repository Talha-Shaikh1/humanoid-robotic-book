#!/usr/bin/env python3

"""
ROS2 Launch File Example

This example demonstrates how to create launch files for ROS2 nodes,
including parameter configuration, node grouping, and conditional launching.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate a launch description for a robot system with multiple nodes.
    This example demonstrates various launch concepts including arguments,
    conditional launching, and node configuration.
    """
    # Declare launch arguments
    # These can be overridden when running the launch file
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='robot1',
        description='Robot namespace'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    enable_camera_arg = DeclareLaunchArgument(
        'enable_camera',
        default_value='true',
        description='Enable camera node'
    )

    robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value='sensor_robot_example.urdf',
        description='Robot model file'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_camera = LaunchConfiguration('enable_camera')
    robot_model = LaunchConfiguration('robot_model')

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                robot_model
            ])}
        ],
        output='screen'
    )

    # Joint state publisher (for visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Joint state publisher GUI (for manual joint control during testing)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        namespace=namespace,
        condition=IfCondition(LaunchConfiguration('use_gui')),
        output='screen'
    )

    # TF2 static transform publisher for base footprint
    base_footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_publisher',
        namespace=namespace,
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Robot controller nodes
    diff_drive_controller = Node(
        package='controller_manager',
        executable='spawner',
        name='diff_drive_controller_spawner',
        namespace=namespace,
        arguments=['diff_drive_controller'],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        name='joint_state_broadcaster_spawner',
        namespace=namespace,
        arguments=['joint_state_broadcaster'],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Camera node (conditionally enabled)
    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'video_device': '/dev/video0'},
            {'image_width': 640},
            {'image_height': 480},
            {'pixel_format': 'yuyv'},
            {'camera_frame_id': 'camera_link'},
            {'camera_name': 'usb_camera'}
        ],
        condition=IfCondition(enable_camera),
        output='screen'
    )

    # IMU sensor node
    imu_node = Node(
        package='microstrain_inertial_driver',
        executable='microstrain_inertial_driver',
        name='imu_node',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'device': '/dev/ttyACM0'},
            {'baudrate': 115200}
        ],
        output='screen'
    )

    # Navigation nodes (example)
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': [
                'map_server',
                'amcl',
                'bt_navigator',
                'controller_server',
                'local_costmap',
                'global_costmap'
            ]}
        ],
        output='screen'
    )

    # Return the complete launch description
    # This includes all actions that will be executed when the launch file runs
    return LaunchDescription([
        # Launch arguments
        namespace_arg,
        use_sim_time_arg,
        enable_camera_arg,
        robot_model_arg,

        # Robot state publishing
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        base_footprint_publisher,

        # Controller management
        diff_drive_controller,
        joint_state_broadcaster,

        # Sensor nodes
        camera_node,
        imu_node,

        # Navigation system
        lifecycle_manager
    ])


"""
Advanced Launch Example with Groups and Complex Configuration

This example demonstrates more advanced launch concepts including node groups,
remapping, and complex parameter configurations.
"""


def generate_advanced_launch_description():
    """
    Generate an advanced launch description with grouped nodes and complex configurations.
    """
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='robot1',
        description='Robot namespace for multi-robot systems'
    )

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_namespace = LaunchConfiguration('robot_namespace')

    # Group 1: Perception nodes
    perception_nodes = [
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            namespace=robot_namespace,
            parameters=[
                {'use_sim_time': use_sim_time},
                {'target_frame': 'base_link'},
                {'source_frame': 'velodyne'},
                {'min_height': 0.0},
                {'max_height': 1.0},
                {'scan_height': 0.5}
            ],
            remappings=[
                ('cloud_in', 'velodyne_points'),
                ('scan', 'scan_filtered')
            ],
            output='screen'
        ),
        Node(
            package='image_proc',
            executable='image_proc',
            name='image_proc',
            namespace=robot_namespace,
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ]

    # Group 2: Control nodes
    control_nodes = [
        Node(
            package='twist_mux',
            executable='twist_mux',
            name='twist_mux',
            namespace=robot_namespace,
            parameters=[
                {'use_sim_time': use_sim_time},
                {'topics': [
                    {'name': 'navigation', 'topic': 'cmd_nav', 'timeout': 0.5, 'priority': 10},
                    {'name': 'joystick', 'topic': 'cmd_joystick', 'timeout': 0.5, 'priority': 100}
                ]}
            ],
            remappings=[
                ('cmd_vel_out', 'cmd_vel')
            ],
            output='screen'
        ),
        Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='teleop_twist_keyboard',
            namespace=robot_namespace,
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen',
            prefix='xterm -e'
        )
    ]

    # Group 3: Visualization nodes
    visualization_nodes = [
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            namespace=robot_namespace,
            arguments=[
                '-d', [FindPackageShare('my_robot_bringup'), '/rviz/robot_view.rviz']
            ],
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ]

    # Create node groups with namespaces
    perception_group = GroupAction(
        actions=[
            PushRosNamespace(robot_namespace),
            *perception_nodes
        ]
    )

    control_group = GroupAction(
        actions=[
            PushRosNamespace(robot_namespace),
            *control_nodes
        ]
    )

    visualization_group = GroupAction(
        actions=[
            PushRosNamespace(robot_namespace),
            *visualization_nodes
        ]
    )

    # Create timed actions for delayed startup
    delayed_control_nodes = TimerAction(
        period=5.0,  # Start after 5 seconds
        actions=[control_group]
    )

    delayed_visualization_nodes = TimerAction(
        period=10.0,  # Start after 10 seconds
        actions=[visualization_group]
    )

    # Return the advanced launch description
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        robot_namespace_arg,

        # Immediate startup nodes
        perception_group,

        # Delayed startup nodes
        delayed_control_nodes,
        delayed_visualization_nodes
    ])


"""
Multi-Robot Launch Example

This example demonstrates how to launch multiple robots with different configurations.
"""


def generate_multi_robot_launch():
    """
    Generate a launch description for multiple robots with different namespaces.
    """
    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time for all robots'
    )

    # Robot configurations
    robot_configs = [
        {'name': 'robot1', 'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        {'name': 'robot2', 'x': 2.0, 'y': 0.0, 'yaw': 0.0},
        {'name': 'robot3', 'x': 0.0, 'y': 2.0, 'yaw': 1.57}
    ]

    # Generate nodes for each robot
    all_robot_nodes = []
    for config in robot_configs:
        # Robot state publisher for each robot
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=config['name'],
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'robot_description': PathJoinSubstitution([
                    FindPackageShare('multi_robot_demo'),
                    'urdf',
                    'unit_box_robot.urdf'
                ])}
            ],
            output='screen'
        )

        # Navigation for each robot
        nav_bringup_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('nav2_bringup'),
                    'launch',
                    'navigation_launch.py'
                ])
            ]),
            launch_arguments={
                'namespace': config['name'],
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': PathJoinSubstitution([
                    FindPackageShare('multi_robot_demo'),
                    'config',
                    'nav2_params.yaml'
                ])
            }.items()
        )

        all_robot_nodes.extend([robot_state_publisher, nav_bringup_launch])

    # Return the multi-robot launch description
    return LaunchDescription([
        use_sim_time_arg,
        *all_robot_nodes
    ])


# Entry point for the launch file
# This allows the launch file to be run directly
def main():
    """
    Main function that returns the default launch description.
    This function is called when the launch file is executed directly.
    """
    return generate_launch_description()


if __name__ == '__main__':
    # When this file is executed as a launch file, ROS2 will call this
    main()