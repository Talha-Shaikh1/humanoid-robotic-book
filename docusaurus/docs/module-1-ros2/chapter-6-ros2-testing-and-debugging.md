---
title: ROS 2 Testing and Debugging
description: Techniques for testing, debugging, and maintaining ROS 2 systems
sidebar_position: 6
learning_outcomes:
  - Implement unit and integration tests for ROS 2 nodes
  - Use debugging tools and techniques for ROS 2 systems
  - Apply logging and monitoring best practices
  - Understand testing strategies for robotic systems
---

# ROS 2 Testing and Debugging: Ensuring Robust Robotic Systems

## Purpose
This chapter covers essential techniques for testing, debugging, and maintaining ROS 2 systems. You'll learn how to write effective tests, debug common issues, and implement proper logging and monitoring to ensure your robotic systems are reliable and maintainable.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Implement unit and integration tests for ROS 2 nodes
- Use debugging tools and techniques for ROS 2 systems
- Apply logging and monitoring best practices
- Understand testing strategies for robotic systems

## Testing in ROS 2

### Unit Testing Basics
Unit testing in ROS 2 follows standard Python testing practices with some ROS-specific considerations:

```python
# test/test_parameter_node.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_pkg.parameter_node import ParameterNode

class TestParameterNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ParameterNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_default_parameters(self):
        """Test that default parameters are set correctly"""
        robot_name = self.node.get_parameter('robot_name').value
        self.assertEqual(robot_name, 'default_robot')

        max_velocity = self.node.get_parameter('max_velocity').value
        self.assertEqual(max_velocity, 1.0)

    def test_parameter_change(self):
        """Test that parameters can be changed"""
        # Change parameter value
        self.node.set_parameters([rclpy.Parameter('max_velocity', value=2.5)])

        # Verify change
        new_velocity = self.node.get_parameter('max_velocity').value
        self.assertEqual(new_velocity, 2.5)

if __name__ == '__main__':
    unittest.main()
```

To run the test:
```bash
# From the package root
python3 -m pytest test/test_parameter_node.py
# Or
python3 -m unittest test.test_parameter_node
```

<!-- RAG_CHUNK_ID: ros2-unit-testing-basics -->

### Integration Testing
Integration tests verify that multiple nodes work together correctly:

```python
# test/test_node_communication.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from my_robot_pkg.talker import TalkerNode
from my_robot_pkg.listener import ListenerNode

class TestNodeCommunication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.talker = TalkerNode()
        self.listener = ListenerNode()

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.talker)
        self.executor.add_node(self.listener)

    def tearDown(self):
        self.talker.destroy_node()
        self.listener.destroy_node()

    def test_message_passing(self):
        """Test that messages pass correctly between nodes"""
        # Create a mock subscription to capture messages
        received_messages = []

        def message_callback(msg):
            received_messages.append(msg.data)

        # Subscribe to the talker's topic
        sub = self.listener.create_subscription(
            String,
            'chatter',
            message_callback,
            10
        )

        # Run for a short time to allow messages to pass
        self.executor.spin_once(timeout_sec=1.0)

        # Check if messages were received
        self.assertGreater(len(received_messages), 0)

        # Clean up
        self.listener.destroy_subscription(sub)

if __name__ == '__main__':
    unittest.main()
```

<!-- RAG_CHUNK_ID: ros2-integration-testing -->

### Launch Testing
ROS 2 provides specialized testing for launch files:

```python
# test/test_launch.py
import unittest
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import launch_testing.actions
import launch_testing.asserts

def generate_test_description():
    """Generate launch description with processes to test"""

    # Declare launch arguments for testing
    test_param = DeclareLaunchArgument(
        'test_param',
        default_value='test_value',
        description='Test parameter'
    )

    # Define the node to test
    test_node = Node(
        package='my_robot_pkg',
        executable='parameter_node',
        name='test_node',
        parameters=[{'robot_name': LaunchConfiguration('test_param')}]
    )

    # Create launch description
    ld = LaunchDescription([
        test_param,
        test_node,
        # Start tests after nodes are launched
        launch_testing.actions.ReadyToTest()
    ])

    # Pass context to test function
    return ld, {'test_node': test_node}

class TestNodeLaunch(unittest.TestCase):
    def test_node_launch(self, proc_info, test_node):
        """Test that the node launches successfully"""
        # Check that the node has started
        proc_info.assertWaitForStart(test_node, timeout=5)

        # Check that the node doesn't exit unexpectedly
        proc_info.assertWaitForShutdown(test_node, timeout=1)
```

<!-- RAG_CHUNK_ID: ros2-launch-testing -->

## Debugging Techniques

### ROS 2 Command-Line Tools
ROS 2 provides several command-line tools for debugging:

```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info /node_name

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /topic_name std_msgs/msg/String

# Publish a message to a topic
ros2 topic pub /topic_name std_msgs/msg/String "data: 'Hello'"

# List all services
ros2 service list

# Call a service
ros2 service call /service_name example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# List all actions
ros2 action list

# Get parameter values
ros2 param list
ros2 param get /node_name param_name

# Set parameter values
ros2 param set /node_name param_name value
```

<!-- RAG_CHUNK_ID: ros2-debugging-command-line-tools -->

### Logging and Debugging in Code

```python
import rclpy
from rclpy.node import Node
import traceback

class DebuggableNode(Node):
    def __init__(self):
        super().__init__('debuggable_node')

        # Set up different logging levels
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # Example of different log levels
        self.get_logger().debug('Debug message - detailed info')
        self.get_logger().info('Info message - normal operation')
        self.get_logger().warn('Warning message - potential issue')
        self.get_logger().error('Error message - something went wrong')
        self.get_logger().fatal('Fatal message - critical error')

        # Create a publisher for debugging
        self.debug_publisher = self.create_publisher(String, 'debug_info', 10)

        # Timer for periodic debugging
        self.debug_timer = self.create_timer(1.0, self.debug_callback)

    def debug_callback(self):
        """Debug callback with error handling"""
        try:
            # Simulate some work that might fail
            self.perform_debug_operation()

        except Exception as e:
            # Log the error with traceback
            self.get_logger().error(f'Error in debug callback: {str(e)}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

            # Publish error information
            error_msg = String()
            error_msg.data = f'Error: {str(e)}'
            self.debug_publisher.publish(error_msg)

    def perform_debug_operation(self):
        """Example operation that might fail"""
        # Add debug points here
        self.get_logger().debug('Starting debug operation')

        # Simulate some computation
        result = 10 / 2  # This is safe

        # Add more debug info
        self.get_logger().debug(f'Operation result: {result}')

        # Publish debug information
        debug_msg = String()
        debug_msg.data = f'Debug info: result = {result}'
        self.debug_publisher.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DebuggableNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {str(e)}')
        node.get_logger().error(f'Traceback: {traceback.format_exc()}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    from std_msgs.msg import String
    main()
```

<!-- RAG_CHUNK_ID: ros2-debugging-in-code -->

### Using GDB for Debugging
For C++ nodes, you can use GDB:

```bash
# Launch node with GDB
gdb --args ros2 run my_robot_pkg my_cpp_node

# Or use valgrind for memory debugging
valgrind --tool=memcheck ros2 run my_robot_pkg my_cpp_node
```

### Remote Debugging
For debugging nodes running on remote robots:

```python
# Enable remote debugging in your node
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
```

<!-- RAG_CHUNK_ID: ros2-remote-debugging -->

## Logging Best Practices

### Structured Logging
Implement structured logging for better analysis:

```python
import rclpy
from rclpy.node import Node
import json
from datetime import datetime

class StructuredLoggingNode(Node):
    def __init__(self):
        super().__init__('structured_logging_node')

        # Create a publisher for structured logs
        self.log_publisher = self.create_publisher(String, 'structured_logs', 10)

    def log_structured_event(self, event_type, data, level='INFO'):
        """Log a structured event with timestamp and metadata"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'node': self.get_name(),
            'event_type': event_type,
            'data': data,
            'level': level
        }

        # Log to ROS logger
        self.get_logger().info(f'Structured log: {json.dumps(log_entry)}')

        # Publish structured log for external analysis
        log_msg = String()
        log_msg.data = json.dumps(log_entry)
        self.log_publisher.publish(log_msg)

    def example_operation(self):
        """Example operation with structured logging"""
        try:
            self.log_structured_event(
                'operation_start',
                {'operation': 'example_operation', 'step': 1},
                'INFO'
            )

            # Perform operation
            result = self.perform_calculation()

            self.log_structured_event(
                'operation_success',
                {'operation': 'example_operation', 'result': result},
                'INFO'
            )

            return result

        except Exception as e:
            self.log_structured_event(
                'operation_error',
                {'operation': 'example_operation', 'error': str(e)},
                'ERROR'
            )
            raise

    def perform_calculation(self):
        """Example calculation that might fail"""
        return 42  # Example result

def main(args=None):
    rclpy.init(args=args)
    node = StructuredLoggingNode()

    try:
        result = node.example_operation()
        node.get_logger().info(f'Operation completed with result: {result}')
    except Exception as e:
        node.get_logger().error(f'Operation failed: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    from std_msgs.msg import String
    main()
```

<!-- RAG_CHUNK_ID: ros2-structured-logging -->

## Monitoring and Performance

### Performance Monitoring
Monitor node performance and resource usage:

```python
import rclpy
from rclpy.node import Node
import psutil
import time
from std_msgs.msg import String

class MonitoringNode(Node):
    def __init__(self):
        super().__init__('monitoring_node')

        # Publishers for different metrics
        self.cpu_publisher = self.create_publisher(String, 'cpu_usage', 10)
        self.memory_publisher = self.create_publisher(String, 'memory_usage', 10)
        self.process_publisher = self.create_publisher(String, 'process_stats', 10)

        # Timers for different monitoring tasks
        self.cpu_timer = self.create_timer(2.0, self.publish_cpu_usage)
        self.memory_timer = self.create_timer(2.0, self.publish_memory_usage)
        self.process_timer = self.create_timer(5.0, self.publish_process_stats)

        # Performance tracking
        self.start_time = time.time()

    def publish_cpu_usage(self):
        """Publish CPU usage statistics"""
        cpu_percent = psutil.cpu_percent(interval=1)

        cpu_msg = String()
        cpu_msg.data = f'CPU Usage: {cpu_percent}%'
        self.cpu_publisher.publish(cpu_msg)

        self.get_logger().debug(f'CPU Usage: {cpu_percent}%')

    def publish_memory_usage(self):
        """Publish memory usage statistics"""
        memory = psutil.virtual_memory()

        memory_msg = String()
        memory_msg.data = f'Memory Usage: {memory.percent}% ({memory.used / 1024 / 1024:.2f} MB used)'
        self.memory_publisher.publish(memory_msg)

        self.get_logger().debug(f'Memory Usage: {memory.percent}%')

    def publish_process_stats(self):
        """Publish process-specific statistics"""
        process = psutil.Process()

        stats_msg = String()
        stats_msg.data = f'Process Stats - PID: {process.pid}, Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB, CPU: {process.cpu_percent()}%'
        self.process_publisher.publish(stats_msg)

        self.get_logger().info(f'Process Stats: PID={process.pid}, Memory={process.memory_info().rss / 1024 / 1024:.2f} MB')

def main(args=None):
    rclpy.init(args=args)
    node = MonitoringNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Monitoring node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-performance-monitoring -->

## Testing Strategies for Robotic Systems

### Test-Driven Development for Robotics
Implement TDD principles in robotic development:

```python
# First, write a test for the desired behavior
# test/test_robot_controller.py
import unittest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class TestRobotController(unittest.TestCase):
    def test_move_forward(self):
        """Test that robot moves forward when given linear velocity command"""
        # Given: Robot at initial position
        initial_position = (0.0, 0.0)

        # When: Command linear velocity forward
        command = Twist()
        command.linear.x = 1.0  # Move forward
        command.angular.z = 0.0  # No rotation

        # Then: Robot should move forward after some time
        # (This would be tested in a simulation environment)
        expected_final_position = (1.0, 0.0)  # After 1 second at 1 m/s
        # assert abs(final_position[0] - expected_final_position[0]) < 0.1
```

### Simulation-Based Testing
Use Gazebo or other simulators for testing:

```python
# test/test_with_gazebo.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class TestWithSimulation(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = rclpy.create_node('test_with_simulation')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        # Publishers and subscribers for testing
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.current_position = None

    def odom_callback(self, msg):
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def test_robot_movement(self):
        """Test robot movement in simulation"""
        # Record initial position
        initial_pos = self.current_position

        # Send movement command
        cmd = Twist()
        cmd.linear.x = 1.0  # Move forward
        self.cmd_vel_pub.publish(cmd)

        # Wait for movement
        self.executor.spin_once(timeout_sec=2.0)

        # Check that robot moved
        final_pos = self.current_position
        if initial_pos and final_pos:
            distance_moved = ((final_pos[0] - initial_pos[0])**2 +
                             (final_pos[1] - initial_pos[1])**2)**0.5
            self.assertGreater(distance_moved, 0.1)  # Should have moved at least 0.1m

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    unittest.main()
```

<!-- RAG_CHUNK_ID: ros2-testing-strategies -->

## Hands-on Exercise
Create a comprehensive testing and debugging setup for a robot controller with unit tests, integration tests, and monitoring:

### Part 1: Package Setup
1. Create a new package: `ros2 pkg create --build-type ament_python robot_testing_pkg --dependencies rclpy std_msgs geometry_msgs`
2. Navigate to the package: `cd ~/ros2_ws/src/robot_testing_pkg`
3. Create directories: `mkdir test launch config`
4. Create the main controller file: `robot_controller.py`

### Part 2: Robot Controller Implementation
Create a robot controller with logging and metrics:
```python
# robot_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import time
import psutil

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Create publisher and subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.odom_pub = self.create_publisher(
            Twist,
            'odom',
            10
        )

        # Performance monitoring
        self.cmd_received_count = 0
        self.last_cmd_time = self.get_clock().now()

        # Timer for publishing performance metrics
        self.timer = self.create_timer(1.0, self.publish_metrics)

        self.get_logger().info('Robot Controller initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands with logging."""
        self.cmd_received_count += 1
        current_time = self.get_clock().now()

        # Log command with performance metrics
        if self.last_cmd_time.nanoseconds > 0:
            dt = (current_time - self.last_cmd_time).nanoseconds / 1e9
            self.get_logger().info(
                f'Command received: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}, '
                f'interval={dt:.3f}s, total_cmds={self.cmd_received_count}',
                throttle_duration_sec=1  # Log once per second max
            )

        self.last_cmd_time = current_time

        # Publish odometry (simulated)
        odom_msg = Twist()
        odom_msg.linear.x = msg.linear.x * 0.95  # Simulate some delay/response
        odom_msg.angular.z = msg.angular.z * 0.95
        self.odom_pub.publish(odom_msg)

    def publish_metrics(self):
        """Publish system metrics for monitoring."""
        # Get system CPU and memory usage
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        self.get_logger().info(
            f'System metrics - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, '
            f'Cmds/sec: {self.cmd_received_count}',
            throttle_duration_sec=0  # Always log metrics
        )

        # Reset command counter
        self.cmd_received_count = 0

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Robot Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Unit Tests Implementation
Create comprehensive unit tests: `test/test_robot_controller.py`
```python
#!/usr/bin/env python3
# test/test_robot_controller.py

import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from robot_testing_pkg.robot_controller import RobotController
import threading
import time

class TestRobotController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = RobotController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_initialization(self):
        """Test that the controller initializes correctly."""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.cmd_received_count, 0)

    def test_cmd_vel_callback(self):
        """Test the command velocity callback function."""
        # Create a test message
        test_cmd = Twist()
        test_cmd.linear.x = 1.0
        test_cmd.angular.z = 0.5

        # Call the callback
        self.node.cmd_vel_callback(test_cmd)

        # Check that command counter increased
        self.assertEqual(self.node.cmd_received_count, 1)

    def test_odom_publisher_exists(self):
        """Test that the odometry publisher exists."""
        self.assertIsNotNone(self.node.odom_pub)

def run_executor(executor):
    """Helper function to run executor in a separate thread."""
    executor.spin()

class TestRobotControllerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.controller = RobotController()

        # Create a publisher to send test commands
        self.test_publisher = self.controller.create_publisher(
            Twist, 'cmd_vel', 10
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.controller)

        # Start executor in a separate thread
        self.executor_thread = threading.Thread(
            target=run_executor, args=(self.executor,), daemon=True
        )
        self.executor_thread.start()

    def tearDown(self):
        self.controller.destroy_node()

    def test_command_response_integration(self):
        """Test end-to-end command and response."""
        # Wait a bit for systems to initialize
        time.sleep(0.5)

        # Record initial state
        initial_count = self.controller.cmd_received_count

        # Publish a test command
        test_cmd = Twist()
        test_cmd.linear.x = 1.0
        test_cmd.angular.z = 0.2

        self.test_publisher.publish(test_cmd)

        # Wait for processing
        time.sleep(0.1)

        # Check that command was received
        self.assertGreater(self.controller.cmd_received_count, initial_count)

if __name__ == '__main__':
    unittest.main()
```

### Part 4: Launch File for Testing
Create a launch file for testing: `launch/test_robot_system.launch.py`
```python
# launch/test_robot_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # Robot controller node
        Node(
            package='robot_testing_pkg',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen',
            emulate_tty=True  # Better output formatting
        ),

        # Additional monitoring node (optional)
        Node(
            package='robot_testing_pkg',
            executable='system_monitor',
            name='system_monitor',
            output='screen'
        ),
    ])
```

### Part 5: System Monitoring Node
Create a system monitoring node: `system_monitor.py`
```python
# system_monitor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
import psutil
import time

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        # Publishers for different metrics
        self.cpu_pub = self.create_publisher(Float64, 'system/cpu_usage', 10)
        self.memory_pub = self.create_publisher(Float64, 'system/memory_usage', 10)
        self.disk_pub = self.create_publisher(Float64, 'system/disk_usage', 10)

        # Timer for periodic monitoring
        self.timer = self.create_timer(2.0, self.monitor_system)

        self.get_logger().info('System Monitor initialized')

    def monitor_system(self):
        """Monitor system resources and publish metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        # Publish metrics
        cpu_msg = Float64()
        cpu_msg.data = float(cpu_percent)
        self.cpu_pub.publish(cpu_msg)

        memory_msg = Float64()
        memory_msg.data = float(memory_percent)
        self.memory_pub.publish(memory_msg)

        disk_msg = Float64()
        disk_msg.data = float(disk_percent)
        self.disk_pub.publish(disk_msg)

        # Log system status
        self.get_logger().info(
            f'System Status - CPU: {cpu_percent:.1f}%, '
            f'Memory: {memory_percent:.1f}%, '
            f'Disk: {disk_percent:.1f}%'
        )

def main(args=None):
    rclpy.init(args=args)
    monitor = SystemMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Shutting down System Monitor')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 6: Testing Commands
1. Build your workspace: `cd ~/ros2_ws && colcon build --packages-select robot_testing_pkg`
2. Source the workspace: `source install/setup.bash`
3. Run unit tests: `cd ~/ros2_ws/build/robot_testing_pkg && python3 -m pytest test/test_robot_controller.py -v`
4. Run the system: `ros2 launch robot_testing_pkg test_robot_system.launch.py`
5. In another terminal, send test commands: `ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}'`
6. Monitor the system: `ros2 run rqt_plot rqt_plot`
7. Use debugging tools:
   - List nodes: `ros2 node list`
   - List topics: `ros2 topic list`
   - Echo topic data: `ros2 topic echo /odom geometry_msgs/Twist`
   - Check node info: `ros2 node info /robot_controller`

### Expected Results
- Unit tests pass successfully
- Robot controller responds to velocity commands
- System monitoring shows resource usage
- All debugging tools work as expected
- Proper logging provides visibility into system operation

<!-- RAG_CHUNK_ID: ros2-hands-on-exercise-testing-debugging -->

## Summary
Testing and debugging are critical for developing reliable robotic systems. ROS 2 provides comprehensive tools for unit testing, integration testing, and system monitoring. By implementing proper logging, using command-line debugging tools, and following testing best practices, you can create robust and maintainable robotic applications.

## Further Reading
- [ROS 2 Testing Documentation](https://docs.ros.org/en/humble/How-To-Guides/Ament-Cmake-Documentation.html)
- [ROS 2 Debugging Tips](https://docs.ros.org/en/humble/How-To-Guides/Debugging.html)
- [Logging Best Practices](https://docs.ros.org/en/humble/How-To-Guides/Logging.html)

## Summary
Testing and debugging are critical for developing reliable robotic systems. ROS 2 provides comprehensive tools for unit testing, integration testing, and system monitoring. By implementing proper logging, using command-line debugging tools, and following testing best practices, you can create robust and maintainable robotic applications.

## Practice Questions
1. What are the different types of testing in ROS 2?
   - *Answer: The main types are unit testing (testing individual components), integration testing (testing interactions between components), and system testing (testing the complete system). Each serves different purposes in ensuring system reliability.*

2. How do you implement structured logging in ROS 2 nodes?
   - *Answer: Use the built-in logger with appropriate severity levels (debug, info, warn, error, fatal). Include relevant context information in log messages and use consistent formatting for easier parsing and analysis.*

3. What command-line tools are available for debugging ROS 2 systems?
   - *Answer: Key tools include `ros2 node list` (show active nodes), `ros2 topic list` (show active topics), `ros2 topic echo` (view topic data), `ros2 service call` (test services), and `rqt_graph` (visualize system graph).*

## Check Your Understanding
**Multiple Choice Questions:**

1. What is the purpose of unit testing in ROS 2?
   A) To test the entire robot system
   B) To test individual components in isolation  *(Correct)*
   C) To test hardware components only
   D) To test network connectivity

   *Explanation: Unit testing focuses on testing individual software components in isolation to ensure they function correctly before integration.*

2. Which ROS 2 command shows all active nodes?
   A) `ros2 topic list`
   B) `ros2 node list`  *(Correct)*
   C) `ros2 service list`
   D) `ros2 action list`

   *Explanation: The `ros2 node list` command displays all currently active ROS 2 nodes in the system.*

3. What is the recommended approach for logging in ROS 2?
   A) Use print statements only
   B) Use the built-in logger with appropriate severity levels  *(Correct)*
   C) Write directly to files
   D) Send logs through topics only

   *Explanation: ROS 2 provides a structured logging system with appropriate severity levels that integrates well with the ROS 2 ecosystem.*

<!-- RAG_CHUNK_ID: ros2-testing-debugging-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->