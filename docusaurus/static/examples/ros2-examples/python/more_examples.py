#!/usr/bin/env python3

"""
Additional ROS2 Python Examples

This file contains additional examples demonstrating various ROS2 concepts
including parameters, custom messages, and more complex node interactions.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String, Int32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math


class ParameterNode(Node):
    """
    A ROS2 node that demonstrates parameter usage.
    """

    def __init__(self):
        """
        Initialize the parameter node.
        """
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('debug_mode', False)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_speed = self.get_parameter('max_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.debug_mode = self.get_parameter('debug_mode').value

        # Set up a parameter callback for dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Create a timer to periodically log parameter values
        self.timer = self.create_timer(2.0, self.log_parameters)

        self.get_logger().info(f'Initialized with robot_name: {self.robot_name}')

    def parameter_callback(self, params):
        """
        Callback function for parameter changes.
        """
        for param in params:
            if param.name == 'max_speed' and param.type_ == Parameter.Type.DOUBLE:
                self.max_speed = param.value
                self.get_logger().info(f'Max speed updated to: {self.max_speed}')
            elif param.name == 'debug_mode' and param.type_ == Parameter.Type.BOOL:
                self.debug_mode = param.value
                status = 'enabled' if self.debug_mode else 'disabled'
                self.get_logger().info(f'Debug mode {status}')

        return SetParametersResult(successful=True)

    def log_parameters(self):
        """
        Periodically log parameter values.
        """
        if self.debug_mode:
            self.get_logger().info(
                f'Parameters - Robot: {self.robot_name}, '
                f'Max Speed: {self.max_speed}, '
                f'Safety Distance: {self.safety_distance}'
            )


class SimpleNavigator(Node):
    """
    A ROS2 node that demonstrates navigation concepts.
    """

    def __init__(self):
        """
        Initialize the navigator node.
        """
        super().__init__('simple_navigator')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create subscriber for laser scan data
        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        # Create subscriber for goal commands
        self.goal_sub = self.create_subscription(
            String,
            'navigation_goal',
            self.goal_callback,
            10
        )

        # Navigation state
        self.current_goal = None
        self.obstacle_detected = False
        self.safety_distance = 0.5  # meters

        # Create a timer for navigation control
        self.nav_timer = self.create_timer(0.1, self.navigation_control)

        self.get_logger().info('Simple Navigator initialized')

    def laser_callback(self, msg):
        """
        Callback function for laser scan data.
        """
        # Check for obstacles in front of the robot
        min_distance = float('inf')
        front_scan_start = len(msg.ranges) // 2 - 10
        front_scan_end = len(msg.ranges) // 2 + 10

        for i in range(front_scan_start, front_scan_end):
            if 0 < msg.ranges[i] < min_distance:
                min_distance = msg.ranges[i]

        self.obstacle_detected = min_distance < self.safety_distance

        if self.obstacle_detected and self.debug_mode:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def goal_callback(self, msg):
        """
        Callback function for navigation goals.
        """
        self.current_goal = msg.data
        self.get_logger().info(f'Received navigation goal: {self.current_goal}')

    def navigation_control(self):
        """
        Main navigation control loop.
        """
        cmd = Twist()

        if self.obstacle_detected:
            # Stop if obstacle detected
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn('Obstacle detected! Stopping.')
        elif self.current_goal:
            # Simple navigation behavior based on goal
            if 'forward' in self.current_goal.lower():
                cmd.linear.x = 0.5  # Move forward at 0.5 m/s
            elif 'turn' in self.current_goal.lower():
                cmd.angular.z = 0.5  # Turn at 0.5 rad/s
            elif 'stop' in self.current_goal.lower():
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.current_goal = None
            else:
                # Default behavior: stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        else:
            # No goal, stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Publish command
        self.cmd_vel_pub.publish(cmd)


class MessageAggregator(Node):
    """
    A ROS2 node that demonstrates message aggregation from multiple topics.
    """

    def __init__(self):
        """
        Initialize the message aggregator node.
        """
        super().__init__('message_aggregator')

        # Create subscribers for different topics
        self.sub1 = self.create_subscription(
            String,
            'topic1',
            self.topic1_callback,
            10
        )

        self.sub2 = self.create_subscription(
            Int32,
            'topic2',
            self.topic2_callback,
            10
        )

        # Create publisher for aggregated messages
        self.pub = self.create_publisher(String, 'aggregated_topic', 10)

        # Storage for incoming messages
        self.last_msg1 = None
        self.last_msg2 = None

        # Create timer to aggregate and publish messages
        self.timer = self.create_timer(1.0, self.aggregate_and_publish)

        self.get_logger().info('Message Aggregator initialized')

    def topic1_callback(self, msg):
        """
        Callback for topic 1 messages.
        """
        self.last_msg1 = msg.data

    def topic2_callback(self, msg):
        """
        Callback for topic 2 messages.
        """
        self.last_msg2 = msg.data

    def aggregate_and_publish(self):
        """
        Aggregate messages and publish to output topic.
        """
        if self.last_msg1 is not None and self.last_msg2 is not None:
            aggregated_msg = f'Message1: {self.last_msg1}, Message2: {self.last_msg2}'
            output_msg = String()
            output_msg.data = aggregated_msg
            self.pub.publish(output_msg)
            self.get_logger().info(f'Published aggregated message: {aggregated_msg}')


class ServiceBasedCalculator(Node):
    """
    A ROS2 node that provides mathematical operations as services.
    """

    def __init__(self):
        """
        Initialize the calculator node.
        """
        super().__init__('service_based_calculator')

        # Create services for different operations
        from example_interfaces.srv import AddTwoInts, Trigger
        from std_srvs.srv import SetBool

        self.add_srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)
        self.multiply_srv = self.create_service(AddTwoInts, 'multiply_two_ints', self.multiply_callback)
        self.toggle_srv = self.create_service(SetBool, 'toggle_status', self.toggle_callback)

        # Internal state for toggle service
        self.status = False

        self.get_logger().info('Service-Based Calculator initialized')

    def add_callback(self, request, response):
        """
        Service callback for addition.
        """
        result = request.a + request.b
        response.sum = result
        self.get_logger().info(f'Addition: {request.a} + {request.b} = {result}')
        return response

    def multiply_callback(self, request, response):
        """
        Service callback for multiplication.
        """
        result = request.a * request.b
        response.sum = result  # Using sum field to store result
        self.get_logger().info(f'Multiplication: {request.a} * {request.b} = {result}')
        return response

    def toggle_callback(self, request, response):
        """
        Service callback for toggling status.
        """
        self.status = request.data
        response.success = True
        response.message = f'Status set to {self.status}'
        self.get_logger().info(f'Toggle: {request.data} -> Status: {self.status}')
        return response


def main_parameter_node(args=None):
    """
    Main function for the parameter node.
    """
    rclpy.init(args=args)

    param_node = ParameterNode()

    try:
        rclpy.spin(param_node)
    except KeyboardInterrupt:
        pass
    finally:
        param_node.destroy_node()
        rclpy.shutdown()


def main_navigator_node(args=None):
    """
    Main function for the navigator node.
    """
    rclpy.init(args=args)

    navigator_node = SimpleNavigator()

    try:
        rclpy.spin(navigator_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigator_node.destroy_node()
        rclpy.shutdown()


def main_aggregator_node(args=None):
    """
    Main function for the message aggregator node.
    """
    rclpy.init(args=args)

    agg_node = MessageAggregator()

    try:
        rclpy.spin(agg_node)
    except KeyboardInterrupt:
        pass
    finally:
        agg_node.destroy_node()
        rclpy.shutdown()


def main_calculator_node(args=None):
    """
    Main function for the service-based calculator node.
    """
    rclpy.init(args=args)

    calc_node = ServiceBasedCalculator()

    try:
        rclpy.spin(calc_node)
    except KeyboardInterrupt:
        pass
    finally:
        calc_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # This file contains multiple main functions for different node types
    # Run one of them depending on the desired functionality
    import sys
    from rcl_interfaces.srv import SetParametersResult

    if len(sys.argv) > 1:
        if sys.argv[1] == 'parameter':
            main_parameter_node()
        elif sys.argv[1] == 'navigator':
            main_navigator_node()
        elif sys.argv[1] == 'aggregator':
            main_aggregator_node()
        elif sys.argv[1] == 'calculator':
            main_calculator_node()
        else:
            print("Usage: python more_examples.py [parameter|navigator|aggregator|calculator]")
    else:
        # Default to parameter node if no argument provided
        main_parameter_node()