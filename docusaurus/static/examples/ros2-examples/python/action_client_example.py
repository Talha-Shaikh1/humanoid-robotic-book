#!/usr/bin/env python3

"""
ROS2 Action Client Example

This example demonstrates how to create an action client that sends goals
to an action server using ROS2 and rclpy.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):
    """
    A ROS2 action client node that sends goals to an action server.
    """

    def __init__(self):
        """
        Initialize the action client node.
        """
        super().__init__('fibonacci_action_client')

        # Create an action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

    def send_goal(self, order):
        """
        Send a goal to the action server.
        """
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create a goal message
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Send the goal and get a future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        # Set callbacks for when the goal is accepted
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Callback function that executes when the goal is accepted or rejected.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get the result future
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """
        Callback function that executes when feedback is received.
        """
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.sequence}')

    def get_result_callback(self, future):
        """
        Callback function that executes when the result is received.
        """
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')


def main(args=None):
    """
    Main function that initializes ROS2, creates the action client node,
    sends a goal, and handles the result.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the action client node
    fibonacci_action_client = FibonacciActionClient()

    # Send a goal
    fibonacci_action_client.send_goal(10)

    try:
        # Start spinning - this keeps the node alive and processes callbacks
        rclpy.spin(fibonacci_action_client)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up: destroy the node and shut down ROS2
        fibonacci_action_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()