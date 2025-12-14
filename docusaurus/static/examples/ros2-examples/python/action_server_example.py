#!/usr/bin/env python3

"""
ROS2 Action Server Example

This example demonstrates how to create an action server that performs long-running tasks
with feedback using ROS2 and rclpy.
"""

import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):
    """
    A ROS2 action server node that performs Fibonacci sequence calculations.
    """

    def __init__(self):
        """
        Initialize the action server node.
        """
        super().__init__('fibonacci_action_server')

        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        """
        Callback function that decides whether to accept or reject a goal.
        """
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Callback function that decides whether to accept or reject a cancel request.
        """
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        Callback function that executes when a goal is requested.
        Performs the Fibonacci sequence calculation with feedback.
        """
        self.get_logger().info('Executing goal...')

        # Create feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Initialize the sequence
        feedback_msg.sequence = [0, 1]

        # Execute the action with feedback
        for i in range(1, goal_handle.request.order):
            # Check if the goal was canceled
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal canceled')
                result_msg.sequence = feedback_msg.sequence
                goal_handle.canceled()
                return result_msg

            # Update the sequence
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

            # Sleep to simulate work
            time.sleep(1)

        # Set the result
        goal_handle.succeed()
        result_msg.sequence = feedback_msg.sequence

        self.get_logger().info(f'Result: {result_msg.sequence}')
        return result_msg


def main(args=None):
    """
    Main function that initializes ROS2, creates the action server node, and starts spinning.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the action server node
    fibonacci_action_server = FibonacciActionServer()

    try:
        # Start spinning - this keeps the node alive and processes action requests
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up: destroy the node and shut down ROS2
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()