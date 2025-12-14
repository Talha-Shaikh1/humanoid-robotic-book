#!/usr/bin/env python3

"""
Basic ROS2 Node Example

This example demonstrates the fundamental structure of a ROS2 node using rclpy.
It creates a simple node that logs messages at regular intervals.
"""

import rclpy
from rclpy.node import Node


class MinimalNode(Node):
    """
    A minimal ROS2 node example that demonstrates the basic structure
    and lifecycle of a ROS2 node.
    """

    def __init__(self):
        """
        Initialize the node with the name 'minimal_node'.
        This is where you set up publishers, subscribers, timers, etc.
        """
        super().__init__('minimal_node')

        # Create a timer that triggers a callback every 0.5 seconds
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Counter to track number of messages
        self.counter = 0

        # Log that the node has started
        self.get_logger().info('Minimal node started')

    def timer_callback(self):
        """
        Callback function that executes every time the timer fires.
        This is where the node's main work happens.
        """
        self.get_logger().info(f'Hello World: {self.counter}')
        self.counter += 1


def main(args=None):
    """
    Main function that initializes ROS2, creates the node, and starts spinning.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the node
    minimal_node = MinimalNode()

    try:
        # Start spinning - this keeps the node alive and processes callbacks
        rclpy.spin(minimal_node)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up: destroy the node and shut down ROS2
        minimal_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()