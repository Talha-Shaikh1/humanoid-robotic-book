#!/usr/bin/env python3

"""
ROS2 Subscriber Example - Listener

This example demonstrates how to create a subscriber that receives messages
from a topic using ROS2 and rclpy.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    """
    A ROS2 subscriber node that receives messages from a topic.
    """

    def __init__(self):
        """
        Initialize the listener node and create a subscriber.
        """
        super().__init__('listener')

        # Create a subscriber for the 'chatter' topic
        # The subscriber will receive String messages
        # The queue size is set to 10
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)

        # Prevent unused variable warning
        self.subscription  # Not technically needed but good practice

    def listener_callback(self, msg):
        """
        Callback function that executes when a message is received.
        Prints the received message.
        """
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """
    Main function that initializes ROS2, creates the listener node, and starts spinning.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the listener node
    listener = Listener()

    try:
        # Start spinning - this keeps the node alive and processes callbacks
        rclpy.spin(listener)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up: destroy the node and shut down ROS2
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()