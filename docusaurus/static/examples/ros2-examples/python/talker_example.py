#!/usr/bin/env python3

"""
ROS2 Publisher Example - Talker

This example demonstrates how to create a publisher that sends messages
over a topic using ROS2 and rclpy.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):
    """
    A ROS2 publisher node that sends messages to a topic.
    """

    def __init__(self):
        """
        Initialize the talker node and create a publisher.
        """
        super().__init__('talker')

        # Create a publisher for the 'chatter' topic
        # The publisher will send String messages
        # The queue size is set to 10
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Set the timer period to 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize a counter
        self.i = 0

    def timer_callback(self):
        """
        Callback function that executes every time the timer fires.
        Creates and publishes a message.
        """
        # Create a String message
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # Publish the message
        self.publisher.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.i += 1


def main(args=None):
    """
    Main function that initializes ROS2, creates the talker node, and starts spinning.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the talker node
    talker = Talker()

    try:
        # Start spinning - this keeps the node alive and processes callbacks
        rclpy.spin(talker)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up: destroy the node and shut down ROS2
        talker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()