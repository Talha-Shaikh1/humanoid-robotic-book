#!/usr/bin/env python3

"""
ROS2 Service Server Example

This example demonstrates how to create a service server that responds to requests
using ROS2 and rclpy.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class ServiceServer(Node):
    """
    A ROS2 service server node that responds to addition requests.
    """

    def __init__(self):
        """
        Initialize the service server node and create a service.
        """
        super().__init__('service_server')

        # Create a service that listens on the 'add_two_ints' topic
        # The service will handle AddTwoInts requests
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        """
        Callback function that executes when a service request is received.
        Adds two integers from the request and returns the result in the response.
        """
        # Perform the addition
        response.sum = request.a + request.b

        # Log the request and response
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\nSum: {response.sum}')

        # Return the response
        return response


def main(args=None):
    """
    Main function that initializes ROS2, creates the service server node, and starts spinning.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the service server node
    service_server = ServiceServer()

    try:
        # Start spinning - this keeps the node alive and processes service requests
        rclpy.spin(service_server)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up: destroy the node and shut down ROS2
        service_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()