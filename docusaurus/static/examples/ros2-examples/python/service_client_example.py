#!/usr/bin/env python3

"""
ROS2 Service Client Example

This example demonstrates how to create a service client that sends requests
to a service server using ROS2 and rclpy.
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class ServiceClient(Node):
    """
    A ROS2 service client node that sends requests to a service server.
    """

    def __init__(self):
        """
        Initialize the service client node and create a client.
        """
        super().__init__('service_client')

        # Create a client for the 'add_two_ints' service
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create a request
        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        """
        Send a request to the service server with two integer values.
        """
        # Set the values in the request
        self.request.a = a
        self.request.b = b

        # Call the service asynchronously
        self.future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)

        # Return the result
        return self.future.result()


def main(args=None):
    """
    Main function that initializes ROS2, creates the service client node,
    sends a request, and handles the response.
    """
    # Initialize the ROS2 client library
    rclpy.init(args=args)

    # Create an instance of the service client node
    service_client = ServiceClient()

    # Get the values from command line arguments (or use defaults)
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Send the request and get the response
    response = service_client.send_request(a, b)

    # Print the result
    service_client.get_logger().info(f'Result of {a} + {b} = {response.sum}')

    # Clean up: destroy the node and shut down ROS2
    service_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()