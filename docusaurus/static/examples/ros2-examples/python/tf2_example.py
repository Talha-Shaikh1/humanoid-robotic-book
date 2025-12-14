#!/usr/bin/env python3

"""
ROS2 TF2 Transformation Example

This example demonstrates how to use TF2 for coordinate transformations
in ROS2, which is essential for robotics applications.
"""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, PointStamped
from tf2_geometry_msgs import do_transform_point


class TF2Publisher(Node):
    """
    A ROS2 node that publishes coordinate transformations using TF2.
    """

    def __init__(self):
        """
        Initialize the TF2 publisher node.
        """
        super().__init__('tf2_publisher')

        # Create a transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create a timer to broadcast transforms periodically
        self.timer = self.create_timer(0.1, self.broadcast_transform)

        # Sample counter for dynamic transforms
        self.sample_counter = 0

    def broadcast_transform(self):
        """
        Broadcast a sample transform from 'world' to 'robot' frame.
        """
        # Create a transform message
        t = TransformStamped()

        # Fill header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'robot'

        # Define transform (position and orientation)
        t.transform.translation.x = 1.0 * (self.sample_counter % 100) / 100.0  # Oscillates between 0 and 1
        t.transform.translation.y = 0.5 * (self.sample_counter % 50) / 50.0   # Oscillates between 0 and 0.5
        t.transform.translation.z = 0.0

        # Simple rotation (identity quaternion for no rotation)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

        # Increment counter for dynamic effect
        self.sample_counter += 1


class TF2Listener(Node):
    """
    A ROS2 node that listens to coordinate transformations using TF2.
    """

    def __init__(self):
        """
        Initialize the TF2 listener node.
        """
        super().__init__('tf2_listener')

        # Create a transform buffer
        self.tf_buffer = Buffer()

        # Create a transform listener
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create a timer to lookup transforms periodically
        self.timer = self.create_timer(1.0, self.lookup_transform)

    def lookup_transform(self):
        """
        Lookup and print the transform from 'world' to 'robot' frame.
        """
        try:
            # Look up the transform
            trans = self.tf_buffer.lookup_transform(
                'world',  # Target frame
                'robot',  # Source frame
                rclpy.time.Time(),  # Time (0 for latest)
                timeout=rclpy.duration.Duration(seconds=1.0)  # Timeout
            )

            # Print the transform
            self.get_logger().info(
                f'Transform from robot to world: '
                f'x={trans.transform.translation.x:.2f}, '
                f'y={trans.transform.translation.y:.2f}, '
                f'z={trans.transform.translation.z:.2f}'
            )

        except Exception as ex:
            # Handle exception if transform is not available
            self.get_logger().warn(f'Could not transform: {str(ex)}')


class TF2PointTransformer(Node):
    """
    A ROS2 node that demonstrates transforming points between coordinate frames.
    """

    def __init__(self):
        """
        Initialize the TF2 point transformer node.
        """
        super().__init__('tf2_point_transformer')

        # Create a transform buffer
        self.tf_buffer = Buffer()

        # Create a transform listener
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create a timer to transform a point periodically
        self.timer = self.create_timer(2.0, self.transform_point)

    def transform_point(self):
        """
        Transform a point from one frame to another.
        """
        # Define a point in the 'robot' frame
        point_stamped = PointStamped()
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.header.frame_id = 'robot'
        point_stamped.point.x = 1.0
        point_stamped.point.y = 0.0
        point_stamped.point.z = 0.0

        try:
            # Look up the transform
            transform = self.tf_buffer.lookup_transform(
                'world',  # Target frame
                'robot',  # Source frame
                rclpy.time.Time(),  # Time (0 for latest)
                timeout=rclpy.duration.Duration(seconds=1.0)  # Timeout
            )

            # Transform the point
            transformed_point = do_transform_point(point_stamped, transform)

            # Print the result
            self.get_logger().info(
                f'Original point in robot frame: ({point_stamped.point.x}, {point_stamped.point.y}, {point_stamped.point.z})'
                f' -> Transformed point in world frame: ({transformed_point.point.x:.2f}, {transformed_point.point.y:.2f}, {transformed_point.point.z:.2f})'
            )

        except Exception as ex:
            # Handle exception if transform is not available
            self.get_logger().warn(f'Could not transform point: {str(ex)}')


def main_publisher(args=None):
    """
    Main function for the TF2 publisher node.
    """
    rclpy.init(args=args)

    tf_publisher = TF2Publisher()

    try:
        rclpy.spin(tf_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        tf_publisher.destroy_node()
        rclpy.shutdown()


def main_listener(args=None):
    """
    Main function for the TF2 listener node.
    """
    rclpy.init(args=args)

    tf_listener = TF2Listener()

    try:
        rclpy.spin(tf_listener)
    except KeyboardInterrupt:
        pass
    finally:
        tf_listener.destroy_node()
        rclpy.shutdown()


def main_transformer(args=None):
    """
    Main function for the TF2 point transformer node.
    """
    rclpy.init(args=args)

    tf_transformer = TF2PointTransformer()

    try:
        rclpy.spin(tf_transformer)
    except KeyboardInterrupt:
        pass
    finally:
        tf_transformer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # This file contains multiple main functions for different node types
    # Run one of them depending on the desired functionality
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'publisher':
            main_publisher()
        elif sys.argv[1] == 'listener':
            main_listener()
        elif sys.argv[1] == 'transformer':
            main_transformer()
        else:
            print("Usage: python tf2_example.py [publisher|listener|transformer]")
    else:
        # Default to publisher if no argument provided
        main_publisher()