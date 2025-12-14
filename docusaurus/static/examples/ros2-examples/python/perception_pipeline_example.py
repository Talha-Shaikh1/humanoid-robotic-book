#!/usr/bin/env python3

"""
ROS2 Perception Pipeline Example

This example demonstrates a complete perception pipeline including
sensor data processing, computer vision, and object detection for robotics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
import cv2
import numpy as np
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point


class PerceptionPipeline(Node):
    """
    A complete perception pipeline node that processes sensor data
    and performs object detection and localization.
    """

    def __init__(self):
        """
        Initialize the perception pipeline node.
        """
        super().__init__('perception_pipeline')

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Create QoS profiles for different sensor types
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Create subscribers for different sensor modalities
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            sensor_qos
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            'lidar/points',
            self.pointcloud_callback,
            sensor_qos
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'lidar/scan',
            self.laser_callback,
            sensor_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            sensor_qos
        )

        # Create publishers for processed data
        self.object_detection_pub = self.create_publisher(
            Detection2DArray,
            'perception/object_detections',
            reliable_qos
        )

        self.segmented_image_pub = self.create_publisher(
            Image,
            'perception/segmented_image',
            sensor_qos
        )

        self.clustered_points_pub = self.create_publisher(
            PointCloud2,
            'perception/clustered_points',
            sensor_qos
        )

        # TF2 buffer and listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Store camera intrinsic parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Object detection parameters
        self.detection_confidence_threshold = 0.5
        self.min_object_size = 10  # pixels
        self.max_object_size = 10000  # pixels

        # Point cloud processing parameters
        self.cluster_eps = 0.1  # DBSCAN clustering epsilon
        self.cluster_min_samples = 5  # DBSCAN minimum samples

        # Create timer for periodic processing
        self.processing_timer = self.create_timer(0.1, self.process_sensors)

        # Internal buffers for sensor data
        self.latest_image = None
        self.latest_pointcloud = None
        self.latest_laser_scan = None
        self.latest_camera_info = None

        # Processing flags
        self.image_ready = False
        self.pointcloud_ready = False
        self.laser_ready = False

        self.get_logger().info('Perception pipeline initialized')

    def image_callback(self, msg):
        """
        Callback function for image messages.
        Stores the latest image for processing.
        """
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Store the image for processing
            self.latest_image = cv_image
            self.image_header = msg.header
            self.image_ready = True

            self.get_logger().debug(f'Image received: {cv_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

    def pointcloud_callback(self, msg):
        """
        Callback function for point cloud messages.
        Stores the latest point cloud for processing.
        """
        # Store point cloud data for processing
        self.latest_pointcloud = msg
        self.pointcloud_header = msg.header
        self.pointcloud_ready = True

        self.get_logger().debug(f'Point cloud received with {msg.width * msg.height} points')

    def laser_callback(self, msg):
        """
        Callback function for laser scan messages.
        Stores the latest laser scan for processing.
        """
        # Store laser scan data for processing
        self.latest_laser_scan = msg
        self.laser_header = msg.header
        self.laser_ready = True

        self.get_logger().debug(f'Laser scan received with {len(msg.ranges)} ranges')

    def camera_info_callback(self, msg):
        """
        Callback function for camera info messages.
        Stores camera intrinsic parameters for 3D reconstruction.
        """
        # Store camera intrinsic parameters
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.latest_camera_info = msg

        self.get_logger().info('Camera intrinsics updated')

    def process_sensors(self):
        """
        Main processing function that runs periodically to process sensor data.
        """
        if self.image_ready:
            # Process image for object detection
            detections = self.detect_objects_in_image(self.latest_image)

            # Publish detections
            if detections:
                self.publish_detections(detections)

                # Publish segmented image
                segmented_img = self.create_segmented_image(self.latest_image, detections)
                self.publish_segmented_image(segmented_img)

        if self.pointcloud_ready:
            # Process point cloud for object clustering
            clusters = self.cluster_pointcloud_objects(self.latest_pointcloud)

            # Publish clustered points
            if clusters:
                self.publish_clustered_points(clusters)

        if self.laser_ready:
            # Process laser scan for obstacle detection
            obstacles = self.detect_obstacles_in_laser(self.latest_laser_scan)

            # This could be integrated with other detections
            self.process_laser_obstacles(obstacles)

    def detect_objects_in_image(self, image):
        """
        Detect objects in the image using computer vision techniques.
        This is a simplified implementation - in practice, this would use
        deep learning models or more sophisticated algorithms.
        """
        # Convert image to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to highlight objects
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on size and shape
        detections = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            # Filter based on size
            if self.min_object_size < area < self.max_object_size:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Create detection object
                detection = {
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour,
                    'confidence': min(0.9, area / self.max_object_size)  # Confidence based on size
                }

                detections.append(detection)

        self.get_logger().debug(f'Detected {len(detections)} objects in image')
        return detections

    def cluster_pointcloud_objects(self, pointcloud_msg):
        """
        Cluster objects in the point cloud using DBSCAN algorithm.
        """
        # Extract points from point cloud message
        points = []
        for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        if len(points) < self.cluster_min_samples:
            return []

        # Perform clustering using DBSCAN
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        cluster_labels = clustering.fit_predict(points)

        # Group points by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indicates noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(points[i])

        self.get_logger().debug(f'Clustered {len(clusters)} objects from point cloud')
        return clusters

    def detect_obstacles_in_laser(self, laser_msg):
        """
        Detect obstacles in laser scan data.
        """
        obstacles = []

        for i, range_val in enumerate(laser_msg.ranges):
            if laser_msg.range_min <= range_val <= laser_msg.range_max:
                # Calculate angle for this range measurement
                angle = laser_msg.angle_min + i * laser_msg.angle_increment

                # Convert polar to Cartesian coordinates
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                # Check if this is an obstacle (within certain distance)
                if range_val < 1.0:  # Consider anything closer than 1m as obstacle
                    obstacles.append({'x': x, 'y': y, 'range': range_val, 'angle': angle})

        self.get_logger().debug(f'Detected {len(obstacles)} obstacles in laser scan')
        return obstacles

    def fuse_sensor_data(self, image_detections, pointcloud_clusters, laser_obstacles):
        """
        Fuse data from multiple sensors to create a unified perception of the environment.
        This is a simplified fusion approach - in practice, more sophisticated
        probabilistic or deep learning methods would be used.
        """
        # Convert 2D image detections to 3D if camera intrinsics are available
        if self.camera_matrix is not None and image_detections:
            # Project image coordinates to 3D space using depth information
            # This would require depth data from stereo camera or RGB-D sensor
            pass

        # Associate point cloud clusters with laser obstacles
        fused_objects = []

        for cluster_id, cluster_points in pointcloud_clusters.items():
            # Calculate centroid of cluster
            centroid = np.mean(cluster_points, axis=0)

            # Find corresponding laser obstacles
            for laser_obstacle in laser_obstacles:
                # Simple distance check between 3D centroid and 2D laser obstacle
                distance_2d = np.sqrt((centroid[0] - laser_obstacle['x'])**2 +
                                     (centroid[1] - laser_obstacle['y'])**2)

                if distance_2d < 0.2:  # Associate if within 20cm
                    fused_objects.append({
                        'type': 'fused',
                        'position': centroid,
                        'laser_data': laser_obstacle,
                        'cluster_id': cluster_id
                    })

        return fused_objects

    def publish_detections(self, detections):
        """
        Publish object detections to ROS2 topic.
        """
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = self.image_header.frame_id if hasattr(self, 'image_header') else 'camera_link'

        for detection in detections:
            detection_msg = Detection2D()

            # Set bounding box
            bbox = detection['bbox']
            detection_msg.bbox.center.x = bbox[0] + bbox[2] / 2  # Center x
            detection_msg.bbox.center.y = bbox[1] + bbox[3] / 2  # Center y
            detection_msg.bbox.size_x = bbox[2]  # Width
            detection_msg.bbox.size_y = bbox[3]  # Height

            # Add hypothesis with confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = "object"  # Generic object type
            hypothesis.score = detection['confidence']
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        self.object_detection_pub.publish(detection_array)

    def create_segmented_image(self, image, detections):
        """
        Create a segmented image with bounding boxes drawn around detected objects.
        """
        # Create a copy of the image to draw on
        segmented_img = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox

            # Draw bounding box
            cv2.rectangle(segmented_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw confidence text
            confidence_text = f"Conf: {detection['confidence']:.2f}"
            cv2.putText(segmented_img, confidence_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return segmented_img

    def publish_segmented_image(self, segmented_image):
        """
        Publish the segmented image to ROS2 topic.
        """
        try:
            # Convert OpenCV image to ROS Image message
            segmented_img_msg = self.cv_bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
            segmented_img_msg.header.stamp = self.get_clock().now().to_msg()
            segmented_img_msg.header.frame_id = self.image_header.frame_id if hasattr(self, 'image_header') else 'camera_link'

            self.segmented_image_pub.publish(segmented_img_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing segmented image: {str(e)}')

    def publish_clustered_points(self, clusters):
        """
        Publish clustered points to ROS2 topic.
        """
        # This would convert clusters back to PointCloud2 format
        # For simplicity, we'll just log the cluster information
        for cluster_id, points in clusters.items():
            self.get_logger().info(f'Cluster {cluster_id}: {len(points)} points')

    def process_laser_obstacles(self, obstacles):
        """
        Process laser obstacles for navigation and planning.
        """
        # This could send obstacles to navigation stack or collision avoidance
        for obstacle in obstacles:
            self.get_logger().debug(f'Obstacle at ({obstacle["x"]:.2f}, {obstacle["y"]:.2f})')

    def transform_point_to_frame(self, point, target_frame, source_frame):
        """
        Transform a point from one frame to another using TF2.
        """
        try:
            # Create a PointStamped message
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]

            # Look up the transform
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

            # Transform the point
            transformed_point = do_transform_point(point_stamped, transform)
            return [transformed_point.point.x,
                   transformed_point.point.y,
                   transformed_point.point.z]
        except TransformException as ex:
            self.get_logger().error(f'Could not transform point: {ex}')
            return None


class ObjectTracker(Node):
    """
    An object tracker that maintains consistent IDs for objects across frames.
    """

    def __init__(self):
        """
        Initialize the object tracker node.
        """
        super().__init__('object_tracker')

        # Create subscriber for detections
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            'perception/object_detections',
            self.detection_callback,
            10
        )

        # Create publisher for tracked objects
        self.tracked_objects_pub = self.create_publisher(
            Detection2DArray,
            'perception/tracked_objects',
            10
        )

        # Object tracking parameters
        self.max_displacement = 0.5  # Maximum displacement between frames
        self.track_lifetime = 10  # Number of frames before removing track
        self.min_track_confidence = 0.3  # Minimum confidence to maintain track

        # Track objects across frames
        self.tracks = {}  # Dictionary to store object tracks
        self.next_track_id = 0

        # Timer for periodic tracking maintenance
        self.tracking_timer = self.create_timer(0.1, self.maintain_tracks)

    def detection_callback(self, msg):
        """
        Callback function for incoming detections to track objects.
        """
        # Update existing tracks with new detections
        updated_detections = self.update_tracks(msg.detections)

        # Create new tracks for unassigned detections
        self.create_new_tracks(updated_detections)

        # Publish tracked objects
        self.publish_tracked_objects()

    def update_tracks(self, detections):
        """
        Update existing tracks with new detections using nearest neighbor association.
        """
        # List to store detections that weren't matched to existing tracks
        unmatched_detections = []

        for detection in detections:
            # Calculate distances to existing tracks
            min_distance = float('inf')
            best_track_id = None

            for track_id, track_info in self.tracks.items():
                # Calculate distance between detection and track
                det_center = (detection.bbox.center.x, detection.bbox.center.y)
                track_center = track_info['position']

                distance = np.sqrt((det_center[0] - track_center[0])**2 +
                                  (det_center[1] - track_center[1])**2)

                if distance < min_distance and distance < self.max_displacement:
                    min_distance = distance
                    best_track_id = track_id

            # If we found a matching track
            if best_track_id is not None:
                # Update track with new information
                self.tracks[best_track_id]['position'] = (
                    detection.bbox.center.x,
                    detection.bbox.center.y
                )
                self.tracks[best_track_id]['confidence'] = detection.results[0].score
                self.tracks[best_track_id]['last_seen'] = self.get_clock().now().nanoseconds
            else:
                # Add to unmatched detections for new track creation
                unmatched_detections.append(detection)

        return unmatched_detections

    def create_new_tracks(self, detections):
        """
        Create new tracks for detections that weren't matched to existing tracks.
        """
        for detection in detections:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                'position': (detection.bbox.center.x, detection.bbox.center.y),
                'confidence': detection.results[0].score,
                'first_seen': self.get_clock().now().nanoseconds,
                'last_seen': self.get_clock().now().nanoseconds,
                'lifetime': 0
            }

    def maintain_tracks(self):
        """
        Remove old tracks and update lifetime counters.
        """
        current_time = self.get_clock().now().nanoseconds

        # Remove tracks that haven't been seen recently
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            time_since_seen = (current_time - track_info['last_seen']) / 1e9  # Convert to seconds

            if time_since_seen > self.track_lifetime:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def publish_tracked_objects(self):
        """
        Publish tracked objects with consistent IDs.
        """
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'  # Use appropriate frame

        for track_id, track_info in self.tracks.items():
            detection_msg = Detection2D()

            # Set bounding box (use previous size or default)
            detection_msg.bbox.center.x = track_info['position'][0]
            detection_msg.bbox.center.y = track_info['position'][1]
            detection_msg.bbox.size_x = 50  # Default size
            detection_msg.bbox.size_y = 50  # Default size

            # Add hypothesis with track ID and confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(track_id)
            hypothesis.score = track_info['confidence']
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        self.tracked_objects_pub.publish(detection_array)


def main(args=None):
    """
    Main function that initializes ROS2, creates perception nodes, and starts spinning.
    """
    rclpy.init(args=args)

    # Create perception pipeline node
    perception_node = PerceptionPipeline()

    # Create object tracker node
    tracker_node = ObjectTracker()

    try:
        # Create executor and add both nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(perception_node)
        executor.add_node(tracker_node)

        # Spin both nodes
        executor.spin()
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down perception pipeline')
        tracker_node.get_logger().info('Shutting down object tracker')
    finally:
        perception_node.destroy_node()
        tracker_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()