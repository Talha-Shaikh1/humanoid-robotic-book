---
title: Introduction to ROS 2
description: An overview of ROS 2 as the robotic nervous system
sidebar_position: 1
learning_outcomes:
  - Understand the fundamental concepts of ROS 2
  - Identify the key differences between ROS 1 and ROS 2
  - Recognize the core components of the ROS 2 architecture
---

# Introduction to ROS 2: The Robotic Nervous System

## Purpose
This chapter introduces you to ROS 2 (Robot Operating System 2), which serves as the "nervous system" of modern robots. You'll learn the fundamental concepts that make ROS 2 the standard middleware for robotics development, enabling communication and coordination between different robotic components.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Understand the fundamental concepts of ROS 2
- Identify the key differences between ROS 1 and ROS 2
- Recognize the core components of the ROS 2 architecture

## What is ROS 2?
ROS 2 (Robot Operating System 2) is not an operating system, but rather a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

ROS 2 addresses many of the limitations of ROS 1, including:
- Improved real-time support
- Better security features
- Enhanced multi-robot support
- More reliable communication middleware

<!-- RAG_CHUNK_ID: ros2-definition-overview -->

## Key Differences from ROS 1
ROS 2 was developed to overcome the limitations of ROS 1. Here are the main differences:

### Middleware Architecture
- **ROS 1**: Uses a centralized master-based architecture
- **ROS 2**: Uses a distributed architecture based on DDS (Data Distribution Service)

### Real-time Support
- **ROS 1**: Limited real-time capabilities
- **ROS 2**: Enhanced real-time performance with better thread safety

### Security
- **ROS 1**: No built-in security
- **ROS 2**: Built-in security with authentication, access control, and encryption

### Multi-robot Support
- **ROS 1**: Challenging multi-robot deployments
- **ROS 2**: Improved support for multi-robot systems

<!-- RAG_CHUNK_ID: ros2-vs-ros1-comparison -->

## Core Architecture Components

### Nodes
In ROS 2, a node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 system.

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Instructions:**
1. Create a new directory for your package: `mkdir -p ~/ros2_ws/src/my_first_pkg`
2. Create the Python file in `~/ros2_ws/src/my_first_pkg/my_first_pkg/minimal_node.py`
3. Make the file executable: `chmod +x ~/ros2_ws/src/my_first_pkg/my_first_pkg/minimal_node.py`
4. Build your workspace: `cd ~/ros2_ws && colcon build`
5. Source the workspace: `source install/setup.bash`
6. Run the node: `ros2 run my_first_pkg minimal_node`

<!-- RAG_CHUNK_ID: ros2-node-basic-structure -->

### Topics
Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic.

```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-topic-publisher-example -->

### Services
Services provide a request/reply communication pattern, which is useful for operations that require a response.

```python
# Service example
from add_two_ints_srv.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-service-example -->

## Hands-on Exercise
Create a simple ROS 2 package that includes both a publisher and subscriber node to understand the publish/subscribe communication pattern:

### Part 1: Package Setup
1. Create a new package: `ros2 pkg create --build-type ament_python my_ros2_exercise`
2. Navigate to the package directory: `cd ~/ros2_ws/src/my_ros2_exercise/my_ros2_exercise`
3. Create two Python files: `publisher_node.py` and `subscriber_node.py`

### Part 2: Publisher Implementation
Create a publisher that sends "Hello from ROS 2" messages every 2 seconds:
```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'hello_topic', 10)
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello from ROS 2: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisher_node = PublisherNode()
    rclpy.spin(publisher_node)
    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Subscriber Implementation
Create a subscriber that listens to these messages and logs them:
```python
# subscriber_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    subscriber_node = SubscriberNode()
    rclpy.spin(subscriber_node)
    subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Testing
1. Make both files executable: `chmod +x publisher_node.py subscriber_node.py`
2. Build your workspace: `cd ~/ros2_ws && colcon build`
3. Source the workspace: `source install/setup.bash`
4. Run the publisher in one terminal: `ros2 run my_ros2_exercise publisher_node`
5. Run the subscriber in another terminal: `ros2 run my_ros2_exercise subscriber_node`
6. Observe the communication between nodes

### Expected Results
- The publisher should send messages every 2 seconds
- The subscriber should receive and log these messages
- This demonstrates the asynchronous publish/subscribe communication pattern in ROS 2

<!-- RAG_CHUNK_ID: ros2-hands-on-exercise-intro -->

## Summary
ROS 2 serves as the nervous system of modern robots, providing a robust framework for communication and coordination. Its distributed architecture, enhanced security, and real-time capabilities make it the standard for robotics development.

## Further Reading
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](http://docs.ros.org/en/humble/Tutorials.html)
- [DDS (Data Distribution Service) Overview](https://www.omg.org/omg-dds-portal/)

## Summary
ROS 2 serves as the nervous system of modern robots, providing a robust framework for communication and coordination. Its distributed architecture, enhanced security, and real-time capabilities make it the standard for robotics development.

## Practice Questions
1. What are the main differences between ROS 1 and ROS 2?
   - *Answer: ROS 2 uses a distributed architecture based on DDS, has better real-time support, includes built-in security features, and provides enhanced multi-robot support compared to ROS 1's centralized master-based architecture.*

2. Explain the role of nodes in ROS 2 architecture.
   - *Answer: Nodes are the fundamental building blocks of a ROS 2 system. They are executables that use ROS 2 to communicate with other nodes, enabling modular and distributed robot software development.*

3. When would you use topics versus services in ROS 2?
   - *Answer: Use topics for asynchronous, continuous data flow (publish/subscribe pattern) like sensor data or robot state. Use services for synchronous request/reply communication like performing a specific computation or task.*

## Check Your Understanding
**Multiple Choice Questions:**

1. What does DDS stand for in the context of ROS 2?
   A) Distributed Data System
   B) Data Distribution Service  *(Correct)*
   C) Dynamic Data Sharing
   D) Decentralized Data Structure

   *Explanation: DDS (Data Distribution Service) is the middleware that enables the distributed architecture of ROS 2, replacing the centralized master of ROS 1.*

2. Which of the following is NOT a key improvement of ROS 2 over ROS 1?
   A) Better real-time support
   B) Built-in security features
   C) Centralized master architecture  *(Correct)*
   D) Enhanced multi-robot support

   *Explanation: ROS 2 actually moved away from centralized master architecture to a distributed DDS-based architecture, which was one of its key improvements over ROS 1.*

3. What is the primary purpose of a node in ROS 2?
   A) To store robot data permanently
   B) To serve as an executable that uses ROS 2 to communicate with other nodes  *(Correct)*
   C) To manage robot hardware directly
   D) To provide a user interface for robot control

   *Explanation: Nodes are the fundamental building blocks of ROS 2 applications. They are executables that use the ROS 2 client library to communicate with other nodes in the system.*

<!-- RAG_CHUNK_ID: ros2-intro-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->