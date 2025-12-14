---
title: ROS 2 Nodes and Topics
description: Understanding the fundamental communication mechanisms in ROS 2
sidebar_position: 2
learning_outcomes:
  - Explain the concept and implementation of ROS 2 nodes
  - Understand the publish-subscribe communication pattern using topics
  - Implement custom message types and publishers/subscribers
---

# ROS 2 Nodes and Topics: The Foundation of Robot Communication

## Purpose
This chapter delves into the fundamental communication mechanisms in ROS 2: nodes and topics. You'll learn how nodes act as the basic computational units of a ROS 2 system and how topics enable asynchronous communication between them through the publish-subscribe pattern.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Explain the concept and implementation of ROS 2 nodes
- Understand the publish-subscribe communication pattern using topics
- Implement custom message types and publishers/subscribers

## Understanding ROS 2 Nodes

### What is a Node?
A node in ROS 2 is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 system, each typically responsible for a specific task or function.

Nodes provide a way to:
- Organize robot functionality into separate, manageable processes
- Enable modular development and debugging
- Facilitate distributed computation across multiple machines

<!-- RAG_CHUNK_ID: ros2-node-concept-definition -->

### Node Lifecycle
The lifecycle of a ROS 2 node typically includes:

1. **Initialization**: Setting up the node with `rclpy.init()` or similar
2. **Node Creation**: Creating the node instance with a unique name
3. **Entity Creation**: Setting up publishers, subscribers, services, etc.
4. **Execution**: Running the main loop or spinning to handle callbacks
5. **Cleanup**: Properly destroying the node and releasing resources

```python
import rclpy
from rclpy.node import Node

class LifecycleNode(Node):
    def __init__(self):
        # Node initialization with a name
        super().__init__('lifecycle_node')

        # Node-specific setup
        self.get_logger().info('Node initialized successfully')

        # Create entities (publishers, subscribers, etc.)
        self.publisher = self.create_publisher(String, 'topic', 10)

        # Create a timer for periodic execution
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info('Timer callback executed')

def main(args=None):
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the node
    node = LifecycleNode()

    # Spin to process callbacks
    rclpy.spin(node)

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-node-lifecycle-implementation -->

## Topics and Publish-Subscribe Pattern

### Topic Communication
Topics enable asynchronous communication between nodes using the publish-subscribe pattern:

- **Publisher**: A node that sends messages to a topic
- **Subscriber**: A node that receives messages from a topic
- **Topic**: The named channel over which messages are sent

This pattern allows for loose coupling between nodes - publishers don't need to know who subscribes to their messages, and subscribers don't need to know who publishes to their topics.

<!-- RAG_CHUNK_ID: ros2-topic-communication-pattern -->

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')

        # Create a publisher with a topic name and queue size
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer to publish messages periodically
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = TalkerNode()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Instructions:**
1. Create the file in your package: `my_robot_pkg/my_robot_pkg/talker.py`
2. Make it executable: `chmod +x my_robot_pkg/my_robot_pkg/talker.py`
3. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
4. Source the setup file: `source install/setup.bash`
5. Run the publisher: `ros2 run my_robot_pkg talker`

<!-- RAG_CHUNK_ID: ros2-publisher-implementation -->

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')

        # Create a subscription to the 'chatter' topic
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)  # QoS history depth
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = ListenerNode()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Instructions:**
1. Create the file in your package: `my_robot_pkg/my_robot_pkg/listener.py`
2. Make it executable: `chmod +x my_robot_pkg/my_robot_pkg/listener.py`
3. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
4. Source the setup file: `source install/setup.bash`
5. Run the subscriber: `ros2 run my_robot_pkg listener`

To see both nodes in action, run the publisher in one terminal and the subscriber in another.

<!-- RAG_CHUNK_ID: ros2-subscriber-implementation -->

## Custom Message Types

### Creating Custom Messages
While ROS 2 provides standard message types like `std_msgs`, you'll often need custom message types for your specific applications.

1. Create a `msg` directory in your package
2. Define your message in a `.msg` file
3. Update your `package.xml` and `setup.py` to include the message

Example custom message (`msg/Num.msg`):
```
int64 num
string description
float64[] values
```

### Using Custom Messages

```python
# Publisher with custom message
import rclpy
from rclpy.node import Node
from my_robot_pkg.msg import Num  # Custom message

class CustomTalker(Node):
    def __init__(self):
        super().__init__('custom_talker')
        self.publisher = self.create_publisher(Num, 'custom_topic', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = Num()
        msg.num = self.counter
        msg.description = f'Count: {self.counter}'
        msg.values = [1.0, 2.0, 3.0]
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.num}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = CustomTalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-custom-message-implementation -->

## Quality of Service (QoS) Settings

### Understanding QoS
Quality of Service settings allow you to control the delivery guarantees for messages on topics. Different applications may require different levels of reliability and performance.

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    depth=10,  # History depth
    reliability=ReliabilityPolicy.RELIABLE,  # Ensure all messages are delivered
    history=HistoryPolicy.KEEP_LAST  # Keep the last N messages
)

# Use the QoS profile when creating publisher/subscriber
publisher = self.create_publisher(String, 'topic', qos_profile)
```

<!-- RAG_CHUNK_ID: ros2-qos-settings -->

## Hands-on Exercise
Create a temperature monitoring system using ROS 2 nodes and topics with custom message types:

### Part 1: Custom Message Definition
1. Create a new package: `ros2 pkg create --build-type ament_python temp_monitor_pkg`
2. Navigate to the package directory: `cd ~/ros2_ws/src/temp_monitor_pkg`
3. Create a `msg` directory: `mkdir msg`
4. Create a custom message file `msg/Temperature.msg`:
```
float64 temperature_value
string unit
builtin_interfaces.msg.Time timestamp
string sensor_id
```
5. Update `package.xml` to include message generation dependencies:
```xml
<depend>builtin_interfaces</depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```
6. Update `setup.py` to include the message:

### Part 2: Publisher Implementation
Create a temperature publisher that simulates realistic temperature readings:
```python
# temp_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
from builtin_interfaces.msg import Time
from temp_monitor_pkg.msg import Temperature  # Update with your package name

class TemperaturePublisher(Node):
    def __init__(self):
        super().__init__('temperature_publisher')
        self.publisher = self.create_publisher(Temperature, 'temperature_topic', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.temperature_value = 20.0  # Starting temperature

    def timer_callback(self):
        msg = Temperature()
        msg.temperature_value = self.temperature_value + random.uniform(-2, 2)  # Add some variation
        msg.unit = 'Celsius'
        msg.timestamp = self.get_clock().now().to_msg()
        msg.sensor_id = 'temp_sensor_01'

        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing temperature: {msg.temperature_value:.2f} {msg.unit}')

        # Simulate temperature drift
        self.temperature_value += random.uniform(-0.5, 0.5)

def main(args=None):
    rclpy.init(args=args)
    temp_publisher = TemperaturePublisher()
    rclpy.spin(temp_publisher)
    temp_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Subscriber Implementation
Create a subscriber that monitors temperature and logs warnings:
```python
# temp_subscriber.py
import rclpy
from rclpy.node import Node
from temp_monitor_pkg.msg import Temperature  # Update with your package name

class TemperatureSubscriber(Node):
    def __init__(self):
        super().__init__('temperature_subscriber')
        self.subscription = self.create_subscription(
            Temperature,
            'temperature_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.threshold = 30.0  # Temperature threshold in Celsius

    def listener_callback(self, msg):
        self.get_logger().info(
            f'Received temperature: {msg.temperature_value:.2f} {msg.unit} '
            f'from sensor {msg.sensor_id} at {msg.timestamp.sec}.{msg.timestamp.nanosec}'
        )

        if msg.temperature_value > self.threshold:
            self.get_logger().warn(f'TEMPERATURE WARNING: {msg.temperature_value:.2f}°C exceeds threshold of {self.threshold}°C!')

def main(args=None):
    rclpy.init(args=args)
    temp_subscriber = TemperatureSubscriber()
    rclpy.spin(temp_subscriber)
    temp_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Testing
1. Build your workspace: `cd ~/ros2_ws && colcon build --packages-select temp_monitor_pkg`
2. Source the workspace: `source install/setup.bash`
3. Run the publisher in one terminal: `ros2 run temp_monitor_pkg temp_publisher`
4. Run the subscriber in another terminal: `ros2 run temp_monitor_pkg temp_subscriber`
5. Observe the temperature readings and warning messages when threshold is exceeded

### Expected Results
- The publisher should send temperature readings every second
- The subscriber should receive and log these readings
- When temperature exceeds 30°C, warning messages should appear
- This demonstrates custom message types and QoS concepts in ROS 2

<!-- RAG_CHUNK_ID: ros2-hands-on-exercise-nodes-topics -->

## Summary
Nodes and topics form the foundation of communication in ROS 2. Nodes provide the computational units for robot functionality, while topics enable asynchronous communication through the publish-subscribe pattern. Understanding these concepts is crucial for developing complex robotic systems.

## Further Reading
- [ROS 2 Nodes and Topics Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html)
- [Quality of Service in ROS 2](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
- [Custom Message Types](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html)

## Summary
Nodes and topics form the foundation of communication in ROS 2. Nodes provide the computational units for robot functionality, while topics enable asynchronous communication through the publish-subscribe pattern. Understanding these concepts is crucial for developing complex robotic systems.

## Practice Questions
1. What is the difference between a publisher and a subscriber in ROS 2?
   - *Answer: A publisher sends messages to a topic, while a subscriber receives messages from a topic. Publishers and subscribers are decoupled, meaning they don't need to know about each other.*

2. Explain the concept of Quality of Service (QoS) in ROS 2 and why it matters.
   - *Answer: QoS defines policies for message delivery, including reliability, durability, and liveliness. It matters because it allows developers to tune communication behavior based on requirements (e.g., reliable vs. best-effort delivery).*

3. When would you use custom message types instead of standard message types?
   - *Answer: Use custom message types when the standard types (String, Int32, Float64, etc.) don't meet your specific data structure needs, such as for complex sensor data or robot-specific information.*

## Check Your Understanding
**Multiple Choice Questions:**

1. What is the primary purpose of a node in ROS 2?
   A) To store robot data permanently
   B) To serve as an executable that uses ROS 2 to communicate with other nodes  *(Correct)*
   C) To manage robot hardware directly
   D) To provide a user interface for robot control

   *Explanation: Nodes are the fundamental building blocks of ROS 2 applications. They are executables that use the ROS 2 client library to communicate with other nodes in the system.*

2. In the publish-subscribe pattern, which statement is true?
   A) Publishers must know all subscribers
   B) Subscribers must know all publishers
   C) Publishers and subscribers are loosely coupled  *(Correct)*
   D) Communication is synchronous only

   *Explanation: In the publish-subscribe pattern, publishers and subscribers are decoupled in time, space, and synchronization. They don't need to know about each other's existence.*

3. What does QoS stand for in ROS 2?
   A) Quality of Service  *(Correct)*
   B) Quick Operating System
   C) Quantitative Observation System
   D) Quality Operation Standard

   *Explanation: Quality of Service (QoS) in ROS 2 defines policies for message delivery, allowing fine-tuning of communication behavior based on application requirements.*

<!-- RAG_CHUNK_ID: ros2-nodes-topics-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->