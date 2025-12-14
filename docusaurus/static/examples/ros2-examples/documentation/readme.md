# ROS2 Examples Documentation

This directory contains example implementations demonstrating various ROS2 concepts for humanoid robotics applications. Each example is designed to illustrate specific functionality relevant to physical AI and humanoid robotics.

## Directory Structure

```
ros2-examples/
├── python/                 # Python-based ROS2 examples
│   ├── basic_nodes/        # Basic node creation and communication
│   ├── tf2_examples/       # Coordinate transformation examples
│   ├── action_examples/    # Action-based communication patterns
│   ├── launch_examples/    # Launch file examples
│   └── control_examples/   # Robot control examples
├── cpp/                    # C++-based ROS2 examples
├── configs/                # Configuration files
├── meshes/                 # 3D model files for simulation
└── worlds/                 # Gazebo world files
```

## Python Examples

### Basic Nodes
- `publisher_subscriber.py`: Demonstrates basic publisher-subscriber communication
- `service_server_client.py`: Shows service-based request-response communication
- `parameter_examples.py`: Illustrates parameter management in ROS2 nodes

### TF2 Examples
- `tf2_broadcaster.py`: Shows how to broadcast coordinate transformations
- `tf2_listener.py`: Demonstrates listening to coordinate transformations
- `tf2_point_transformer.py`: Example of transforming points between frames

### Action Examples
- `action_server_example.py`: Implements an action server for long-running tasks
- `action_client_example.py`: Shows how to create an action client
- `fibonacci_action_server.py`: Complete example of Fibonacci sequence calculation with feedback

### Launch Examples
- `robot_simulation.launch.py`: Complete example for launching robot simulation
- `multi_robot.launch.py`: Example of launching multiple robots simultaneously
- `simulation_with_rviz.launch.py`: Launch file with both Gazebo and RViz

### Control Examples
- `joint_state_controller.py`: Example of controlling robot joints
- `trajectory_controller.py`: Demonstrates trajectory-based motion control
- `balance_controller.py`: Example of balance control for bipedal robots

## C++ Examples

### Basic Functionality
- `minimal_publisher.cpp`: C++ equivalent of basic publisher
- `minimal_subscriber.cpp`: C++ equivalent of basic subscriber
- `minimal_service.cpp`: C++ service implementation
- `minimal_action_server.cpp`: C++ action server example

## Configuration Files

### Control Configurations
- `control_config.yaml`: Complete configuration for ROS2 control system
- `joint_limits.yaml`: Joint position, velocity, and effort limits
- `pid_gains.yaml`: PID controller gains for different joints

### URDF Models
- `simple_humanoid.urdf`: Basic humanoid robot model for examples
- `sensor_configs.urdf`: URDF extensions with sensor configurations

## Getting Started

### Prerequisites
- ROS2 Humble Hawksbill installed
- Gazebo Garden or Fortress
- Python 3.10+
- Appropriate ROS2 packages for humanoid robotics

### Setup
1. Source your ROS2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Build your workspace:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select ros2_examples
   source install/setup.bash
   ```

### Running Examples

#### Basic Publisher-Subscriber
```bash
# Terminal 1: Start the publisher
ros2 run ros2_examples talker

# Terminal 2: Start the subscriber
ros2 run ros2_examples listener
```

#### TF2 Example
```bash
# Terminal 1: Start the TF2 broadcaster
ros2 run ros2_examples tf2_broadcaster

# Terminal 2: Start the TF2 listener
ros2 run ros2_examples tf2_listener
```

#### Action Example
```bash
# Terminal 1: Start the action server
ros2 run ros2_examples fibonacci_action_server

# Terminal 2: Send a goal to the server
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 5}"
```

#### Launch Example
```bash
# Launch the complete robot simulation
ros2 launch ros2_examples robot_simulation.launch.py
```

## Best Practices Demonstrated

### Node Design
- Proper inheritance from `rclpy.Node`
- Use of `rclpy.spin()` for processing callbacks
- Proper cleanup in destructors
- Parameter declarations and usage

### Message Types
- Use of standard message types from `std_msgs`, `geometry_msgs`, `sensor_msgs`
- Custom message definitions when needed
- Proper serialization and deserialization

### TF2 Usage
- Broadcasting transforms with proper timestamps
- Listening to transforms with error handling
- Transforming points and poses between frames
- Using TF2 for robot kinematics

### Actions
- Proper action server implementation with feedback
- Action client with goal handling
- Canceling and aborting goals appropriately
- Using action results for completion status

### Launch Files
- Parameter declarations and configurations
- Node grouping and namespaces
- Conditional launching
- Passing arguments to nodes

## Common Patterns

### Publisher Pattern
```python
# Create publisher in constructor
self.publisher = self.create_publisher(MessageType, 'topic_name', 10)

# Publish messages in callbacks or timers
msg = MessageType()
msg.data = "example"
self.publisher.publish(msg)
```

### Subscriber Pattern
```python
# Create subscriber in constructor
self.subscription = self.create_subscription(
    MessageType,
    'topic_name',
    self.callback_function,
    10
)

# Define callback function
def callback_function(self, msg):
    # Process received message
    self.get_logger().info(f'Received: {msg.data}')
```

### Service Pattern
```python
# Create service server in constructor
self.srv = self.create_service(RequestType, 'service_name', self.service_callback)

# Define service callback
def service_callback(self, request, response):
    # Process request and populate response
    response.result = request.a + request.b
    return response
```

### Action Pattern
```python
# Create action server in constructor
self._action_server = ActionServer(
    self,
    ActionType,
    'action_name',
    self.execute_callback
)

# Define execute callback
def execute_callback(self, goal_handle):
    # Execute the action with feedback
    feedback_msg = ActionType.Feedback()
    result_msg = ActionType.Result()

    for i in range(goal_handle.request.order):
        # Publish feedback
        feedback_msg.sequence.append(i)
        goal_handle.publish_feedback(feedback_msg)

        # Check for cancellation
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result_msg.sequence = feedback_msg.sequence
            return result_msg

    # Complete successfully
    goal_handle.succeed()
    result_msg.sequence = feedback_msg.sequence
    return result_msg
```

## Simulation Integration

### Gazebo Integration
The examples demonstrate how to integrate with Gazebo simulation:
- Spawning robots in simulation
- Controlling simulated robots
- Working with simulated sensors
- Using Gazebo plugins for ROS2 control

### RViz Visualization
Examples include RViz configuration files showing:
- Robot model visualization
- Sensor data visualization
- TF frame visualization
- Path and trajectory visualization

## Troubleshooting

### Common Issues
1. **Node not found**: Make sure to source the workspace after building
2. **Topic not found**: Verify topic names and namespaces match
3. **TF transform not found**: Check that transforms are being broadcasted
4. **Action server not responding**: Verify action name matches server

### Debugging Commands
```bash
# List available topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /topic_name

# List available services
ros2 service list

# Call a service
ros2 service call /service_name service_type "{request_field: value}"

# List available actions
ros2 action list

# Get action information
ros2 action info /action_name
```

## Extending Examples

To extend these examples for your own humanoid robot:

1. **Modify URDF**: Update the robot model in `simple_humanoid.urdf` with your robot's specifications
2. **Update Control Config**: Modify `control_config.yaml` with your robot's joint names and parameters
3. **Add Custom Messages**: Create custom message types if needed
4. **Extend Functionality**: Build upon the basic examples to implement your specific robot behaviors

## Reference Materials

- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [TF2 Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/index.html)
- [ROS2 Actions](https://docs.ros.org/en/humble/Tutorials/Intermediate/Creating-an-Action.html)
- [ROS2 Launch](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)
- [Robotics Stack Exchange](https://robotics.stackexchange.com/) for community support

## Contributing

If you have improvements to these examples:
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request with a clear description of your changes

## License

These examples are licensed under the Apache 2.0 license. See the LICENSE file for details.