---
title: ROS 2 Parameters and Lifecycle Management
description: Configuration management and node lifecycle control in ROS 2
sidebar_position: 4
learning_outcomes:
  - Understand the ROS 2 parameter system for configuration management
  - Implement parameter declaration and usage in nodes
  - Manage node lifecycle states and transitions
  - Apply best practices for configuration and lifecycle management
---

# ROS 2 Parameters and Lifecycle Management: Configuration and State Control

## Purpose
This chapter covers two critical aspects of ROS 2 systems: parameter management for configuration and lifecycle management for controlling node states. You'll learn how to make your nodes configurable and how to manage their startup, operation, and shutdown phases systematically.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Understand the ROS 2 parameter system for configuration management
- Implement parameter declaration and usage in nodes
- Manage node lifecycle states and transitions
- Apply best practices for configuration and lifecycle management

## Parameters in ROS 2

### Parameter System Overview
Parameters in ROS 2 provide a way to configure nodes at runtime. They allow you to:
- Modify node behavior without recompiling
- Adapt to different environments or hardware configurations
- Separate configuration from code logic
- Enable dynamic reconfiguration during operation

Parameters can be:
- Declared at runtime or compile time
- Changed dynamically during node operation
- Loaded from configuration files
- Shared between nodes when needed

<!-- RAG_CHUNK_ID: ros2-parameter-system-overview -->

### Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_threshold', 0.5)
        self.declare_parameter('debug_mode', False)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.debug_mode = self.get_parameter('debug_mode').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')

        # Set up a parameter callback for dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.max_velocity = param.value
                self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
            elif param.name == 'debug_mode' and param.type_ == Parameter.Type.BOOL:
                self.debug_mode = param.value
                self.get_logger().info(f'Debug mode: {self.debug_mode}')

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    from rclpy.parameter import Parameter
    from rclpy.parameter_service import SetParametersResult
    main()
```

**Instructions:**
1. Create the file in your package: `my_robot_pkg/my_robot_pkg/parameter_node.py`
2. Make it executable: `chmod +x my_robot_pkg/my_robot_pkg/parameter_node.py`
3. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
4. Source the setup file: `source install/setup.bash`
5. Run the node: `ros2 run my_robot_pkg parameter_node`

<!-- RAG_CHUNK_ID: ros2-parameter-declaration-implementation -->

### Parameter Types and Validation

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import SetParameters

class ValidatedParameterNode(Node):
    def __init__(self):
        super().__init__('validated_parameter_node')

        # Declare parameters with specific types and constraints
        self.declare_parameter(
            'wheel_diameter',
            0.1,  # Default value
            ParameterDescriptor(
                name='wheel_diameter',
                type=ParameterType.PARAMETER_DOUBLE,
                description='Diameter of the robot wheels in meters',
                additional_constraints='Must be positive',
                floating_point_range=[rclpy.Parameter.FloatingPointRange(from_value=0.01, to_value=1.0)]
            )
        )

        self.declare_parameter(
            'motor_ids',
            [1, 2, 3, 4],  # Default value
            ParameterDescriptor(
                name='motor_ids',
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description='IDs of the motors in the robot'
            )
        )

        # Get validated parameters
        self.wheel_diameter = self.get_parameter('wheel_diameter').value
        self.motor_ids = self.get_parameter('motor_ids').value

        self.get_logger().info(f'Wheel diameter: {self.wheel_diameter}')
        self.get_logger().info(f'Motor IDs: {self.motor_ids}')

def main(args=None):
    rclpy.init(args=args)
    node = ValidatedParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-parameter-validation-implementation -->

### Loading Parameters from Files

You can load parameters from YAML files, which is useful for different robot configurations:

Example YAML file (`config/robot_params.yaml`):
```yaml
parameter_node:
  ros__parameters:
    robot_name: 'husky_robot'
    max_velocity: 2.0
    safety_threshold: 0.8
    debug_mode: false
    wheel_diameter: 0.33
    motor_ids: [1, 2, 3, 4]
```

To load parameters from a file:
```bash
ros2 run my_robot_pkg parameter_node --ros-args --params-file config/robot_params.yaml
```

<!-- RAG_CHUNK_ID: ros2-parameter-file-loading -->

## Node Lifecycle Management

### Understanding Lifecycle Nodes
Lifecycle nodes provide a standardized way to manage the state of nodes through well-defined states and transitions. This is particularly useful for:
- Complex nodes that need initialization
- Nodes that need to be started/stopped in a specific order
- Systems requiring graceful startup/shutdown
- Resource management and coordination

The standard lifecycle states are:
- Unconfigured
- Inactive
- Active
- Finalized

<!-- RAG_CHUNK_ID: ros2-lifecycle-node-concept -->

### Creating a Lifecycle Node

```python
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.logging import get_logger

class LifecycleManagedNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_managed_node')
        self.get_logger().info('Lifecycle node initialized in unconfigured state')

    def on_configure(self, state):
        """Called when transitioning to configuring state"""
        self.get_logger().info('Configuring node...')

        # Initialize resources
        self.some_resource = "initialized"

        # Return success to allow transition to inactive state
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Called when transitioning to activating state"""
        self.get_logger().info('Activating node...')

        # Activate any necessary components
        self.get_logger().info(f'Resource value: {self.some_resource}')

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Called when transitioning to deactivating state"""
        self.get_logger().info('Deactivating node...')

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Called when transitioning to cleaning up state"""
        self.get_logger().info('Cleaning up node...')

        # Clean up resources
        self.some_resource = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """Called when transitioning to shutting down state"""
        self.get_logger().info('Shutting down node...')

        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        """Called when transitioning to error processing state"""
        self.get_logger().info('Error processing...')

        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleManagedNode()

    try:
        # The node will be spun like a regular node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Instructions:**
1. Create the file in your package: `my_robot_pkg/my_robot_pkg/lifecycle_node.py`
2. Make it executable: `chmod +x my_robot_pkg/my_robot_pkg/lifecycle_node.py`
3. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
4. Source the setup file: `source install/setup.bash`
5. Run the node: `ros2 run my_robot_pkg lifecycle_node`

To manage the lifecycle, use the lifecycle tools:
```bash
# Get current state
ros2 lifecycle get /lifecycle_managed_node

# Change state to configure
ros2 lifecycle configure /lifecycle_managed_node

# Change state to activate
ros2 lifecycle activate /lifecycle_managed_node

# Change state to deactivate
ros2 lifecycle deactivate /lifecycle_managed_node

# Change state to cleanup
ros2 lifecycle cleanup /lifecycle_managed_node

# Shutdown the node
ros2 lifecycle shutdown /lifecycle_managed_node
```

<!-- RAG_CHUNK_ID: ros2-lifecycle-node-implementation -->

### Lifecycle Node with Parameters

```python
from lifecycle_msgs.msg import Transition
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

class ConfigurableLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('configurable_lifecycle_node')

        # Declare parameters that can be used during lifecycle transitions
        self.declare_parameter('startup_delay', 2.0)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('enable_diagnostics', True)

    def on_configure(self, state):
        self.get_logger().info('Configuring node...')

        # Use parameters during configuration
        self.startup_delay = self.get_parameter('startup_delay').value
        self.max_retries = self.get_parameter('max_retries').value
        self.enable_diagnostics = self.get_parameter('enable_diagnostics').value

        # Initialize components that require configuration
        self.get_logger().info(f'Configured with startup delay: {self.startup_delay}s')

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating node...')

        # Start active operations
        if self.enable_diagnostics:
            self.get_logger().info('Diagnostics enabled')

        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = ConfigurableLifecycleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-lifecycle-with-parameters -->

## Best Practices

### Parameter Best Practices
1. **Use meaningful names**: Use descriptive parameter names with consistent naming conventions
2. **Provide defaults**: Always provide sensible default values
3. **Validate inputs**: Use parameter descriptors to validate ranges and types
4. **Document parameters**: Include clear descriptions for each parameter
5. **Group related parameters**: Organize related parameters logically

### Lifecycle Best Practices
1. **Use when appropriate**: Only use lifecycle nodes when you need the state management
2. **Implement proper cleanup**: Ensure resources are properly released in cleanup
3. **Handle errors gracefully**: Implement error handling in lifecycle callbacks
4. **Maintain consistency**: Keep lifecycle transitions consistent across similar nodes
5. **Use with composition**: Combine with node composition for complex systems

<!-- RAG_CHUNK_ID: ros2-best-practices-parameters-lifecycle -->

## Hands-on Exercise
Create a robot configuration system that combines parameters and lifecycle management for sensor calibration:

### Part 1: Package Setup
1. Create a new package: `ros2 pkg create --build-type ament_python sensor_calibration_pkg`
2. Navigate to the package: `cd ~/ros2_ws/src/sensor_calibration_pkg`
3. Update `package.xml` to include lifecycle dependencies:
```xml
<depend>rclpy</depend>
<depend>lifecycle_msgs</depend>
```

### Part 2: Lifecycle Node with Parameters Implementation
Create a lifecycle node that manages sensor calibration:
```python
# calibration_node.py
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64
import time

class CalibrationLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('calibration_lifecycle_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('calibration_tolerance', 0.01,
                              rclpy.ParameterDescriptor(description='Tolerance for calibration'))
        self.declare_parameter('num_calibration_samples', 10,
                              rclpy.ParameterDescriptor(description='Number of samples for calibration'))
        self.declare_parameter('calibration_frequency', 1.0,
                              rclpy.ParameterDescriptor(description='Calibration frequency in Hz'))

        # Initialize parameter values
        self.tolerance = self.get_parameter('calibration_tolerance').value
        self.num_samples = self.get_parameter('num_calibration_samples').value
        self.frequency = self.get_parameter('calibration_frequency').value

        # Initialize calibration state
        self.calibration_active = False
        self.current_sample = 0
        self.calibration_data = []

        # Add parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Calibration Lifecycle Node initialized')

    def parameter_callback(self, params):
        """Validate and update parameters dynamically."""
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == 'calibration_tolerance' and param.value <= 0:
                self.get_logger().error('Calibration tolerance must be positive')
                return SetParametersResult(successful=False)
            elif param.name == 'num_calibration_samples' and param.value <= 0:
                self.get_logger().error('Number of samples must be positive')
                return SetParametersResult(successful=False)
            elif param.name == 'calibration_frequency' and param.value <= 0:
                self.get_logger().error('Frequency must be positive')
                return SetParametersResult(successful=False)

        # Update values if validation passes
        for param in params:
            if param.name == 'calibration_tolerance':
                self.tolerance = param.value
                self.get_logger().info(f'Updated tolerance to {self.tolerance}')
            elif param.name == 'num_calibration_samples':
                self.num_samples = param.value
                self.get_logger().info(f'Updated number of samples to {self.num_samples}')
            elif param.name == 'calibration_frequency':
                self.frequency = param.value
                self.get_logger().info(f'Updated frequency to {self.frequency}')

        return SetParametersResult(successful=True)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when transitioning to CONFIGURING state."""
        self.get_logger().info(f'Configuring node, tolerance: {self.tolerance}, samples: {self.num_samples}')

        # Initialize publishers and subscribers
        self.sensor_pub = self.create_publisher(Float64, 'sensor_data', 10)
        self.calibration_pub = self.create_publisher(Float64, 'calibration_result', 10)

        # Initialize calibration data
        self.calibration_data = []
        self.current_sample = 0

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when transitioning to ACTIVATING state."""
        self.get_logger().info('Activating calibration process')

        # Activate publishers
        self.sensor_pub.on_activate()
        self.calibration_pub.on_activate()

        # Start calibration timer
        self.calibration_timer = self.create_timer(
            1.0 / self.frequency,
            self.calibration_callback
        )

        self.calibration_active = True
        self.current_sample = 0
        self.calibration_data = []

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when transitioning to DEACTIVATING state."""
        self.get_logger().info('Deactivating calibration process')

        # Deactivate publishers
        self.sensor_pub.on_deactivate()
        self.calibration_pub.on_deactivate()

        # Stop timer
        self.destroy_timer(self.calibration_timer)
        self.calibration_active = False

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when transitioning to CLEANINGUP state."""
        self.get_logger().info('Cleaning up calibration node')

        # Reset all data
        self.calibration_data = []
        self.current_sample = 0
        self.calibration_active = False

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when transitioning to SHUTTINGDOWN state."""
        self.get_logger().info('Shutting down calibration node')
        return TransitionCallbackReturn.SUCCESS

    def calibration_callback(self):
        """Main calibration logic."""
        if not self.calibration_active:
            return

        # Simulate sensor data
        import random
        sensor_value = 10.0 + random.uniform(-0.5, 0.5)  # Simulated sensor reading
        sensor_msg = Float64()
        sensor_msg.data = sensor_value
        self.sensor_pub.publish(sensor_msg)

        # Collect calibration data
        self.calibration_data.append(sensor_value)
        self.current_sample += 1

        self.get_logger().info(f'Calibration sample {self.current_sample}/{self.num_samples}, value: {sensor_value:.3f}')

        # Check if calibration is complete
        if self.current_sample >= self.num_samples:
            # Calculate calibration result
            avg_value = sum(self.calibration_data) / len(self.calibration_data)
            calibration_msg = Float64()
            calibration_msg.data = avg_value
            self.calibration_pub.publish(calibration_msg)

            self.get_logger().info(f'Calibration complete! Average value: {avg_value:.3f}')

            # Reset for next calibration cycle
            self.current_sample = 0
            self.calibration_data = []


def main(args=None):
    rclpy.init(args=args)

    node = CalibrationLifecycleNode()

    # Use the lifecycle node container to manage state transitions
    rclpy.spin(node.get_node())

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Lifecycle Client for State Management
Create a client to manage the lifecycle node:
```python
# lifecycle_client.py
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition
from rclpy.qos import QoSProfile

class LifecycleClient(Node):
    def __init__(self):
        super().__init__('lifecycle_client')

        # Create clients for lifecycle management
        self.get_state_client = self.create_client(
            GetState,
            'calibration_lifecycle_node/get_state'
        )
        self.change_state_client = self.create_client(
            ChangeState,
            'calibration_lifecycle_node/change_state'
        )

        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('State service not available, waiting again...')

        while not self.change_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Change state service not available, waiting again...')

    def get_state(self):
        request = GetState.Request()
        future = self.get_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().current_state.label

    def change_state(self, transition_id):
        request = ChangeState.Request()
        request.transition.id = transition_id
        future = self.change_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

def main(args=None):
    rclpy.init(args=args)
    client = LifecycleClient()

    # Test lifecycle transitions
    print(f'Initial state: {client.get_state()}')

    # Configure the node
    print('Configuring node...')
    client.change_state(Transition.TRANSITION_CONFIGURE)
    print(f'State after configure: {client.get_state()}')

    # Activate the node
    print('Activating node...')
    client.change_state(Transition.TRANSITION_ACTIVATE)
    print(f'State after activate: {client.get_state()}')

    # Wait for some calibration to happen
    import time
    time.sleep(10)  # Wait 10 seconds for calibration

    # Deactivate the node
    print('Deactivating node...')
    client.change_state(Transition.TRANSITION_DEACTIVATE)
    print(f'State after deactivate: {client.get_state()}')

    # Cleanup the node
    print('Cleaning up node...')
    client.change_state(Transition.TRANSITION_CLEANUP)
    print(f'State after cleanup: {client.get_state()}')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Testing
1. Build your workspace: `cd ~/ros2_ws && colcon build --packages-select sensor_calibration_pkg`
2. Source the workspace: `source install/setup.bash`
3. Run the lifecycle node in one terminal: `ros2 run sensor_calibration_pkg calibration_node`
4. Run the client in another terminal after a few seconds: `ros2 run sensor_calibration_pkg lifecycle_client`
5. Change parameters dynamically: `ros2 param set /calibration_lifecycle_node calibration_tolerance 0.05`
6. Observe the state transitions and parameter changes

### Expected Results
- The node transitions through different lifecycle states properly
- Parameters can be changed dynamically during operation
- Calibration runs when the node is active
- Parameter validation prevents invalid values
- This demonstrates the integration of parameters and lifecycle management

<!-- RAG_CHUNK_ID: ros2-hands-on-exercise-parameters-lifecycle -->

## Summary
Parameters and lifecycle management are essential tools for creating robust and configurable ROS 2 systems. Parameters allow you to separate configuration from code, making nodes adaptable to different environments. Lifecycle management provides a standardized approach to node state transitions, enabling complex systems with proper initialization, activation, and cleanup procedures.

## Further Reading
- [ROS 2 Parameters Documentation](https://docs.ros.org/en/humble/How-To-Guides/Using-Parameters-In-A-Class-Python.html)
- [ROS 2 Lifecycle Nodes Tutorial](https://docs.ros.org/en/humble/Tutorials/Managed-Nodes.html)
- [Parameter Best Practices](https://docs.ros.org/en/humble/How-To-Guides/Parameters-YAML-files.html)

## Summary
Parameters and lifecycle management are essential tools for creating robust and configurable ROS 2 systems. Parameters allow you to separate configuration from code, making nodes adaptable to different environments. Lifecycle management provides a standardized approach to node state transitions, enabling complex systems with proper initialization, activation, and cleanup procedures.

## Practice Questions
1. What is the difference between ROS 2 parameters and regular variables?
   - *Answer: Parameters are configurable values that can be changed at runtime without recompiling, while regular variables are fixed at compile time. Parameters can be loaded from files, set via command line, or changed dynamically during operation.*

2. When would you use a lifecycle node instead of a regular node?
   - *Answer: Use lifecycle nodes for complex systems where you need explicit control over initialization, configuration, activation, and shutdown. They're useful when nodes have dependencies on other nodes or when you need to coordinate startup/shutdown sequences.*

3. How do you validate parameters in ROS 2?
   - *Answer: You can use parameter callbacks with `add_on_set_parameters_callback` to validate parameter values before they're set. You can check value ranges, types, or other constraints and reject invalid values.*

## Check Your Understanding
**Multiple Choice Questions:**

1. What is the purpose of ROS 2 parameters?
   A) To store temporary data during execution
   B) To configure nodes at runtime without recompiling  *(Correct)*
   C) To manage node lifecycle states
   D) To handle message passing between nodes

   *Explanation: Parameters provide a way to configure nodes at runtime without needing to recompile the code, allowing for environment-specific configurations.*

2. Which lifecycle state is entered after successful configuration?
   A) Unconfigured
   B) Active
   C) Inactive  *(Correct)*
   D) Finalized

   *Explanation: After a lifecycle node is configured successfully, it transitions to the Inactive state. It can then be activated to move to the Active state.*

3. What happens when you call `declare_parameter` in ROS 2?
   A) The parameter is immediately set to a value
   B) The parameter is registered with default value and metadata  *(Correct)*
   C) The node is reconfigured
   D) A service is created for parameter management

   *Explanation: The `declare_parameter` function registers a parameter with the node, setting up its default value and metadata but doesn't immediately set a value.*

<!-- RAG_CHUNK_ID: ros2-parameters-lifecycle-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->