---
title: ROS 2 Services and Actions
description: Understanding request-response communication and long-running tasks in ROS 2
sidebar_position: 3
learning_outcomes:
  - Understand the service-server communication pattern in ROS 2
  - Implement custom services for request-response interactions
  - Distinguish between services and actions for different use cases
  - Implement actions for long-running tasks with feedback
---

# ROS 2 Services and Actions: Request-Response Communication and Long-Running Tasks

## Purpose
This chapter explores two important communication patterns in ROS 2: services for request-response interactions and actions for long-running tasks with feedback. You'll learn when to use each pattern and how to implement them effectively in your robotic applications.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Understand the service-server communication pattern in ROS 2
- Implement custom services for request-response interactions
- Distinguish between services and actions for different use cases
- Implement actions for long-running tasks with feedback

## Services in ROS 2

### Service-Server Pattern
Services provide a request-response communication pattern in ROS 2. This is synchronous communication where:
- A client sends a request to a server
- The server processes the request and sends back a response
- The client waits for the response before continuing

This pattern is suitable for operations that:
- Have a clear request and response
- Complete relatively quickly
- Don't require ongoing feedback during execution

<!-- RAG_CHUNK_ID: ros2-service-concept-definition -->

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        # Create a service that listens on 'add_two_ints' topic
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        # Process the request and set the response
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a} b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Instructions:**
1. Create the file in your package: `my_robot_pkg/my_robot_pkg/service_server.py`
2. Make it executable: `chmod +x my_robot_pkg/my_robot_pkg/service_server.py`
3. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
4. Source the setup file: `source install/setup.bash`
5. Run the service server: `ros2 run my_robot_pkg service_server`

<!-- RAG_CHUNK_ID: ros2-service-server-implementation -->

### Creating a Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        # Create a client for the 'add_two_ints' service
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        # Call the service asynchronously
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()

    # Send a request to the service
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')

    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Instructions:**
1. Create the file in your package: `my_robot_pkg/my_robot_pkg/service_client.py`
2. Make it executable: `chmod +x my_robot_pkg/my_robot_pkg/service_client.py`
3. Build your package: `cd ~/ros2_ws && colcon build --packages-select my_robot_pkg`
4. Source the setup file: `source install/setup.bash`
5. Run the client (after starting the server): `ros2 run my_robot_pkg service_client`

<!-- RAG_CHUNK_ID: ros2-service-client-implementation -->

## Custom Services

### Creating Custom Service Definitions
Services require a definition file with the `.srv` extension that specifies the request and response format.

Example custom service (`srv/CalculateDistance.srv`):
```
# Request
float64 x1
float64 y1
float64 x2
float64 y2
---
# Response
float64 distance
bool success
string message
```

### Using Custom Services

```python
# Service server with custom service
from my_robot_pkg.srv import CalculateDistance  # Custom service
import rclpy
from rclpy.node import Node
import math

class DistanceService(Node):
    def __init__(self):
        super().__init__('distance_service')
        self.srv = self.create_service(
            CalculateDistance,
            'calculate_distance',
            self.calculate_distance_callback
        )

    def calculate_distance_callback(self, request, response):
        # Calculate Euclidean distance
        distance = math.sqrt(
            (request.x2 - request.x1)**2 +
            (request.y2 - request.y1)**2
        )

        response.distance = distance
        response.success = True
        response.message = f'Distance calculated successfully: {distance:.2f}'

        self.get_logger().info(
            f'Calculated distance between ({request.x1}, {request.y1}) and '
            f'({request.x2}, {request.y2}): {distance:.2f}'
        )

        return response

def main(args=None):
    rclpy.init(args=args)
    node = DistanceService()

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

<!-- RAG_CHUNK_ID: ros2-custom-service-implementation -->

## Actions in ROS 2

### When to Use Actions
Actions are designed for long-running tasks that require:
- Feedback during execution
- The ability to cancel the task
- Goal status reporting
- Result reporting upon completion

Examples of appropriate use cases:
- Robot navigation to a goal location
- Object manipulation tasks
- Calibration procedures
- Data collection over time

### Action Structure
An action in ROS 2 consists of three message types:
- **Goal**: Defines what the action should do
- **Feedback**: Provides ongoing status during execution
- **Result**: Contains the final outcome of the action

<!-- RAG_CHUNK_ID: ros2-action-concept-definition -->

### Creating an Action Server

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from my_robot_pkg.action import Fibonacci  # Custom action
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        # Accept or reject a goal
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept or reject a cancel request
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Create feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Initialize the sequence
        feedback_msg.sequence = [0, 1]

        # Execute the action with feedback
        for i in range(1, goal_handle.request.order):
            # Check if the goal was canceled
            if goal_handle.is_cancel_requested:
                result_msg.sequence = feedback_msg.sequence
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return result_msg

            # Update the sequence
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

            # Sleep to simulate work
            time.sleep(1)

        # Set the result
        goal_handle.succeed()
        result_msg.sequence = feedback_msg.sequence

        self.get_logger().info(f'Result: {result_msg.sequence}')
        return result_msg

def main(args=None):
    rclpy.init(args=args)
    node = FibonacciActionServer()

    try:
        executor = MultiThreadedExecutor()
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import time
    main()
```

<!-- RAG_CHUNK_ID: ros2-action-server-implementation -->

### Creating an Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_pkg.action import Fibonacci  # Custom action

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        # Create an action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

    def send_goal(self, order):
        # Wait for the action server to be available
        self._action_client.wait_for_server()

        # Create a goal message
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Send the goal and get a future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        # Set callbacks for when the goal is accepted
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get the result future
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()

    # Send a goal
    action_client.send_goal(10)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

<!-- RAG_CHUNK_ID: ros2-action-client-implementation -->

## Services vs Actions: When to Use Which

| Feature | Services | Actions |
|---------|----------|---------|
| Communication Type | Request-Response (sync) | Goal-Feedback-Result (async) |
| Execution Time | Short | Long-running |
| Feedback | No | Yes |
| Cancellation | No | Yes |
| Status Updates | No | Yes |
| Use Case | Quick calculations, queries | Navigation, manipulation, calibration |

### Decision Matrix
- Use **Services** when:
  - Task completes quickly (< 1 second)
  - No feedback needed during execution
  - Simple request-response pattern suffices
  - Cancellation is not required

- Use **Actions** when:
  - Task takes a long time to complete
  - Feedback during execution is valuable
  - Ability to cancel the task is needed
  - Status updates are important

<!-- RAG_CHUNK_ID: ros2-services-vs-actions-comparison -->

## Hands-on Exercise
Create a robot arm control system using both services and actions to understand their different use cases:

### Part 1: Package Setup and Custom Messages/Actions
1. Create a new package: `ros2 pkg create --build-type ament_python robot_arm_control`
2. Navigate to the package: `cd ~/ros2_ws/src/robot_arm_control`
3. Create directories for custom interfaces: `mkdir srv action`
4. Create a service definition `srv/GetPosition.srv`:
```
# Request: no parameters needed
---
# Response: return current joint angles
float64[] joint_angles
string arm_status
```
5. Create an action definition `action/MoveArm.action`:
```
# Goal: target joint angles
float64[] target_angles
---
# Result: final position and success status
bool success
float64[] final_angles
string message
---
# Feedback: current progress
float64[] current_angles
float64 progress_percentage
string status
```

### Part 2: Service Server Implementation
Create a service server to get current arm position:
```python
# position_service.py
import rclpy
from rclpy.node import Node
from robot_arm_control.srv import GetPosition  # Update with your package name

class PositionService(Node):
    def __init__(self):
        super().__init__('position_service')
        self.srv = self.create_service(
            GetPosition,
            'get_current_position',
            self.get_position_callback
        )
        # Simulate initial joint angles
        self.current_joint_angles = [0.0, 0.0, 0.0, 0.0]  # 4 joints
        self.arm_status = "IDLE"

    def get_position_callback(self, request, response):
        response.joint_angles = self.current_joint_angles
        response.arm_status = self.arm_status
        self.get_logger().info(f'Returning current position: {self.current_joint_angles}')
        return response

def main(args=None):
    rclpy.init(args=args)
    position_service = PositionService()
    rclpy.spin(position_service)
    position_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Action Server Implementation
Create an action server for moving the arm:
```python
# move_arm_server.py
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from robot_arm_control.action import MoveArm  # Update with your package name
import time
import math

class MoveArmServer(Node):
    def __init__(self):
        super().__init__('move_arm_server')
        self._action_server = ActionServer(
            self,
            MoveArm,
            'move_arm',
            self.execute_callback,
            cancel_callback=self.cancel_callback
        )
        # Simulate current joint angles
        self.current_joint_angles = [0.0, 0.0, 0.0, 0.0]
        self.is_cancel_requested = False

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        self.is_cancel_requested = True
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing move arm action...')

        target_angles = goal_handle.request.target_angles
        self.get_logger().info(f'Moving to target: {target_angles}')

        # Simulate movement with feedback
        start_angles = self.current_joint_angles[:]
        steps = 50  # Number of feedback steps

        for i in range(steps):
            if self.is_cancel_requested:
                goal_handle.canceled()
                result = MoveArm.Result()
                result.success = False
                result.message = "Movement canceled"
                result.final_angles = self.current_joint_angles
                self.is_cancel_requested = False
                return result

            # Calculate intermediate position
            progress = i / (steps - 1)
            for j in range(len(self.current_joint_angles)):
                self.current_joint_angles[j] = start_angles[j] + \
                    (target_angles[j] - start_angles[j]) * progress

            # Send feedback
            feedback_msg = MoveArm.Feedback()
            feedback_msg.current_angles = self.current_joint_angles
            feedback_msg.progress_percentage = progress * 100.0
            feedback_msg.status = f'Moving... {progress * 100.0:.1f}% complete'
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(0.1)  # Simulate time delay

        # Update final position
        self.current_joint_angles = target_angles[:]

        # Complete the goal
        goal_handle.succeed()
        result = MoveArm.Result()
        result.success = True
        result.message = "Movement completed successfully"
        result.final_angles = self.current_joint_angles

        self.get_logger().info(f'Movement completed. Final angles: {self.current_joint_angles}')
        return result

def main(args=None):
    rclpy.init(args=args)
    move_arm_server = MoveArmServer()
    rclpy.spin(move_arm_server)
    move_arm_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Client Implementation
Create a client that uses both service and action:
```python
# arm_control_client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_arm_control.srv import GetPosition  # Update with your package name
from robot_arm_control.action import MoveArm  # Update with your package name
import time

class ArmControlClient(Node):
    def __init__(self):
        super().__init__('arm_control_client')

        # Create service client
        self.cli = self.create_client(GetPosition, 'get_current_position')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create action client
        self._action_client = ActionClient(
            self,
            MoveArm,
            'move_arm'
        )

    def get_current_position(self):
        request = GetPosition.Request()
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def move_arm_to_position(self, target_angles):
        goal_msg = MoveArm.Goal()
        goal_msg.target_angles = target_angles

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return None

        self.get_logger().info('Goal accepted, waiting for result...')
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        return get_result_future.result().result

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(
            f'Feedback: {feedback_msg.feedback.status}'
        )

def main(args=None):
    rclpy.init(args=args)
    arm_client = ArmControlClient()

    # Get current position
    arm_client.get_logger().info("Getting current arm position...")
    response = arm_client.get_current_position()
    if response:
        arm_client.get_logger().info(f'Current position: {response.joint_angles}')

    # Move arm to new position
    target = [1.57, 0.78, -0.78, 0.0]  # Example target angles
    arm_client.get_logger().info(f"Moving arm to position: {target}")
    result = arm_client.move_arm_to_position(target)
    if result:
        arm_client.get_logger().info(f'Movement result: {result.message}')

    arm_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 5: Testing
1. Build your workspace: `cd ~/ros2_ws && colcon build --packages-select robot_arm_control`
2. Source the workspace: `source install/setup.bash`
3. Run the service server in one terminal: `ros2 run robot_arm_control position_service`
4. Run the action server in another terminal: `ros2 run robot_arm_control move_arm_server`
5. Run the client in a third terminal: `ros2 run robot_arm_control arm_control_client`
6. Observe the service requests and action feedback

### Expected Results
- The service should quickly return current arm position
- The action should provide feedback during the movement process
- You can cancel the action during execution
- This demonstrates the different use cases for services vs actions

<!-- RAG_CHUNK_ID: ros2-hands-on-exercise-services-actions -->

## Summary
Services and actions provide different communication patterns for different needs in ROS 2. Services are ideal for quick, synchronous request-response interactions, while actions are designed for long-running tasks that require feedback, status updates, and cancellation capabilities. Understanding when to use each pattern is crucial for effective robotic system design.

## Further Reading
- [ROS 2 Services Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html)
- [ROS 2 Actions Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Actions-In-Python.html)
- [Services vs Actions Comparison](https://docs.ros.org/en/humble/Concepts/About-Actions.html)

## Summary
Services and actions provide different communication patterns for different needs in ROS 2. Services are ideal for quick, synchronous request-response interactions, while actions are designed for long-running tasks that require feedback, status updates, and cancellation capabilities. Understanding when to use each pattern is crucial for effective robotic system design.

## Practice Questions
1. What is the main difference between services and actions in ROS 2?
   - *Answer: Services provide synchronous request-response communication for short tasks, while actions are designed for long-running tasks with feedback, status updates, and cancellation capabilities.*

2. When would you use an action instead of a service?
   - *Answer: Use actions for long-running tasks that need to provide feedback during execution, can be cancelled, or have complex status reporting. Use services for quick, simple request-response operations.*

3. What are the three components of an action in ROS 2?
   - *Answer: The three components are: Goal (request sent to the action server), Result (final outcome sent back to the client), and Feedback (intermediate status updates during execution).*

## Check Your Understanding
**Multiple Choice Questions:**

1. What type of communication does a ROS 2 service provide?
   A) Publish-subscribe
   B) Request-response  *(Correct)*
   C) Broadcast
   D) Peer-to-peer

   *Explanation: Services provide synchronous request-response communication where a client sends a request and waits for a response from the server.*

2. Which of the following is a characteristic of ROS 2 actions?
   A) Synchronous communication only
   B) No feedback during execution
   C) Ability to cancel long-running tasks  *(Correct)*
   D) Short execution time only

   *Explanation: Actions are designed for long-running tasks and include features like feedback during execution and the ability to cancel ongoing tasks.*

3. In the service-server pattern, what happens when a client sends a request?
   A) The client continues execution immediately
   B) The client waits for a response from the server  *(Correct)*
   C) The server sends feedback during processing
   D) The communication is asynchronous

   *Explanation: In the service-server pattern, communication is synchronous - the client sends a request and waits for the response before continuing.*

<!-- RAG_CHUNK_ID: ros2-services-actions-chapter-summary -->
<!-- URDU_TODO: Translate this chapter to Urdu -->