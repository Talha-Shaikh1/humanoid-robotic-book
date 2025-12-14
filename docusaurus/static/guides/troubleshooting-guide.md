# Troubleshooting and Debugging Guide: AI-Native Humanoid Robotics

## Purpose
This guide provides systematic approaches for troubleshooting and debugging humanoid robotics systems built with ROS2, Gazebo, Isaac Sim, and AI components. It covers common issues, diagnostic techniques, and resolution strategies for complex multimodal robotic systems.

<!-- URDU_TODO: Translate this section to Urdu -->

## Learning Outcomes
- Apply systematic troubleshooting methodologies to robotic systems
- Use diagnostic tools for ROS2, Gazebo, and Isaac Sim environments
- Debug multimodal AI-robot integration issues
- Implement error handling and recovery strategies
- Create effective debugging workflows for complex systems

## Troubleshooting Methodology

### Systematic Approach to Problem Solving
Effective troubleshooting follows a systematic approach:

1. **Reproduce the Issue**: Verify the problem occurs consistently
2. **Isolate the Component**: Identify which system component is failing
3. **Gather Information**: Collect logs, error messages, and system state
4. **Formulate Hypothesis**: Propose potential causes based on evidence
5. **Test Hypothesis**: Verify the proposed cause through testing
6. **Implement Solution**: Apply the fix to resolve the issue
7. **Verify Resolution**: Confirm the problem is solved
8. **Document Solution**: Record the issue and resolution for future reference

<!-- RAG_CHUNK_ID: troubleshooting-methodology -->

### Information Gathering Tools
Use these tools to collect diagnostic information:

```bash
# ROS2 diagnostic tools
ros2 topic list                    # List all active topics
ros2 topic info /topic_name       # Get information about a topic
ros2 topic echo /topic_name       # Monitor messages on a topic
ros2 service list                 # List all available services
ros2 service call /service_name   # Call a service
ros2 action list                  # List all available actions
ros2 node list                    # List all active nodes
ros2 node info /node_name         # Get information about a node
ros2 param list                   # List all parameters
ros2 param get /node_name param_name  # Get parameter value

# System monitoring
htop                            # Monitor CPU and memory usage
nvidia-smi                      # Monitor GPU usage (for Isaac Sim)
df -h                           # Check disk space usage
iotop                           # Monitor disk I/O

# Network diagnostics
ping hostname                   # Test network connectivity
netstat -tuln                   # List network connections
ss -tuln                        # Alternative to netstat
```

<!-- RAG_CHUNK_ID: diagnostic-tools-list -->

## Common ROS2 Issues and Solutions

### Node Communication Problems

#### Topic Connection Issues
**Symptoms**: Nodes not receiving messages from publishers
**Diagnosis**:
```bash
# Check if topic exists
ros2 topic list | grep topic_name

# Check topic information
ros2 topic info /topic_name

# Verify message types match
ros2 topic type /topic_name

# Monitor traffic on the topic
ros2 topic echo /topic_name
```

**Solutions**:
1. Verify topic names match exactly (including namespaces)
2. Check that message types are compatible between publisher and subscriber
3. Ensure QoS profiles are compatible
4. Check for typos in topic names
5. Verify nodes are on the same ROS_DOMAIN_ID

#### Service Call Failures
**Symptoms**: Service calls return errors or timeouts
**Diagnosis**:
```bash
# Check if service exists
ros2 service list | grep service_name

# Check service type
ros2 service type /service_name

# Test service call
ros2 service call /service_name service_type "{request_field: value}"
```

**Solutions**:
1. Verify service is running and accepting calls
2. Check service type matches client expectations
3. Ensure service and client are on same domain
4. Verify service has proper resource allocation

### Parameter Configuration Issues

#### Parameter Not Found
**Symptoms**: Nodes report missing required parameters
**Diagnosis**:
```bash
# List all parameters for a node
ros2 param list -n /node_name

# Get specific parameter value
ros2 param get /node_name param_name

# Check parameter descriptions
ros2 param describe /node_name param_name
```

**Solutions**:
1. Verify parameter name spelling and case sensitivity
2. Check if parameter is declared in the node
3. Ensure parameter file is loaded correctly
4. Verify YAML syntax in parameter files

#### Parameter Value Issues
**Symptoms**: Nodes behave unexpectedly despite parameter configuration
**Diagnosis**:
```bash
# Check current parameter value
ros2 param get /node_name param_name

# Compare with expected value
# Verify parameter type matches expectation
```

**Solutions**:
1. Check parameter type (integer vs float vs string)
2. Verify value is within expected range
3. Ensure parameter is not being overridden elsewhere
4. Validate YAML formatting for complex types

<!-- RAG_CHUNK_ID: ros2-common-issues -->

## Gazebo Simulation Troubleshooting

### Physics and Collision Issues

#### Robot Falling Through Ground
**Symptoms**: Robot falls through floor or other static objects
**Causes**:
- Missing or incorrect collision geometries
- Improper mass/inertia properties
- Physics engine instability
- Incorrect joint configurations

**Diagnosis**:
```bash
# Check URDF for proper collision elements
check_urdf /path/to/robot.urdf

# Verify mass and inertia values
# Look for zero masses or invalid inertia values
```

**Solutions**:
1. Ensure all links have proper collision geometries
2. Verify mass values are positive and realistic
3. Check inertia values are physically plausible
4. Adjust physics parameters in Gazebo world file

#### Joint Limits Not Working
**Symptoms**: Joints exceed defined limits in simulation
**Diagnosis**:
```xml
<!-- Check URDF joint definition -->
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <!-- Verify limits are defined -->
</joint>
```

**Solutions**:
1. Ensure joint limits are properly defined in URDF
2. Check Gazebo plugin configuration for joint limits
3. Verify physics engine parameters support joint limits
4. Consider using safety controllers for additional protection

### Sensor Simulation Problems

#### Camera Not Publishing Images
**Symptoms**: Camera topic exists but no images published
**Diagnosis**:
```bash
# Check if camera topic exists
ros2 topic list | grep camera

# Monitor camera info topic
ros2 topic echo /camera_name/camera_info

# Check Gazebo plugins
gz topic -l | grep camera
```

**Solutions**:
1. Verify camera plugin is properly configured in URDF
2. Check sensor parameters (resolution, rate, etc.)
3. Ensure GPU rendering is enabled if using GPU sensors
4. Verify Gazebo rendering engine compatibility

#### LIDAR Spikes and Noise
**Symptoms**: LIDAR data shows unrealistic distances or spikes
**Solutions**:
1. Adjust LIDAR sensor parameters (range, resolution)
2. Check for reflective surfaces causing multipath
3. Verify sensor mounting position and orientation
4. Consider environment lighting conditions

<!-- RAG_CHUNK_ID: gazebo-simulation-issues -->

## Isaac Sim Troubleshooting

### Environment Setup Issues

#### Isaac Sim Not Launching
**Symptoms**: Isaac Sim fails to start or crashes immediately
**Diagnosis**:
```bash
# Check system requirements
nvidia-smi  # Verify GPU compatibility
free -h     # Check available memory
df -h       # Check disk space

# Check Isaac Sim logs
# Usually located in ~/.nvidia-isaac/logs/
```

**Solutions**:
1. Verify NVIDIA GPU with sufficient VRAM (recommended 8GB+)
2. Ensure latest NVIDIA drivers are installed
3. Check CUDA compatibility (Isaac Sim requires specific CUDA version)
4. Verify sufficient system RAM and disk space

### AI Model Integration Issues

#### Model Not Loading
**Symptoms**: AI models fail to load or cause errors
**Diagnosis**:
```bash
# Check model file paths and permissions
ls -la /path/to/model

# Verify model format compatibility
# Check Isaac Sim documentation for supported formats
```

**Solutions**:
1. Verify model file paths are correct and accessible
2. Check model format compatibility with Isaac Sim
3. Ensure model dependencies are installed
4. Verify GPU memory is sufficient for model size

#### Model Performance Issues
**Symptoms**: AI models running slowly or causing frame drops
**Solutions**:
1. Optimize model for inference (quantization, pruning)
2. Adjust simulation update rates
3. Check GPU utilization and memory usage
4. Consider model simplification if possible

<!-- RAG_CHUNK_ID: isaac-sim-issues -->

## AI Integration Troubleshooting

### Vision-Language-Action (VLA) Issues

#### VLA Model Not Responding
**Symptoms**: VLA system doesn't generate actions for vision-language inputs
**Diagnosis**:
```python
# Check model loading
print(f"Model loaded: {model is not None}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Check input shapes
print(f"Vision input shape: {vision_input.shape}")
print(f"Language input shape: {language_input.shape}")
```

**Solutions**:
1. Verify model checkpoint files are complete and correct
2. Check input preprocessing matches training preprocessing
3. Ensure vision and language modalities are properly fused
4. Validate action space compatibility

#### Multimodal Fusion Problems
**Symptoms**: Vision and language inputs don't combine meaningfully
**Solutions**:
1. Verify cross-attention mechanisms are properly configured
2. Check feature dimension compatibility between modalities
3. Validate fusion layer training and initialization
4. Ensure proper normalization of different modalities

### Training and Inference Issues

#### GPU Memory Exhaustion
**Symptoms**: Training/inference fails due to out-of-memory errors
**Solutions**:
1. Reduce batch size
2. Use gradient accumulation instead of larger batches
3. Enable mixed precision training (fp16)
4. Clear GPU cache periodically: `torch.cuda.empty_cache()`

#### Model Divergence
**Symptoms**: Training loss increases or becomes NaN
**Solutions**:
1. Reduce learning rate
2. Implement gradient clipping
3. Check for proper weight initialization
4. Verify data preprocessing and normalization

<!-- RAG_CHUNK_ID: ai-integration-issues -->

## Debugging Techniques

### Logging and Monitoring

#### Effective ROS2 Logging
```python
import rclpy
from rclpy.node import Node
import logging

class DebuggableNode(Node):
    def __init__(self):
        super().__init__('debuggable_node')

        # Set up different log levels
        self.get_logger().set_level(logging.DEBUG)

        # Log important events with context
        self.get_logger().debug('Node initialized with parameters: %s', self.get_parameters_by_descriptor())

        # Log state changes
        self.state = 'idle'
        self.get_logger().info(f'State changed to: {self.state}')

        # Log performance metrics
        import time
        start_time = time.time()
        # ... some operation ...
        elapsed = time.time() - start_time
        self.get_logger().debug(f'Operation completed in {elapsed:.3f}s')
```

<!-- RAG_CHUNK_ID: ros2-logging-best-practices -->

#### Performance Monitoring
```python
import psutil
import GPUtil
from rclpy.qos import QoSProfile
from std_msgs.msg import String

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        # Create publisher for system status
        qos_profile = QoSProfile(depth=1)
        self.status_pub = self.create_publisher(String, '/system_status', qos_profile)

        # Create timer for periodic monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

    def monitor_system(self):
        """Monitor system resources and publish status"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU usage (if available)
        gpu_percent = 0
        gpu_memory_percent = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100
            gpu_memory_percent = gpus[0].memoryUtil * 100

        # Create status message
        status_msg = f"CPU: {cpu_percent}%, Memory: {memory_percent}%, GPU: {gpu_percent}%"

        # Publish status
        status = String()
        status.data = status_msg
        self.status_pub.publish(status)

        # Log warnings if resources are high
        if cpu_percent > 80 or memory_percent > 80 or gpu_percent > 80:
            self.get_logger().warn(f'High resource usage detected: {status_msg}')
```

### Diagnostic Messages
```python
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

class RobotDiagnosticNode(Node):
    def __init__(self):
        super().__init__('robot_diagnostic_node')

        # Publisher for diagnostic messages
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # Timer for periodic diagnostics
        self.diag_timer = self.create_timer(5.0, self.publish_diagnostics)

    def publish_diagnostics(self):
        """Publish comprehensive diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Create diagnostic status for each system
        status_list = []

        # Joint system diagnostics
        joint_diag = DiagnosticStatus()
        joint_diag.name = 'Joint System'
        joint_diag.level = DiagnosticStatus.OK
        joint_diag.message = 'All joints operational'

        # Add key-value pairs for detailed info
        joint_diag.values.extend([
            KeyValue(key='Left Hip Status', value='OK'),
            KeyValue(key='Right Knee Position', value='0.25 rad'),
            KeyValue(key='Total Joints', value='28')
        ])

        status_list.append(joint_diag)

        # Sensor system diagnostics
        sensor_diag = DiagnosticStatus()
        sensor_diag.name = 'Sensor System'
        sensor_diag.level = DiagnosticStatus.OK
        sensor_diag.message = 'All sensors operational'

        sensor_diag.values.extend([
            KeyValue(key='Camera Status', value='OK'),
            KeyValue(key='IMU Status', value='OK'),
            KeyValue(key='LIDAR Status', value='OK')
        ])

        status_list.append(sensor_diag)

        # AI system diagnostics
        ai_diag = DiagnosticStatus()
        ai_diag.name = 'AI System'
        ai_diag.level = DiagnosticStatus.OK
        ai_diag.message = 'AI models loaded and ready'

        ai_diag.values.extend([
            KeyValue(key='VLA Model Status', value='Ready'),
            KeyValue(key='GPU Memory Used', value='6.2/8.0 GB'),
            KeyValue(key='Inference Rate', value='30 Hz')
        ])

        status_list.append(ai_diag)

        diag_array.status = status_list
        self.diag_pub.publish(diag_array)
```

<!-- RAG_CHUNK_ID: diagnostic-monitoring-tools -->

## Error Recovery Strategies

### Graceful Degradation
Implement systems that can continue operating with reduced functionality when components fail:

```python
class RobustRobotController(Node):
    def __init__(self):
        super().__init__('robust_robot_controller')

        # Initialize components with fallback strategies
        self.vision_system_available = True
        self.ai_system_available = True

        # Fallback control methods
        self.position_controller = self.initialize_position_controller()
        self.impedance_controller = self.initialize_impedance_controller()

        # Timer for health checks
        self.health_check_timer = self.create_timer(1.0, self.health_check)

    def health_check(self):
        """Check system health and activate fallbacks if needed"""
        # Check vision system
        vision_healthy = self.check_vision_system()
        if not vision_healthy and self.vision_system_available:
            self.get_logger().warn('Vision system failed, switching to position control')
            self.vision_system_available = False

        # Check AI system
        ai_healthy = self.check_ai_system()
        if not ai_healthy and self.ai_system_available:
            self.get_logger().warn('AI system failed, switching to rule-based control')
            self.ai_system_available = False

    def check_vision_system(self):
        """Check if vision system is responding"""
        try:
            # Implement vision system health check
            # This could be checking if camera topics are active
            # or if vision processing nodes are publishing
            return True  # Placeholder
        except Exception as e:
            self.get_logger().error(f'Vision system health check failed: {e}')
            return False

    def check_ai_system(self):
        """Check if AI system is responding"""
        try:
            # Implement AI system health check
            # This could be checking if AI nodes are active
            # or if model inference is working
            return True  # Placeholder
        except Exception as e:
            self.get_logger().error(f'AI system health check failed: {e}')
            return False

    def execute_action(self, action_request):
        """Execute action with fallback strategies"""
        if self.ai_system_available:
            # Use AI-based control
            try:
                return self.ai_control(action_request)
            except Exception as e:
                self.get_logger().error(f'AI control failed: {e}, falling back to rule-based')
                return self.rule_based_control(action_request)
        else:
            # Use rule-based control
            return self.rule_based_control(action_request)
```

### Emergency Procedures
```python
class EmergencyHandler(Node):
    def __init__(self):
        super().__init__('emergency_handler')

        # Emergency stop publisher
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 1)

        # Joint state subscriber to monitor for anomalies
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Initialize safety thresholds
        self.safety_thresholds = {
            'joint_effort': 100.0,  # N*m
            'joint_velocity': 5.0,  # rad/s
            'temperature': 75.0     # Celsius
        }

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_monitor)

        # Store previous joint states for velocity calculation
        self.prev_joint_states = None
        self.prev_time = None

    def joint_state_callback(self, msg):
        """Store joint states for safety monitoring"""
        self.current_joint_states = msg
        self.current_time = self.get_clock().now()

    def safety_monitor(self):
        """Monitor for safety violations and trigger emergency procedures"""
        if not hasattr(self, 'current_joint_states'):
            return

        # Check for safety violations
        violations = []

        # Check joint efforts
        for i, effort in enumerate(self.current_joint_states.effort):
            if abs(effort) > self.safety_thresholds['joint_effort']:
                violations.append(f"Joint {self.current_joint_states.name[i]} effort violation: {effort}")

        # Check joint velocities (if we have previous state)
        if self.prev_joint_states is not None and self.prev_time is not None:
            time_diff = (self.current_time.nanoseconds - self.prev_time.nanoseconds) / 1e9

            if time_diff > 0:  # Avoid division by zero
                for i in range(len(self.current_joint_states.position)):
                    if i < len(self.prev_joint_states.position):
                        vel = (self.current_joint_states.position[i] -
                              self.prev_joint_states.position[i]) / time_diff

                        if abs(vel) > self.safety_thresholds['joint_velocity']:
                            violations.append(f"Joint {self.current_joint_states.name[i]} velocity violation: {vel}")

        # Trigger emergency stop if violations detected
        if violations:
            self.get_logger().error(f"Safety violations detected: {violations}")
            self.trigger_emergency_stop()

        # Update previous states
        self.prev_joint_states = self.current_joint_states
        self.prev_time = self.current_time

    def trigger_emergency_stop(self):
        """Trigger emergency stop procedure"""
        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        # Log the emergency stop
        self.get_logger().fatal("EMERGENCY STOP ACTIVATED - Safety violations detected")

        # Additional emergency procedures can be added here
        # Such as saving system state, notifying operators, etc.
```

<!-- RAG_CHUNK_ID: error-recovery-strategies -->

## Debugging Workflows

### Development Debugging Workflow
1. **Reproduce**: Consistently reproduce the issue in a controlled environment
2. **Isolate**: Narrow down to the specific component causing the issue
3. **Instrument**: Add logging and monitoring to gather more information
4. **Hypothesize**: Formulate theories about the root cause
5. **Test**: Create specific tests to validate hypotheses
6. **Fix**: Implement the solution
7. **Verify**: Confirm the fix resolves the issue without introducing new problems

### Production Debugging Workflow
1. **Monitor**: Continuously monitor system performance and health
2. **Alert**: Set up alerts for anomalous behavior
3. **Preserve**: Capture system state when issues occur
4. **Analyze**: Investigate captured state and logs
5. **Mitigate**: Implement immediate workaround if needed
6. **Resolve**: Develop permanent fix
7. **Deploy**: Safely deploy fix with proper testing

### Remote Debugging
For debugging robots deployed in remote locations:

```python
class RemoteDebugNode(Node):
    def __init__(self):
        super().__init__('remote_debug_node')

        # Publisher for debug snapshots
        self.snapshot_pub = self.create_publisher(String, '/debug_snapshot', 1)

        # Service to request debug information
        self.debug_service = self.create_service(
            Trigger, '/request_debug_info', self.debug_info_callback
        )

        # Timer for periodic health snapshots
        self.snapshot_timer = self.create_timer(30.0, self.capture_snapshot)

    def capture_snapshot(self):
        """Capture and publish system health snapshot"""
        import json
        import psutil

        snapshot_data = {
            'timestamp': self.get_clock().now().nanoseconds,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'active_nodes': self.get_active_nodes(),
            'topic_statistics': self.get_topic_statistics(),
            'recent_errors': self.get_recent_errors()
        }

        snapshot_msg = String()
        snapshot_msg.data = json.dumps(snapshot_data, indent=2)
        self.snapshot_pub.publish(snapshot_msg)

    def debug_info_callback(self, request, response):
        """Handle request for immediate debug information"""
        try:
            # Capture immediate system state
            debug_info = self.capture_detailed_state()

            response.success = True
            response.message = debug_info
        except Exception as e:
            response.success = False
            response.message = f"Debug info capture failed: {str(e)}"

        return response

    def capture_detailed_state(self):
        """Capture detailed system state for debugging"""
        # This would include detailed information about:
        # - All node states
        # - Current parameter values
        # - Recent log entries
        # - System resource usage
        # - Network connectivity status
        # - Component health statuses
        pass
```

<!-- RAG_CHUNK_ID: debugging-workflows -->

## Troubleshooting Scenarios

### Scenario 1: Robot Not Responding to Commands
**Problem**: Robot receives commands but doesn't execute actions
**Troubleshooting Steps**:
1. Check if command topic has subscribers: `ros2 topic info /cmd_vel`
2. Verify controller is running: `ros2 node list | grep controller`
3. Check controller parameters: `ros2 param list -n /controller_node`
4. Monitor controller state: `ros2 topic echo /controller_state`
5. Check joint state publisher: `ros2 topic echo /joint_states`
6. Verify hardware interface communication

### Scenario 2: AI Model Producing Erroneous Actions
**Problem**: VLA system generates inappropriate or dangerous actions
**Troubleshooting Steps**:
1. Verify input preprocessing pipeline
2. Check model confidence scores
3. Validate action space bounds
4. Inspect training data quality
5. Test with known good inputs
6. Implement action validation layer

### Scenario 3: Simulation Performance Degradation
**Problem**: Gazebo simulation running slowly or with dropped frames
**Troubleshooting Steps**:
1. Monitor system resources: `htop`, `nvidia-smi`
2. Check Gazebo physics update rate
3. Verify graphics driver performance
4. Reduce simulation complexity temporarily
5. Check for inefficient plugins or callbacks
6. Optimize collision geometries

<!-- RAG_CHUNK_ID: troubleshooting-scenarios -->

## Best Practices

### Preventive Measures
1. **Comprehensive Testing**: Implement unit, integration, and system testing
2. **Monitoring**: Set up continuous monitoring for all critical systems
3. **Validation**: Validate inputs and outputs at each system boundary
4. **Documentation**: Maintain clear documentation of system behavior
5. **Version Control**: Use version control for all code and configuration
6. **Backup Plans**: Maintain fallback procedures for critical failures

### Diagnostic Tool Development
Create custom diagnostic tools for your specific robot:

```python
class CustomRobotDiagnostics:
    """Custom diagnostic tools for humanoid robot systems"""

    @staticmethod
    def check_balance_stability(robot_state):
        """Check if robot is maintaining balance"""
        # Calculate ZMP (Zero Moment Point) if available
        # Check if ZMP is within support polygon
        pass

    @staticmethod
    def validate_action_feasibility(action):
        """Validate if action is mechanically feasible"""
        # Check joint limits
        # Check kinematic constraints
        # Check dynamic feasibility
        pass

    @staticmethod
    def assess_sensor_health(sensor_data):
        """Assess health of sensor systems"""
        # Check for sensor timeouts
        # Validate sensor data ranges
        # Monitor sensor noise levels
        pass
```

### Knowledge Base Maintenance
- Document common issues and their solutions
- Create troubleshooting checklists for different subsystems
- Maintain hardware-specific troubleshooting guides
- Record performance baselines for comparison
- Keep track of known issues and workarounds

<!-- RAG_CHUNK_ID: troubleshooting-best-practices -->

## Summary
Effective troubleshooting and debugging of AI-native humanoid robotics systems requires a systematic approach, appropriate tools, and well-designed error recovery strategies. By implementing comprehensive monitoring, logging, and diagnostic capabilities, you can quickly identify and resolve issues while maintaining system safety and reliability. The key is to anticipate potential problems, implement preventive measures, and maintain detailed documentation of system behavior and known solutions.

## Quick Reference
- Use `ros2 doctor` for comprehensive system diagnostics
- Monitor resource usage with `htop` and `nvidia-smi`
- Check topic connectivity with `ros2 topic info`
- Use RViz for visual debugging of transforms and sensor data
- Implement emergency stop capabilities in all safety-critical systems
- Maintain system snapshots for post-incident analysis

## Practice Exercises
1. Simulate a joint limit violation and implement appropriate error handling
2. Create a diagnostic node that monitors multiple subsystems
3. Implement a fallback controller for when AI systems fail
4. Design a remote debugging protocol for deployed robots

## Check Your Understanding
**Multiple Choice Questions:**
1. What is the first step in the systematic troubleshooting methodology?
   A) Implement a solution
   B) Reproduce the issue
   C) Formulate a hypothesis
   D) Gather information

2. Which tool is used to check resource usage on NVIDIA GPUs?
   A) htop
   B) df -h
   C) nvidia-smi
   D) netstat

3. What should you check first when a robot doesn't respond to commands?
   A) AI model accuracy
   B) Topic connectivity and subscribers
   C) Joint limits
   D) Sensor calibration

<!-- RAG_CHUNK_ID: troubleshooting-guide-summary -->
<!-- URDU_TODO: Translate this guide to Urdu -->