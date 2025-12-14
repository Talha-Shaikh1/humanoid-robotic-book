---
title: Recommended Workstation Specifications
description: Hardware and software requirements for robotics development
sidebar_position: 1
---

# Recommended Workstation Specifications

## Minimum Hardware Requirements

### CPU
- **Recommended**: Multi-core processor (Intel i7 or AMD Ryzen 7)
- **Minimum**: Quad-core processor with 2.5 GHz clock speed
- **Note**: Robotics simulations are CPU-intensive, especially for physics calculations

### RAM
- **Recommended**: 32 GB DDR4 RAM
- **Minimum**: 16 GB DDR4 RAM
- **Note**: Gazebo and Isaac Sim simulations require significant memory

### Graphics Card
- **Recommended**: NVIDIA RTX 3060 or higher with 8GB+ VRAM
- **Minimum**: NVIDIA GTX 1060 or AMD equivalent with 6GB+ VRAM
- **Requirements**:
  - OpenGL 3.3+ support
  - CUDA support for Isaac Sim (NVIDIA cards)
  - Dedicated GPU for simulation rendering

### Storage
- **Recommended**: 1 TB SSD
- **Minimum**: 500 GB SSD
- **Note**: SSDs significantly improve build and simulation performance

### Operating System
- **Required**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Alternative**: Ubuntu 20.04 LTS (Focal Fossa) with ROS2 compatibility

## Software Requirements

### ROS2 Distribution
- **Recommended**: ROS2 Humble Hawksbill (LTS)
- **Alternative**: ROS2 Iron Irwini (for newer features)

### Development Tools
- **Build System**: colcon
- **Package Manager**: apt, pip, conda
- **IDE Options**:
  - Visual Studio Code with ROS extension
  - CLion for C++ development
  - PyCharm for Python development

### Simulation Environments
- **Gazebo**: Garden or Fortress
- **Isaac Sim**: Latest version compatible with your GPU
- **Docker**: For containerized development environments

## Cloud Alternatives

### AWS RoboMaker
- Managed service for robotics applications
- Integration with ROS/ROS2
- Simulation capabilities
- Development and deployment tools

### Google Cloud Platform
- Compute Engine instances with GPU support
- Container-Optimized OS for robotics workloads
- Integration with ML/AI services

### Azure IoT Robotics
- Cloud-based robotics services
- Simulation and testing environments
- Integration with Azure AI services

### Container-Based Development

#### Docker Setup
```bash
# Install Docker
sudo apt update
sudo apt install docker.io
sudo usermod -a -G docker $USER
```

#### Recommended Docker Images
- `osrf/ros:humble-desktop-full` - Full ROS2 Humble installation
- `nvidia/cuda:11.8-devel-ubuntu22.04` - CUDA development environment

#### Docker Compose for Robotics Development
```yaml
version: '3.8'
services:
  ros2-dev:
    image: osrf/ros:humble-desktop-full
    container_name: ros2-humble-dev
    environment:
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - .:/workspace
    network_mode: host
    stdin_open: true
    tty: true
```

## Setup Checklist

### Before Installation
- [ ] Verify hardware meets minimum requirements
- [ ] Backup existing system if applicable
- [ ] Ensure stable internet connection for downloads
- [ ] Allocate sufficient time (2-4 hours for full setup)

### Installation Process
- [ ] Install Ubuntu 22.04 LTS
- [ ] Update system packages
- [ ] Install ROS2 Humble
- [ ] Install Gazebo simulation environment
- [ ] Install Isaac Sim (if applicable)
- [ ] Configure development environment
- [ ] Test basic ROS2 functionality

### Post-Installation Verification
- [ ] Verify ROS2 installation: `ros2 --version`
- [ ] Test basic ROS2 commands: `ros2 run demo_nodes_cpp talker`
- [ ] Verify Gazebo installation: `gazebo --version`
- [ ] Test basic simulation environment
- [ ] Confirm all required packages are accessible

## Troubleshooting Common Issues

### ROS2 Installation Issues
- Ensure correct Ubuntu version is installed
- Verify system locale settings (should be UTF-8)
- Check internet connectivity and proxy settings

### Graphics/Simulation Issues
- Verify GPU drivers are correctly installed
- Check OpenGL support: `glxinfo | grep "OpenGL version"`
- Ensure proper X11 forwarding for GUI applications

### Performance Optimization
- Increase swap space for memory-intensive simulations
- Configure CPU governor for performance mode
- Consider using zram for additional virtual memory

## Additional Resources

- [ROS2 Installation Guide](https://docs.ros.org/en/humble/Installation.html)
- [Gazebo Installation Guide](http://gazebosim.org/tutorials?cat=install)
- [Ubuntu Hardware Requirements](https://help.ubuntu.com/community/Installation/SystemRequirements)
- [NVIDIA GPU Compatibility](https://developer.nvidia.com/cuda-gpus)