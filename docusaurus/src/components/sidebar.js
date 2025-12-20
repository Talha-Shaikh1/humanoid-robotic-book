// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Home',
      items: ['index'], // Assuming there's an index file
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1 - ROS 2: Robotic Nervous System',
      items: [
        'module-1-ros2/chapter-1-introduction-to-ros2',
        'module-1-ros2/chapter-2-ros2-nodes-and-topics',
        'module-1-ros2/chapter-3-ros2-services-and-actions',
        'module-1-ros2/chapter-4-ros2-parameters-and-lifecycle',
        'module-1-ros2/chapter-5-ros2-launch-systems',
        'module-1-ros2/chapter-6-ros2-testing-and-debugging'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2 - Gazebo & Unity: Digital Twin',
      items: [
        'module-2-gazebo-unity/chapter-1-simulation-environments',
        'module-2-gazebo-unity/chapter-2-urdf-modeling',
        'module-2-gazebo-unity/chapter-3-gazebo-integration'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3 - NVIDIA Isaac: AI-Robot Brain',
      items: [
        'module-3-isaac/chapter-1-isaac-architecture',
        'module-3-isaac/chapter-2-isaac-ros-integration',
        'module-3-isaac/chapter-3-isaac-ai-perception'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4 - Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/chapter-1-vla-introduction',
        'module-4-vla/chapter-2-vla-architectures',
        'module-4-vla/chapter-3-vla-training-methods'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Appendix',
      items: [
        'appendix/workstation-specs',
        'appendix/rag-implementation',
        'appendix/urdu-translation-guide'
      ],
      collapsed: true,
    }
  ],
};

export default sidebars;