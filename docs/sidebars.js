/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Modules',
      items: [
        'modules/ros2',
        'modules/gazebo-unity',
        'modules/nvidia-isaac',
        'modules/vla'
      ],
    },
    {
      type: 'category',
      label: 'Weekly Breakdown',
      items: [
        'week1-13/week1',
        'week1-13/week2',
        'week1-13/week3',
        'week1-13/week4',
      ],
    },
    {
      type: 'category',
      label: 'Assessments',
      items: [
        'assessments/intro',
        'assessments/ros2-assessment',
        'assessments/gazebo-assessment',
        'assessments/nvidia-isaac-assessment',
        'assessments/vla-assessment'
      ],
    },
  ],
};

module.exports = sidebars;