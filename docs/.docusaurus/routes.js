import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/humanoid-robotics-book/__docusaurus/debug',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug', 'fbb'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/config',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/config', '58c'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/content',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/content', '9ef'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/globalData', '5e0'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/metadata', 'a6b'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/registry',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/registry', '969'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/routes',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/routes', '919'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/docs',
    component: ComponentCreator('/humanoid-robotics-book/docs', 'd47'),
    routes: [
      {
        path: '/humanoid-robotics-book/docs',
        component: ComponentCreator('/humanoid-robotics-book/docs', '514'),
        routes: [
          {
            path: '/humanoid-robotics-book/docs',
            component: ComponentCreator('/humanoid-robotics-book/docs', 'dab'),
            routes: [
              {
                path: '/humanoid-robotics-book/docs/assessments/gazebo-assessment',
                component: ComponentCreator('/humanoid-robotics-book/docs/assessments/gazebo-assessment', '1f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/assessments/intro',
                component: ComponentCreator('/humanoid-robotics-book/docs/assessments/intro', 'c34'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/assessments/nvidia-isaac-assessment',
                component: ComponentCreator('/humanoid-robotics-book/docs/assessments/nvidia-isaac-assessment', '408'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/assessments/ros2-assessment',
                component: ComponentCreator('/humanoid-robotics-book/docs/assessments/ros2-assessment', '01a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/assessments/vla-assessment',
                component: ComponentCreator('/humanoid-robotics-book/docs/assessments/vla-assessment', 'eed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/intro',
                component: ComponentCreator('/humanoid-robotics-book/docs/intro', '3e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/modules/gazebo-unity',
                component: ComponentCreator('/humanoid-robotics-book/docs/modules/gazebo-unity', '083'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/modules/nvidia-isaac',
                component: ComponentCreator('/humanoid-robotics-book/docs/modules/nvidia-isaac', 'fab'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/modules/ros2',
                component: ComponentCreator('/humanoid-robotics-book/docs/modules/ros2', 'be5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/modules/vla',
                component: ComponentCreator('/humanoid-robotics-book/docs/modules/vla', '84a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/week1-13/week1',
                component: ComponentCreator('/humanoid-robotics-book/docs/week1-13/week1', '0d0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/week1-13/week2',
                component: ComponentCreator('/humanoid-robotics-book/docs/week1-13/week2', '789'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/week1-13/week3',
                component: ComponentCreator('/humanoid-robotics-book/docs/week1-13/week3', 'eec'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics-book/docs/week1-13/week4',
                component: ComponentCreator('/humanoid-robotics-book/docs/week1-13/week4', '027'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
