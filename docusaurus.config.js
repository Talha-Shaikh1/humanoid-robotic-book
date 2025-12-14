// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI-Native Humanoid Robotics Textbook',
  tagline: 'A comprehensive guide to Physical AI & Humanoid Robotics',
  favicon: 'img/favicon.ico',

    url: 'https://humanoid-robotic-book-eight.vercel.app/',
    baseUrl: '/humanoid-robotic-book/',


  // GitHub repo info
  organizationName: 'Talha-Shaikh1',
  projectName: 'humanoid-robotic-book',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: './docusaurus/docs',
          sidebarPath: './docusaurus/sidebars.js',
          editUrl:
            'https://github.com/Talha-Shaikh1/humanoid-robotic-book/tree/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'Humanoid Robotics Textbook',
      logo: {
        alt: 'Robotics Textbook Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/Talha-Shaikh1/humanoid-robotic-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            {
              label: 'ROS 2: Robotic Nervous System',
              to: '/docs/module-1-ros2/chapter-1-introduction-to-ros2',
            },
            {
              label: 'Gazebo & Unity: Digital Twin',
              to: '/docs/module-2-gazebo-unity/chapter-1-simulation-environments',
            },
            {
              label: 'NVIDIA Isaac: AI-Robot Brain',
              to: '/docs/module-3-isaac/chapter-1-isaac-architecture',
            },
            {
              label: 'Vision-Language-Action (VLA)',
              to: '/docs/module-4-vla/chapter-1-vla-introduction',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Talha-Shaikh1/humanoid-robotic-book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI-Native Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'cpp', 'bash', 'json', 'yaml'],
    },
  },
};

export default config;
