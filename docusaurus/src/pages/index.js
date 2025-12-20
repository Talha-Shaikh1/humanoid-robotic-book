import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">🤖 AI-Native Humanoid Robotics Textbook</h1>
        <p className="hero__subtitle">Your comprehensive guide to Physical AI & Humanoid Robotics</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/module-1-ros2/chapter-1-introduction-to-ros2">
            📚 Start Learning - Module 1
          </Link>
          <Link
            className="button button--primary button--lg"
            to="/docs">
            📖 View Full Curriculum
          </Link>
        </div>
        <div className={styles.quickModules}>
          <Link
            className="button button--outline button--lg"
            to="/docs/module-1-ros2/chapter-1-introduction-to-ros2">
            🧠 ROS 2: Robotic Nervous System
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/module-2-gazebo-unity/chapter-1-simulation-environments">
            🌐 Gazebo & Unity: Digital Twin
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/module-3-isaac/chapter-1-isaac-architecture">
            🤖 NVIDIA Isaac: AI-Robot Brain
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/module-4-vla/chapter-1-vla-introduction">
            🧠 Vision-Language-Action (VLA)
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Comprehensive textbook for AI-Native Humanoid Robotics">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <h2>🎯 Learning Approach</h2>
                <p>This textbook follows best practices for robotics education:</p>
                <ul>
                  <li>🎯 <strong>Hands-on Learning</strong>: Each chapter includes practical exercises</li>
                  <li>🔧 <strong>Real-World Applications</strong>: Concepts connected to actual robotics problems</li>
                  <li>🔄 <strong>Modular Design</strong>: Learn at your own pace through interconnected modules</li>
                  <li>🤖 <strong>AI-Native Approach</strong>: Modern AI techniques integrated throughout</li>
                </ul>
              </div>
            </div>

            <div className="row" style={{marginTop: '2rem'}}>
              <div className="col col--12">
                <h2>🚀 Getting Started</h2>
                <ol>
                  <li><strong>Begin with <Link to="/docs/module-1-ros2/chapter-1-introduction-to-ros2">Module 1</Link></strong> if you're new to robotics</li>
                  <li><strong>Proceed through modules</strong> sequentially or jump to specific topics</li>
                  <li><strong>Complete hands-on exercises</strong> to reinforce learning</li>
                  <li><strong>Use the quizzes and assessments</strong> to validate your understanding</li>
                </ol>
              </div>
            </div>

            <div className="row" style={{marginTop: '2rem', backgroundColor: '#f5f5f5', padding: '1.5rem', borderRadius: '8px'}}>
              <div className="col col--12">
                <h2>👥 Target Audience</h2>
                <ul>
                  <li>🎓 Students learning Physical AI, ROS 2, Gazebo, Isaac Sim, VLA</li>
                  <li>👨‍💻 Beginners → intermediate robotics developers</li>
                  <li>🧑‍🏫 Educators adopting a robotics curriculum</li>
                  <li>🏆 Hackathon judges evaluating textbook quality and reproducibility</li>
                </ul>
              </div>
            </div>
          </div>
        </section>
      </main>
      
    </Layout>
  );
}