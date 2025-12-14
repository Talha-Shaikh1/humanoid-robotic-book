import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function Hero() {
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.fadeUp}>
          <h1 className="hero__title">
            <span role="img" aria-label="robot">🤖</span> AI‑Native Humanoid Robotics
          </h1>
          <p className="hero__subtitle">
            Master ROS 2, Simulation, and AI‑Driven Robotics with hands‑on labs and real‑world projects
          </p>
          <div className={styles.heroActions}>
            <Link 
              className="button button--primary button--lg" 
              to="/docs/module-1-ros2/chapter-1-introduction-to-ros2"
            >
              <span role="img" aria-label="rocket">🚀</span> Start Learning
            </Link>
            <Link 
              className="button button--secondary button--lg" 
              to="/docs"
            >
              <span role="img" aria-label="books">📚</span> View Curriculum
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function QuickLinks() {
  const links = [
    {
      icon: '⚡',
      title: 'ROS 2 Fundamentals',
      desc: 'Learn robotic operating system from basics to advanced',
      to: '/docs/module-1-ros2/chapter-1-introduction-to-ros2',
      color: '#3b82f6'
    },
    {
      icon: '🎮',
      title: 'Simulation Labs',
      desc: 'Gazebo & Unity digital twins for testing and development',
      to: '/docs/module-2-gazebo-unity/chapter-1-simulation-environments',
      color: '#8b5cf6'
    },
    {
      icon: '🧠',
      title: 'AI Integration',
      desc: 'Vision-Language-Action models for intelligent robots',
      to: '/docs/module-4-vla/chapter-1-vla-introduction',
      color: '#10b981'
    },
    {
      icon: '🤖',
      title: 'Humanoid Projects',
      desc: 'Build and program humanoid robots from scratch',
      to: '/docs',
      color: '#f59e0b'
    }
  ];

  return (
    <section className={styles.quickLinks}>
      <div className="container">
        <h2>Quick Start Guides</h2>
        <div className={styles.linksGrid}>
          {links.map((link) => (
            <Link key={link.title} className={styles.linkCard} to={link.to}>
              <h3>
                <span style={{ color: link.color }}>{link.icon}</span>
                {link.title}
              </h3>
              <p>{link.desc}</p>
              <span className={styles.linkCta}>
                Explore <span>→</span>
              </span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function QuickActions() {
  return (
    <section className={styles.quickActions}>
      <div className="container">
        <div className={styles.actionsBar}>
          <Link 
            className="button button--outline button--secondary" 
            to="/docs"
          >
            <span role="img" aria-label="tutorial">📖</span> Tutorials
          </Link>
          <Link 
            className="button button--outline button--secondary" 
            to="/docs"
          >
            <span role="img" aria-label="lab">🔬</span> Hands-on Labs
          </Link>
          <Link 
            className="button button--outline button--secondary" 
            to="https://github.com/Talha-Shaikh1/humanoid-robotic-book"
            target="_blank"
          >
            <span role="img" aria-label="github">💻</span> GitHub Repo
          </Link>
          <Link 
            className="button button--outline button--secondary" 
            to="/docs"
          >
            <span role="img" aria-label="community">👥</span> Community
          </Link>
        </div>
      </div>
    </section>
  );
}

function Features() {
  const features = [
    {
      icon: '🧪',
      title: 'Hands‑On First',
      desc: 'Practical labs with every theory lesson'
    },
    {
      icon: '🏭',
      title: 'Industry‑Ready',
      desc: 'Learn tools used by robotics companies'
    },
    {
      icon: '🧩',
      title: 'Modular Design',
      desc: 'Learn in order or jump to what you need'
    },
    {
      icon: '🚀',
      title: 'Production‑Ready',
      desc: 'Code and examples ready for deployment'
    }
  ];

  return (
    <section className={styles.features}>
      <div className="container">
        <h2>Why Choose This Curriculum?</h2>
        <div className={styles.featuresGrid}>
          {features.map((feature) => (
            <div key={feature.title} className={styles.featureItem}>
              <div className={styles.featureIcon}>{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTA() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <h2>Start Building Intelligent Robots Today</h2>
        <p>
          Join hundreds of developers learning cutting‑edge robotics with AI integration
        </p>
        <div className={styles.ctaButtons}>
          <Link 
            className="button button--primary button--lg" 
            to="/docs"
          >
            <span role="img" aria-label="book">📘</span> Get Started Free
          </Link>
          <Link 
            className="button button--secondary button--lg" 
            to="https://github.com/Talha-Shaikh1/humanoid-robotic-book"
            target="_blank"
          >
            <span role="img" aria-label="star">⭐</span> Star on GitHub
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  
  return (
    <Layout
      title={siteConfig.title}
      description="Learn AI‑Native Humanoid Robotics with ROS 2, Simulation, and VLA models"
    >
      <Hero />
      <QuickLinks />
      <QuickActions />
      <Features />
      <CTA />
    </Layout>
  );
}