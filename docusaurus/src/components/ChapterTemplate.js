import React from 'react';
import clsx from 'clsx';
import styles from './ChapterTemplate.module.css';

// This is a basic template component for textbook chapters
// It can be customized based on specific chapter needs
const ChapterTemplate = ({ children, title, description, learningOutcomes }) => {
  return (
    <div className={styles.chapterContainer}>
      <header className={styles.chapterHeader}>
        <h1 className={styles.chapterTitle}>{title}</h1>
        {description && <p className={styles.chapterDescription}>{description}</p>}
      </header>

      {learningOutcomes && learningOutcomes.length > 0 && (
        <section className={styles.learningOutcomes}>
          <h2>Learning Outcomes</h2>
          <ul>
            {learningOutcomes.map((outcome, index) => (
              <li key={index}>{outcome}</li>
            ))}
          </ul>
        </section>
      )}

      <main className={styles.chapterContent}>
        {children}
      </main>
    </div>
  );
};

export default ChapterTemplate;