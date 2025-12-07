import React, { useState } from 'react';
import { usePrismTheme } from 'docusaurus-theme-classic';

const CodeExample = ({ children, language = 'python', title = 'Code Example' }) => {
  const [copied, setCopied] = useState(false);
  const prismTheme = usePrismTheme();

  const copyToClipboard = () => {
    navigator.clipboard.writeText(children.trim());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Determine background color from theme
  const backgroundColor = prismTheme.plain.backgroundColor || '#f6f8fa';

  return (
    <div className="code-example-container" style={{
      border: '1px solid #ddd',
      borderRadius: '6px',
      margin: '16px 0',
      overflow: 'hidden'
    }}>
      <div style={{
        backgroundColor: backgroundColor,
        padding: '8px 12px',
        borderBottom: '1px solid #ddd',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <span style={{ fontWeight: 'bold', color: '#333' }}>{title}</span>
        <button
          onClick={copyToClipboard}
          style={{
            background: 'none',
            border: '1px solid #ccc',
            borderRadius: '4px',
            padding: '4px 8px',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre style={{
        margin: 0,
        padding: '12px',
        overflowX: 'auto',
        backgroundColor: backgroundColor
      }}>
        <code className={`language-${language}`}>
          {children}
        </code>
      </pre>
      <div style={{
        padding: '8px 12px',
        backgroundColor: '#f8f9fa',
        fontSize: '12px',
        color: '#666',
        borderTop: '1px solid #ddd'
      }}>
        You can run this code in your local environment or with a Python/ROS setup.
      </div>
    </div>
  );
};

export default CodeExample;