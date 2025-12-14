import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import mermaid from 'mermaid';

// DiagramViewer component to handle both Mermaid and ASCII diagrams
const DiagramViewer = ({ type, content, caption, altText }) => {
  const renderMermaid = (containerId, diagramCode) => {
    if (typeof window !== 'undefined') {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        securityLevel: 'loose'
      });

      try {
        mermaid.render(containerId, diagramCode, (svgCode) => {
          const element = document.getElementById(containerId);
          if (element) {
            element.innerHTML = svgCode;
          }
        });
      } catch (error) {
        console.error('Error rendering mermaid diagram:', error);
        const element = document.getElementById(containerId);
        if (element) {
          element.innerHTML = `<pre>${diagramCode}</pre>`;
        }
      }
    }
  };

  if (type === 'mermaid') {
    const containerId = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    return (
      <div className="diagram-container">
        <BrowserOnly
          fallback={<div>Loading diagram...</div>}
        >
          {() => {
            renderMermaid(containerId, content);
            return <div id={containerId}></div>;
          }}
        </BrowserOnly>
        {caption && <div className="diagram-caption" aria-label={altText || caption}>{caption}</div>}
      </div>
    );
  } else if (type === 'ascii') {
    return (
      <div className="ascii-diagram-container">
        <pre className="ascii-diagram">{content}</pre>
        {caption && <div className="diagram-caption" aria-label={altText || caption}>{caption}</div>}
      </div>
    );
  } else {
    // For image diagrams
    return (
      <div className="image-diagram-container">
        <img src={content} alt={altText} />
        {caption && <div className="diagram-caption">{caption}</div>}
      </div>
    );
  }
};

export default DiagramViewer;