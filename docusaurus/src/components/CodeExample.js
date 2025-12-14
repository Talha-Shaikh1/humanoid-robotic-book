import React from 'react';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';

// CodeExample component to display code with language tabs and execution instructions
const CodeExample = ({
  title,
  description,
  code,
  language = 'python',
  dependencies = [],
  instructions = [],
  rosDistro = 'Humble',
  launchCommands = [],
  expectedOutput = ''
}) => {
  return (
    <div className="code-example-container">
      <div className="code-example-header">
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </div>

      <div className="code-example-content">
        <Tabs>
          <TabItem value="code" label="Code">
            <CodeBlock language={language}>
              {code}
            </CodeBlock>
          </TabItem>

          {expectedOutput && (
            <TabItem value="output" label="Expected Output">
              <CodeBlock language="text">
                {expectedOutput}
              </CodeBlock>
            </TabItem>
          )}
        </Tabs>

        <div className="code-example-details">
          {dependencies.length > 0 && (
            <div className="dependencies-section">
              <h4>Dependencies:</h4>
              <ul>
                {dependencies.map((dep, index) => (
                  <li key={index}>{dep}</li>
                ))}
              </ul>
            </div>
          )}

          {instructions.length > 0 && (
            <div className="instructions-section">
              <h4>Instructions:</h4>
              <ol>
                {instructions.map((inst, index) => (
                  <li key={index}>{inst}</li>
                ))}
              </ol>
            </div>
          )}

          <div className="environment-section">
            <h4>Environment:</h4>
            <p>ROS2 Distribution: {rosDistro}</p>
          </div>

          {launchCommands.length > 0 && (
            <div className="commands-section">
              <h4>Launch Commands:</h4>
              <CodeBlock language="bash">
                {launchCommands.join('\n')}
              </CodeBlock>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CodeExample;