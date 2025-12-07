import React, { useState, useEffect } from 'react';
import { useDocusaurusContext } from '@docusaurus/core';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Initialize with a welcome message
  useEffect(() => {
    setMessages([
      {
        id: 1,
        role: 'assistant',
        content: 'Hello! I\'m your AI assistant for the Physical AI & Humanoid Robotics textbook. How can I help you today?'
      }
    ]);
  }, []);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // In a real implementation, this would call the backend API
      // For now, we'll simulate a response
      setTimeout(() => {
        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: `I received your message: "${inputValue}". In a full implementation, this would connect to our RAG system to provide relevant textbook content.`
        };
        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);

      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <div className="chatbot-container" style={{
      border: '1px solid #ccc',
      borderRadius: '8px',
      padding: '16px',
      margin: '16px 0',
      maxHeight: '500px',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <h3>AI Assistant</h3>

      <div className="chat-messages" style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '16px',
        maxHeight: '300px'
      }}>
        {messages.map((message) => (
          <div
            key={message.id}
            style={{
              textAlign: message.role === 'user' ? 'right' : 'left',
              marginBottom: '8px'
            }}
          >
            <span style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: '18px',
              backgroundColor: message.role === 'user' ? '#007cba' : '#f0f0f0',
              color: message.role === 'user' ? 'white' : 'black',
              maxWidth: '80%'
            }}>
              {message.content}
            </span>
          </div>
        ))}
      </div>

      <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about the textbook..."
          style={{
            flex: 1,
            padding: '8px 12px',
            border: '1px solid #ccc',
            borderRadius: '18px'
          }}
          disabled={isLoading}
        />
        <button
          type="submit"
          style={{
            padding: '8px 16px',
            backgroundColor: '#007cba',
            color: 'white',
            border: 'none',
            borderRadius: '18px',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
          disabled={isLoading || !inputValue.trim()}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default Chatbot;