import React from 'react';
import Layout from '@theme-original/Layout';
import ChatbotWidget from '../../../docusaurus/src/components/ChatbotWidget/ChatbotWidget';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <ChatbotWidget />
    </>
  );
}
