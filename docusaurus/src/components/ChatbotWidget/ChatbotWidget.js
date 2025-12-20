import { useState, useEffect, useRef } from "react";
import axios from "axios";
import styles from "./ChatbotWidget.module.css";

const ChatbotWidget = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedText, setSelectedText] = useState("");

  const chatBodyRef = useRef(null);
  const chatModalRef = useRef(null);

  const toggleChat = () => setOpen(!open);

  // 🔹 Close chat when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        open &&
        chatModalRef.current &&
        !chatModalRef.current.contains(event.target) &&
        !event.target.closest(`.${styles.widgetButton}`)
      ) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [open]);

  // 🔹 Auto scroll to bottom
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // 🔹 Detect selected text from book
  useEffect(() => {
    const handleMouseUp = () => {
      const text = window.getSelection()?.toString().trim();
      if (text && text.length > 20) {
        setSelectedText(text);
        setOpen(true);
      }
    };
    document.addEventListener("mouseup", handleMouseUp);
    return () => document.removeEventListener("mouseup", handleMouseUp);
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userText = input;
    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: userText }]);

    try {
      let res;

      if (selectedText) {
        res = await axios.post("https://01talha-humanoid-robotics-book.hf.space/ask/selection", {
          question: "Explain the selected text in simple terms.",
          selected_text: selectedText,
        });
      } else {
        res = await axios.post("https://01talha-humanoid-robotics-book.hf.space/ask", {
          question: userText,
        });
      }

      setMessages((prev) => [
        ...prev,
        { role: "bot", text: res.data.answer },
      ]);

      setSelectedText("");
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "⚠️ Sorry, something went wrong." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleWidgetClick = (e) => {
    e.stopPropagation();
    toggleChat();
  };

  const handleModalClick = (e) => {
    e.stopPropagation();
  };

  return (
    <>
      <div 
        className={styles.widgetButton} 
        onClick={handleWidgetClick}
        role="button"
        aria-label="Open chat"
      >
        {open ? "✖" : "💬"}
      </div>

      {open && (
        <div 
          className={styles.chatModal} 
          ref={chatModalRef}
          onClick={handleModalClick}
        >
          <div className={styles.chatHeader}>
            <span>📘 AI Book Assistant</span>
            <button 
              onClick={toggleChat} 
              className={styles.closeButton}
              aria-label="Close chat"
            >
              ✖
            </button>
          </div>

          <div className={styles.chatBody} ref={chatBodyRef}>
            {selectedText && (
              <div className={styles.selectedBox}>
                <strong>Selected Text</strong>
                <p>{selectedText.slice(0, 300)}...</p>
              </div>
            )}

            {messages.length === 0 && !selectedText && (
              <div className={styles.welcomeMessage}>
                <h3>Welcome to Book Assistant!</h3>
                <p>You can:</p>
                <ul>
                  <li>Ask questions about the book</li>
                  <li>Select text from the book to get explanations</li>
                  <li>Get summaries of specific sections</li>
                </ul>
              </div>
            )}

            {messages.map((m, i) => (
              <div
                key={i}
                className={
                  m.role === "user" ? styles.userRow : styles.botRow
                }
              >
                <span
                  className={
                    m.role === "user" ? styles.userMsg : styles.botMsg
                  }
                >
                  {m.text}
                </span>
              </div>
            ))}

            {loading && (
              <div className={styles.loadingRow}>
                <div className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>

          <div className={styles.inputContainer}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Ask about the book or selected text…"
              className={styles.chatInput}
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={sendMessage}
              className={styles.sendButton}
              disabled={loading}
            >
              {loading ? "..." : "Send"}
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatbotWidget;