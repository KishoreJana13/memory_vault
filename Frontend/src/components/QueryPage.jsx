import React, { useState, useEffect, useRef } from "react";
import "./QueryPage.css";
import { Sparkles, MessageSquare, PlusCircle, X, Send, Clock, Loader } from "lucide-react";
import axios from "axios";

const QueryPage = () => {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [queries, setQueries] = useState([]);
  const [currentQuery, setCurrentQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const conversationEndRef = useRef(null);

  const suggestedPrompts = [
    "Tell me about a happy memory",
    "Tell me about a time I spent with my family",
    "What was my favorite childhood toy?",
    "Describe a memorable vacation I took",
  ];

  useEffect(() => {
    const savedConversations = JSON.parse(
      localStorage.getItem("conversations") || "[]"
    );
    const sortedConversations = savedConversations.sort((a, b) => b.id - a.id);
    setConversations(sortedConversations);

    if (sortedConversations.length === 0) {
      startNewConversation(true);
    } else {
      setCurrentConversationId(sortedConversations[0].id);
      setQueries(sortedConversations[0].queries);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [queries]);

  const scrollToBottom = () => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const saveToLocalStorage = (updatedConversations) => {
    localStorage.setItem("conversations", JSON.stringify(updatedConversations));
  };

  const startNewConversation = (isInitialLoad = false) => {
    const newId = Date.now().toString();
    const newConversation = { id: newId, queries: [] };
    const updatedConversations = [newConversation, ...conversations];
    setConversations(updatedConversations);
    setCurrentConversationId(newId);
    setQueries([]);
    saveToLocalStorage(updatedConversations);
  };

  const deleteConversation = (id, event) => {
    event.stopPropagation();
    const updatedConversations = conversations.filter((conv) => conv.id !== id);
    setConversations(updatedConversations);
    saveToLocalStorage(updatedConversations);

    if (id === currentConversationId) {
      if (updatedConversations.length > 0) {
        setCurrentConversationId(updatedConversations[0].id);
        setQueries(updatedConversations[0].queries);
      } else {
        setCurrentConversationId(null);
        setQueries([]);
      }
    }
  };

  const handlePromptClick = (prompt) => {
    setShowWelcome(false);
    startNewConversation();
    setCurrentQuery(prompt);
    handleSubmit(null, prompt);
  };

  const selectConversation = (id) => {
    setCurrentConversationId(id);
    const selectedConversation = conversations.find((conv) => conv.id === id);
    setQueries(selectedConversation.queries);
    setShowWelcome(false);
  };

  const handleSubmit = async (e, manualPrompt = null) => {
    if (e) e.preventDefault();
    const queryToSend = manualPrompt || currentQuery;
    if (!queryToSend.trim()) return;

    const newQuery = {
      query: queryToSend,
      response: null,
      image: null,
      isLoading: true
    };

    const updatedQueries = [...queries, newQuery];
    setQueries(updatedQueries);
    setCurrentQuery("");
    setIsLoading(true);

    try {
      const response = await axios.get(
        "http://localhost:8080/query",
        {
          params: { query: queryToSend },
        }
      );

      const updatedQueriesWithResponse = updatedQueries.map((q, index) =>
        index === updatedQueries.length - 1
          ? {
              ...q,
              response: response.data.text,
              image: response.data.image,
              isLoading: false
            }
          : q
      );

      setQueries(updatedQueriesWithResponse);

      const updatedConversations = conversations.map((conv) =>
        conv.id === currentConversationId
          ? { ...conv, queries: updatedQueriesWithResponse }
          : conv
      );

      setConversations(updatedConversations);
      saveToLocalStorage(updatedConversations);
    } catch (error) {
      console.error("Error fetching response:", error);
      setQueries(
        updatedQueries.map((q, index) =>
          index === updatedQueries.length - 1
            ? { ...q, response: "Error fetching response", isLoading: false }
            : q
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const renderWelcomeOverlay = () => (
    <div className="welcome-overlay">
      <div className="welcome-content">
        <div className="welcome-header">
          <Sparkles className="welcome-icon" size={32} />
          <h2>Welcome to MemoryVault</h2>
        </div>
        <p>Begin your journey down memory lane with these prompts:</p>
        <div className="suggested-prompts">
          {suggestedPrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => handlePromptClick(prompt)}
              className="prompt-btn"
            >
              <MessageSquare size={16} />
              {prompt}
            </button>
          ))}
        </div>
        <button onClick={() => setShowWelcome(false)} className="close-welcome-btn">
          <X size={16} />
        </button>
      </div>
    </div>
  );

  const renderEmptyChatState = () => (
    <div className="empty-chat-state">
      <div className="memory-orb">
        <div className="orb-glow"></div>
        <Sparkles className="orb-sparkle" size={24} />
      </div>
      <h3 className="empty-title">Your Memory Vault Awaits</h3>
      <p className="empty-description">
        Ask questions about your memories and I'll help you recall those special moments
      </p>
      <div className="memory-particles">
        {[...Array(6)].map((_, i) => (
          <div key={i} className={`particle particle-${i + 1}`}>
            <Sparkles size={i % 2 === 0 ? 12 : 8} />
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="memory-vault">
      {showWelcome && renderWelcomeOverlay()}
      <div className="sidebar">
        <button onClick={() => startNewConversation()} className="new-conversation-btn">
          <PlusCircle size={18} />
          <span>New Memory Journey</span>
        </button>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <div key={conv.id} className="conversation-item">
              <button
                onClick={() => selectConversation(conv.id)}
                className={`conversation-btn ${
                  currentConversationId === conv.id ? "active" : ""
                }`}
              >
                <Clock size={14} />
                <span>{new Date(parseInt(conv.id)).toLocaleString()}</span>
              </button>
              <button
                onClick={(e) => deleteConversation(conv.id, e)}
                className="delete-btn"
                aria-label="Delete conversation"
              >
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="main-content">
        <div className="conversation">
          {queries.length === 0 ? (
            renderEmptyChatState()
          ) : (
            queries.map((item, index) => (
              <div key={index} className="query-response">
                <div className="query-bubble">
                  <MessageSquare size={16} />
                  <p>{item.query}</p>
                </div>
                {item.isLoading ? (
                  <div className="loading-animation">
                    <div className="loading-dots">
                      <div className="dot dot1"></div>
                      <div className="dot dot2"></div>
                      <div className="dot dot3"></div>
                    </div>
                  </div>
                ) : (
                  <div className="response-container">
                    {item.image && (
                      <div className="image-container">
                        <img
                          src={item.image}
                          alt="Generated memory visualization"
                          className="response-image"
                        />
                      </div>
                    )}
                    <div className="response-bubble">
                      <Sparkles size={16} />
                      <p>{typeof item.response === "object"
                        ? JSON.stringify(item.response)
                        : item.response}</p>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={conversationEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="query-form">
          <input
            type="text"
            value={currentQuery}
            onChange={(e) => setCurrentQuery(e.target.value)}
            placeholder="Ask about your memories..."
            className="query-input"
            disabled={isLoading}
          />
          <button type="submit" className="submit-button" disabled={isLoading}>
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
};

export default QueryPage;