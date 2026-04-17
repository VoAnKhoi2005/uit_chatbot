import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chat, getPdfUrl } from "./api";
import About from "./About";
import PdfViewer from "./PdfViewer";
import { db, Conversation, ChatMessage } from "./db";

type Message = {
  role: "user" | "bot";
  text: string;
  timestamp: string;
  sources?: { article_id?: string; clause_id?: string; text: string; doc_id?: string; title?: string; doc_title?: string; so_hieu?: string }[];
  questionType?: string;
};

type Theme = "light" | "dark" | "system";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"connected" | "error">("connected");
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const [pdfViewerUrl, setPdfViewerUrl] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("theme") as Theme;
    return saved || "system";
  });
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [selectedMessageSources, setSelectedMessageSources] = useState<number | null>(null);
  const [editingConversationId, setEditingConversationId] = useState<number | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const chatRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Initialize database
  useEffect(() => {
    db.init().then(() => {
      loadConversations();
    });
  }, []);

  const loadConversations = async () => {
    const convs = await db.getAllConversations();
    setConversations(convs);
  };

  const loadConversationMessages = async (conversationId: number) => {
    const msgs = await db.getMessagesByConversation(conversationId);
    const formattedMessages: Message[] = msgs.map((msg) => ({
      role: msg.role,
      text: msg.text,
      timestamp: msg.timestamp,
      sources: msg.sources?.map(s => ({
        article_id: s.article_id,
        clause_id: s.clause_id,
        text: s.text,
        doc_id: s.doc_id,
        title: s.title,
        doc_title: s.doc_title,
        so_hieu: s.so_hieu,
      })),
      questionType: msg.questionType,
    }));
    setMessages(formattedMessages);
    setCurrentConversationId(conversationId);
  };

  const createNewConversation = async () => {
    const title = "New Conversation";
    const conversationId = await db.createConversation(title);
    await loadConversations();
    setCurrentConversationId(conversationId);
    setMessages([]);
  };

  const deleteConversation = async (conversationId: number) => {
    await db.deleteConversation(conversationId);
    await loadConversations();
    if (currentConversationId === conversationId) {
      setCurrentConversationId(null);
      setMessages([]);
    }
  };

  const renameConversation = async (conversationId: number, newTitle: string) => {
    await db.updateConversation(conversationId, { title: newTitle });
    await loadConversations();
  };

  const startEditingConversation = (conversationId: number, currentTitle: string) => {
    setEditingConversationId(conversationId);
    setEditingTitle(currentTitle);
  };

  const saveConversationEdit = async () => {
    if (editingConversationId && editingTitle.trim()) {
      await renameConversation(editingConversationId, editingTitle.trim());
      setEditingConversationId(null);
      setEditingTitle("");
    }
  };

  const cancelConversationEdit = () => {
    setEditingConversationId(null);
    setEditingTitle("");
  };

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;
    
    if (theme === "system") {
      const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.setAttribute("data-theme", isDark ? "dark" : "light");
    } else {
      root.setAttribute("data-theme", theme);
    }
    
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Listen for system theme changes
  useEffect(() => {
    if (theme !== "system") return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (e: MediaQueryListEvent) => {
      document.documentElement.setAttribute("data-theme", e.matches ? "dark" : "light");
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [theme]);

  const cycleTheme = () => {
    setTheme((current) => {
      if (current === "system") return "light";
      if (current === "light") return "dark";
      return "system";
    });
  };

  // Get last bot message sources and question type for side panel
  const displayedSources = useMemo(() => {
    if (selectedMessageSources !== null) {
      const message = messages[selectedMessageSources];
      return message?.sources || [];
    }
    
    // Default to last bot message
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "bot") {
        return messages[i].sources || [];
      }
    }
    return [];
  }, [messages, selectedMessageSources]);

  const displayedQuestionType = useMemo(() => {
    if (selectedMessageSources !== null) {
      return messages[selectedMessageSources]?.questionType;
    }
    
    // Default to last bot message
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "bot") {
        return messages[i].questionType;
      }
    }
    return undefined;
  }, [messages, selectedMessageSources]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 3 * 24; // ~3 lines
      textareaRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
    }
  }, [input]);

  // Scroll chat to bottom on new message
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    const timestamp = new Date().toLocaleTimeString();
    const userMessage: Message = { role: "user", text: question, timestamp };
    
    // Show message immediately for better UX
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setError(null);
    setLoading(true);
    setApiStatus("connected");

    try {
      // Auto-create conversation if none exists
      let conversationId = currentConversationId;
      let isNewConversation = false;
      if (!conversationId) {
        conversationId = await db.createConversation("New Conversation");
        setCurrentConversationId(conversationId);
        isNewConversation = true;
      }
      
      // Check if this is the first message in the conversation and update title
      const existingMessages = await db.getMessagesByConversation(conversationId);
      if (existingMessages.length === 0) {
        const shortTitle = question.length > 50 ? question.slice(0, 50) + "..." : question;
        await db.updateConversation(conversationId, { title: shortTitle });
        if (isNewConversation) {
          await loadConversations();
        }
      }

      // Save user message to DB
      await db.addMessage({
        conversationId,
        role: "user",
        text: question,
        timestamp,
      });

      // Build conversation history from current messages
      const conversationHistory = messages.map((m) => ({
        role: m.role === "user" ? "user" : "assistant",
        content: m.text,
      }));
      conversationHistory.push({ role: "user", content: question });
      
      const res = await chat(question, conversationHistory);
      const botTimestamp = new Date().toLocaleTimeString();
      const botMessage: Message = {
        role: "bot",
        text: res.answer,
        timestamp: botTimestamp,
        sources: res.sources || [],
        questionType: res.question_type,
      };
      
      setMessages((prev) => [...prev, botMessage]);
      setApiStatus("connected");

      // Save bot message to DB
      await db.addMessage({
        conversationId,
        role: "bot",
        text: res.answer,
        timestamp: botTimestamp,
        sources: res.sources,
        questionType: res.question_type,
      });

      // Reload conversations to reflect updated title
      await loadConversations();
    } catch (err: any) {
      console.error("Chat error:", err);
      setError(err?.message || "Có lỗi xảy ra");
      setApiStatus("error");
      
      // Remove the user message that failed
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const isSendDisabled = useMemo(
    () => loading || input.trim().length === 0,
    [loading, input]
  );

  const MAX_SOURCE_LENGTH = 200;

  const toggleSourceExpand = (index: number) => {
    setExpandedSources((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  return (
    <div className="app-root">
      {/* Left Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">UIT Chatbot</h1>
          <p className="sidebar-subtitle">Regulations assistant</p>
        </div>
        
        <div className="conversations-section">
          <div className="conversations-header">
            <h3>Conversations</h3>
            <button className="new-conversation-btn" onClick={createNewConversation} title="Start new conversation">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
            </button>
          </div>
          <div className="conversations-list">
            {conversations.length === 0 ? (
              <div className="conversations-empty">
                <p>No conversations yet.</p>
                <p>Start typing to begin!</p>
              </div>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`conversation-item ${conv.id === currentConversationId ? "active" : ""}`}
                  onClick={() => editingConversationId !== conv.id && loadConversationMessages(conv.id!)}
                >
                  {editingConversationId === conv.id ? (
                    <input
                      type="text"
                      className="conversation-title-input"
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          saveConversationEdit();
                        } else if (e.key === "Escape") {
                          cancelConversationEdit();
                        }
                      }}
                      onBlur={saveConversationEdit}
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <div className="conversation-title">{conv.title}</div>
                  )}
                  <div className="conversation-actions">
                    {editingConversationId !== conv.id && (
                      <button
                        className="rename-conversation-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          startEditingConversation(conv.id!, conv.title);
                        }}
                        title="Rename conversation"
                      >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                      </button>
                    )}
                    <button
                      className="delete-conversation-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm("Delete this conversation?")) {
                          deleteConversation(conv.id!);
                        }
                      }}
                      title="Delete conversation"
                    >
                      ×
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
        
        <div className="sidebar-footer">
          <button className="theme-toggle" onClick={cycleTheme} title="Toggle theme">
            {theme === "system" && (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="9" y1="9" x2="15" y2="15"></line>
              </svg>
            )}
            {theme === "light" && (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
              </svg>
            )}
            {theme === "dark" && (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
              </svg>
            )}
            <span className="theme-label">
              {theme === "system" ? "Auto" : theme === "light" ? "Light" : "Dark"}
            </span>
          </button>
          <button className="about-button" onClick={() => setShowAbout(true)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span>About</span>
          </button>
          <p>Powered by GPT + Ontology</p>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Top Bar */}
        <header className="topbar">
          <h2 className="topbar-title">UIT Regulations Chat</h2>
          <div className={`status-pill ${apiStatus}`}>
            {apiStatus === "connected" ? "Connected to API" : "Error"}
          </div>
        </header>

        {/* Error Banner */}
        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="error-dismiss">×</button>
          </div>
        )}

        {/* Main Panel: Chat + Sources */}
        <div className="main-panel">
          {/* Chat Area */}
          <div className="chat-area">
            <div className="chat-messages" ref={chatRef}>
              {messages.length === 0 && !loading && (
                <div className="empty-state">
                  <div className="welcome-icon">💬</div>
                  <h2>Welcome to UIT Chatbot!</h2>
                  <p>Ask me anything about UIT regulations and policies.</p>
                  <div className="example-questions">
                    <p className="example-label">Try asking:</p>
                    <div 
                      className="example-chip" 
                      onClick={() => setInput("Điều kiện tốt nghiệp là gì?")}
                    >
                      "Điều kiện tốt nghiệp là gì?"
                    </div>
                  </div>
                </div>
              )}
              {messages.map((m, idx) => (
                <div key={idx} className={`message ${m.role}`}>
                  <div className="message-bubble">
                    <div className="message-text">
                      {m.role === "bot" ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {m.text}
                        </ReactMarkdown>
                      ) : (
                        m.text
                      )}
                    </div>
                    <div className="message-footer">
                      <div className="message-timestamp">{m.timestamp}</div>
                      {m.role === "bot" && m.sources && m.sources.length > 0 && (
                        <button 
                          className={`view-sources-btn ${selectedMessageSources === idx ? 'active' : ''}`}
                          onClick={() => setSelectedMessageSources(selectedMessageSources === idx ? null : idx)}
                          title="View sources"
                        >
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <polyline points="10 9 9 9 8 9"></polyline>
                          </svg>
                          {m.sources.length} {m.sources.length === 1 ? 'source' : 'sources'}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message bot">
                  <div className="message-bubble typing">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <form className="chat-input-form" onSubmit={handleSubmit}>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={currentConversationId ? "Enter a question about regulations..." : "Enter a question to start the conversation..."}
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              <button type="submit" disabled={isSendDisabled} className="send-button">
                {currentConversationId ? "Send" : "Start Conversation"}
              </button>
            </form>
          </div>

          {/* Sources Panel */}
          <aside className="sources-panel">
            <div className="sources-header">
              <h3 className="sources-title">Sources</h3>
              {selectedMessageSources !== null && (
                <button 
                  className="clear-selection-btn"
                  onClick={() => setSelectedMessageSources(null)}
                  title="Back to latest"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="15 18 9 12 15 6"></polyline>
                  </svg>
                  Latest
                </button>
              )}
            </div>
            <div className="sources-content">
              {displayedSources.length > 0 ? (
                <div className="sources-list">
                  {displayedSources.map((s, i) => {
                    const sourceText = s.text || "";
                    const isLong = sourceText.length > MAX_SOURCE_LENGTH;
                    const isExpanded = expandedSources.has(i);
                    const displayText = isLong && !isExpanded 
                      ? sourceText.slice(0, MAX_SOURCE_LENGTH) + "..."
                      : sourceText;
                    const hasDocId = s.doc_id && s.doc_id.trim().length > 0;

                    return (
                      <div key={i} className="source-card">
                        <div 
                          className={`source-header ${hasDocId ? 'clickable' : ''}`}
                          onClick={() => hasDocId && setPdfViewerUrl(getPdfUrl(s.doc_id!))}
                          title={hasDocId ? 'Click to view PDF' : ''}
                        >
                          {s.article_id && (
                            <span className="source-id">
                              {[
                                s.title,
                                s.doc_title,
                                s.so_hieu && `Số hiệu: ${s.so_hieu}`
                              ].filter(Boolean).join(" – ")}
                            </span>
                          )}
                          {hasDocId && (
                            <svg 
                              className="pdf-icon" 
                              viewBox="0 0 24 24" 
                              fill="none" 
                              stroke="currentColor" 
                              strokeWidth="2"
                            >
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                              <polyline points="14 2 14 8 20 8"></polyline>
                            </svg>
                          )}
                        </div>
                        <p className="source-text">{displayText}</p>
                        {isLong && (
                          <button 
                            className="expand-button"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleSourceExpand(i);
                            }}
                          >
                            {isExpanded ? "Show less" : "Show more"}
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="sources-empty">
                  <p>No sources yet. Ask a question about UIT regulations to see citations here.</p>
                </div>
              )}
            </div>
            {/* Debug Box */}
            {displayedQuestionType && (
              <div className="debug-box">
                <div className="debug-label">Question Type</div>
                <div className={`question-type-pill ${displayedQuestionType.toLowerCase()}`}>
                  {displayedQuestionType}
                </div>
              </div>
            )}
          </aside>
        </div>
      </div>

      {/* About Modal */}
      {showAbout && (
        <div className="modal-backdrop" onClick={() => setShowAbout(false)}>
          <div className="modal-container" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>About UIT Chatbot</h2>
              <button className="modal-close" onClick={() => setShowAbout(false)}>×</button>
            </div>
            <div className="modal-content">
              <About />
            </div>
          </div>
        </div>
      )}

      {/* PDF Viewer Modal */}
      {pdfViewerUrl && (
        <PdfViewer pdfUrl={pdfViewerUrl} onClose={() => setPdfViewerUrl(null)} />
      )}
    </div>
  );
}

