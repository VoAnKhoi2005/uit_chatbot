import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chat, openPdfInNewTab } from "./api";
import About from "./About";

type Message = {
  role: "user" | "bot";
  text: string;
  timestamp: string;
  sources?: { article_id?: string; clause_id?: string; text: string; doc_id?: string; title?: string; doc_title?: string; so_hieu?: string }[];
  questionType?: string;
};

type Page = "chat" | "about";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"connected" | "error">("connected");
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const [currentPage, setCurrentPage] = useState<Page>("chat");
  const chatRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Get last bot message sources and question type for side panel
  const lastBotMessage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "bot") {
        return messages[i];
      }
    }
    return null;
  }, [messages]);

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
    setMessages((prev) => [...prev, { role: "user", text: question, timestamp }]);
    setInput("");
    setError(null);
    setLoading(true);
    setApiStatus("connected");
    try {
      // Conversation history disabled - sending empty array
      const conversationHistory: any[] = [];
      
      const res = await chat(question, conversationHistory);
      const botTimestamp = new Date().toLocaleTimeString();
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: res.answer,
          timestamp: botTimestamp,
          sources: res.sources || [],
          questionType: res.question_type,
        },
      ]);
      setApiStatus("connected");
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Có lỗi xảy ra");
      setApiStatus("error");
      const botTimestamp = new Date().toLocaleTimeString();
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: "Xin lỗi, hệ thống đang gặp sự cố.",
          timestamp: botTimestamp,
        },
      ]);
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
        <nav className="sidebar-nav">
          <div className="nav-item active">Chat</div>
          <div className="nav-item">About</div>
        </nav>
        <div className="sidebar-footer">
          <p>Powered by GPT + Ontology</p>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Top Bar */}
        <header className="topbar">
          <h2 className="topbar-title">
            {currentPage === "chat" ? "UIT Regulations Chat" : "About UIT Chatbot"}
          </h2>
          {currentPage === "chat" && (
            <div className={`status-pill ${apiStatus}`}>
              {apiStatus === "connected" ? "Connected to API" : "Error"}
            </div>
          )}
        </header>

        {/* Error Banner */}
        {error && currentPage === "chat" && (
          <div className="error-banner">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="error-dismiss">×</button>
          </div>
        )}

        {/* Page Content */}
        {currentPage === "about" ? (
          <About />
        ) : (
          /* Main Panel: Chat + Sources */
          <div className="main-panel">
          {/* Chat Area */}
          <div className="chat-area">
            <div className="chat-messages" ref={chatRef}>
              {messages.length === 0 && !loading && (
                <div className="empty-state">
                  <p>Chào mừng đến với UIT Chatbot!</p>
                  <p>Hãy đặt câu hỏi về quy chế đào tạo của trường.</p>
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
                    <div className="message-timestamp">{m.timestamp}</div>
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
                placeholder="Nhập câu hỏi về quy chế..."
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              <button type="submit" disabled={isSendDisabled} className="send-button">
                Gửi
              </button>
            </form>
          </div>

          {/* Sources Panel */}
          <aside className="sources-panel">
            <h3 className="sources-title">Sources</h3>
            <div className="sources-content">
              {lastBotMessage?.sources && lastBotMessage.sources.length > 0 ? (
                <div className="sources-list">
                  {lastBotMessage.sources.map((s, i) => {
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
                          onClick={() => hasDocId && openPdfInNewTab(s.doc_id!)}
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
            {lastBotMessage?.questionType && (
              <div className="debug-box">
                <div className="debug-label">Question Type</div>
                <div className={`question-type-pill ${lastBotMessage.questionType.toLowerCase()}`}>
                  {lastBotMessage.questionType}
                </div>
              </div>
            )}
          </aside>
        </div>
        )}
      </div>
    </div>
  );
}

