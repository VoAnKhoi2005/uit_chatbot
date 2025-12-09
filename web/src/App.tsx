import { FormEvent, useMemo, useRef, useState } from "react";
import { chat } from "./api";

type Message = {
  role: "user" | "bot";
  text: string;
  sources?: { article_id?: string; clause_id?: string; text: string }[];
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;
    const question = input.trim();
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setError(null);
    setLoading(true);
    try {
      const res = await chat(question);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: res.answer, sources: res.sources || [] },
      ]);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Có lỗi xảy ra");
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Xin lỗi, hệ thống đang gặp sự cố." },
      ]);
    } finally {
      setLoading(false);
      if (listRef.current) {
        listRef.current.scrollTop = listRef.current.scrollHeight;
      }
    }
  };

  const isSendDisabled = useMemo(
    () => loading || input.trim().length === 0,
    [loading, input]
  );

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>UIT Chatbot</h1>
          <p>Hỏi đáp về quy chế — dùng /chat backend</p>
        </div>
        <div className="status">
          API: <code>{import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}</code>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages" ref={listRef}>
          {messages.map((m, idx) => (
            <div key={idx} className={`message ${m.role}`}>
              <div className="bubble">
                <div className="text">{m.text}</div>
                {m.role === "bot" && m.sources && m.sources.length > 0 && (
                  <div className="sources">
                    <div className="sources-title">Nguồn:</div>
                    <ul>
                      {m.sources.map((s, i) => (
                        <li key={i}>
                          <strong>
                            {s.article_id ? `Điều ${s.article_id}` : "Điều ?"}
                            {s.clause_id ? `, Khoản ${s.clause_id}` : ""}
                          </strong>
                          : {s.text}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message bot">
              <div className="bubble">Đang trả lời…</div>
            </div>
          )}
        </div>
        {error && <div className="error">{error}</div>}
      </main>

      <form className="input-bar" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nhập câu hỏi..."
          rows={2}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button type="submit" disabled={isSendDisabled}>
          Gửi
        </button>
      </form>
    </div>
  );
}

