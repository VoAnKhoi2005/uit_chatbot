export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "https://api.uitchatbot.io.vn";

export async function chat(question: string, conversationHistory?: Array<{role: string; content: string}>) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      question,
      conversation_history: conversationHistory || []
    }),
  });
  if (!res.ok) {
    throw new Error(`Chat API error: ${res.status}`);
  }
  return res.json() as Promise<{
    answer: string;
    question_type: string;
    sources: { article_id?: string; title?: string; clause_id?: string; text: string; doc_id?: string ; doc_title?: string; so_hieu?: string }[];
  }>;
}

export function getPdfUrl(docId: string): string {
  return `${API_BASE_URL}/pdf/${docId}`;
}

export function openPdfInNewTab(docId: string) {
  window.open(getPdfUrl(docId), '_blank');
}

