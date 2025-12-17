export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "https://api.uitchatbot.io.vn";

export async function chat(question: string, conversationHistory?: Array<{role: string; content: string}>) {
  try {
    const payload = { 
      question,
      conversation_history: conversationHistory || []
    };
    
    console.log("[API] Sending chat request:", payload);
    
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      console.error("[API] Error response:", errorText);
      throw new Error(`Chat API error: ${res.status}`);
    }
    
    return res.json() as Promise<{
      answer: string;
      question_type: string;
      sources: { article_id?: string; title?: string; clause_id?: string; text: string; doc_id?: string ; doc_title?: string; so_hieu?: string }[];
    }>;
  } catch (error) {
    console.error("[API] Fetch error:", error);
    throw error;
  }
}

export function getPdfUrl(docId: string): string {
  return `${API_BASE_URL}/document/${docId}`;
}

export function openPdfInNewTab(docId: string) {
  window.open(getPdfUrl(docId), '_blank');
}

