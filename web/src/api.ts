export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:10000";

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
    sources: { article_id?: string; clause_id?: string; text: string }[];
  }>;
}

