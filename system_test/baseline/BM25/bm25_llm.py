"""
BM25 RAG with LLM Answer Generation using OpenAI GPT-4o-mini
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re
from pathlib import Path
import os
from openai import OpenAI


ANSWER_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời câu hỏi cho sinh viên dựa trên **quy chế UIT**.

MỤC TIÊU:
- Trả lời rõ ràng, thân thiện, dễ hiểu.
- Ưu tiên trả lời thẳng vào ý chính trong 1–2 câu đầu, sau đó có thể giải thích thêm.
- Dùng ngôi xưng phù hợp với ngữ cảnh (có thể dùng "em", "bạn" theo câu hỏi).

QUAN TRỌNG CHO NEAR_RULE (câu hỏi dùng ngôn ngữ đời thường):
- Câu hỏi có thể dùng từ ngữ thân mật như:
  "rớt môn", "bị sao", "có ảnh hưởng gì", "nặng không",…
- BẮT BUỘC phải cố gắng map ngôn ngữ đời thường sang các khái niệm quy chế chính thức:
  * "rớt môn" / "rớt nhiều môn" → nguy cơ bị cảnh báo học vụ, ảnh hưởng đến ĐTBHK, tín chỉ tích lũy.
  * "bị cảnh báo" → tình trạng học vụ, hạn chế, trách nhiệm của sinh viên.
  * "điểm rèn luyện" → điều kiện tốt nghiệp, mức điểm tối thiểu.
- Nếu ngữ cảnh có chứa quy định liên quan (ví dụ: quy định về cảnh báo học vụ, điều kiện về điểm, số tín chỉ…), 
  BẮT BUỘC phải trả lời dựa trên đó:
  - Giải thích mối liên hệ: rớt nhiều môn → ĐTBHK giảm → có thể chạm ngưỡng cảnh báo.
- TRÁNH các câu trả lời làm người hỏi thấy bị "đuổi khéo":
  - Hạn chế dùng "Thông tin bạn cung cấp không đề cập đến..." với giọng tiêu cực.
  - Chỉ nên nói không đủ thông tin khi thật sự không có đoạn trích liên quan nào trong ngữ cảnh.

GỢI Ý CÁCH DIỄN ĐẠT KHI THIẾU NGỮ CẢNH:
- Nếu ngữ cảnh thực sự không đủ hoặc không có quy định liên quan:
  - Giải thích nhẹ nhàng, trung tính, ví dụ:
    "Trong các đoạn trích hiện tại, tôi chưa thấy quy định nêu rõ trường hợp này. 
     Em nên xem thêm toàn văn quy chế hoặc liên hệ Phòng Đào tạo/cố vấn học tập để được hướng dẫn cụ thể hơn."
- Tuyệt đối không bịa số liệu, điều kiện, hoặc trích dẫn điều/khoản không có trong ngữ cảnh.

QUY TẮC CHUNG:
- Chỉ sử dụng thông tin trong ngữ cảnh (các đoạn trích).
- Nếu có thể, nêu rõ điều/khoản hoặc tên mục liên quan.
- Trả lời ngắn gọn, mạch lạc, bằng tiếng Việt, giọng thân thiện và tôn trọng sinh viên.
- Thêm format markdown (danh sách, in đậm, in nghiêng, xuống dòng, v.v.) sao cho dễ đọc nhất.
""".strip()


class BM25RAG_LLM:
    def __init__(self, db_path: str, openai_api_key: str = None):
        """Initialize BM25 RAG system with LLM answer generation"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.documents = []
        self.doc_ids = []
        self.bm25 = None
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
    def load_documents(self):
        """Load documents from database"""
        query = """
        SELECT 
            i.id,
            i.doc_id,
            i.title,
            i.heading,
            i.content,
            i.level,
            i.path,
            d.title as doc_title,
            d.so_hieu
        FROM items i
        JOIN documents d ON i.doc_id = d.id
        WHERE i.content IS NOT NULL AND i.content != ''
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        # Create combined text for each document
        for _, row in df.iterrows():
            # Combine all text fields
            text_parts = []
            
            if pd.notna(row['doc_title']):
                text_parts.append(f"Văn bản: {row['doc_title']}")
            if pd.notna(row['so_hieu']):
                text_parts.append(f"Số hiệu: {row['so_hieu']}")
            if pd.notna(row['title']):
                text_parts.append(f"Tiêu đề: {row['title']}")
            if pd.notna(row['heading']):
                text_parts.append(f"Đề mục: {row['heading']}")
            if pd.notna(row['content']):
                text_parts.append(row['content'])
            
            combined_text = " ".join(text_parts)
            
            self.documents.append({
                'id': row['id'],
                'doc_id': row['doc_id'],
                'text': combined_text,
                'title': row['title'],
                'heading': row['heading'],
                'content': row['content'],
                'doc_title': row['doc_title'],
                'so_hieu': row['so_hieu'],
                'level': row['level'],
                'path': row['path']
            })
            self.doc_ids.append(row['id'])
        
        print(f"Loaded {len(self.documents)} documents")
        
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation"""
        if not text:
            return []
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def build_index(self):
        """Build BM25 index"""
        tokenized_corpus = [self.tokenize(doc['text']) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant documents for a query"""
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(scores[idx])
            results.append(doc)
        
        return results
    
    def format_context(self, retrieved_docs: List[Dict], top_k: int = 5) -> str:
        """Format retrieved documents as context for LLM"""
        if not retrieved_docs:
            return "Không tìm thấy thông tin liên quan."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:top_k], 1):
            part = f"--- Đoạn trích {i} ---\n"
            if doc.get('doc_title'):
                part += f"Văn bản: {doc['doc_title']}\n"
            if doc.get('so_hieu'):
                part += f"Số hiệu: {doc['so_hieu']}\n"
            if doc.get('title'):
                part += f"Điều/Khoản: {doc['title']}\n"
            if doc.get('heading'):
                part += f"Đề mục: {doc['heading']}\n"
            if doc.get('content'):
                part += f"Nội dung: {doc['content']}\n"
            
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def generate_answer_with_llm(self, question: str, context: str) -> str:
        """Generate answer using LLM with context"""
        user_prompt = f"Câu hỏi: {question}\n\nNgữ cảnh:\n{context}"
        
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        
        return response.choices[0].message.content
    
    def answer_question(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """End-to-end QA: retrieve and generate answer with LLM"""
        retrieved_docs = self.retrieve(question, top_k)
        context = self.format_context(retrieved_docs, top_k)
        answer = self.generate_answer_with_llm(question, context)
        return answer, retrieved_docs
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main function to demonstrate BM25 RAG system with LLM"""
    import sys
    
    # Database path
    db_path = r"E:\Github\uit_chatbot\normailizer\uit_law_points.db"
    
    print("Initializing BM25 RAG system with LLM...")
    rag = BM25RAG_LLM(db_path)
    
    print("Loading documents from database...")
    rag.load_documents()
    
    print("Building BM25 index...")
    rag.build_index()
    
    # Test with a sample question
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Người học bồi dưỡng nâng cao trình độ học vấn, nghề nghiệp thì có được cấp chứng chỉ không?"
    
    print(f"\nQuestion: {question}")
    print("\nRetrieving relevant documents and generating answer...")
    
    answer, docs = rag.answer_question(question, top_k=5)
    
    print("\n" + "="*80)
    print("ANSWER (Generated by LLM):")
    print("="*80)
    print(answer)
    
    print("\n" + "="*80)
    print("TOP RETRIEVED DOCUMENTS:")
    print("="*80)
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Score: {doc['score']:.4f}")
        print(f"   Title: {doc.get('title', 'N/A')}")
        print(f"   Doc: {doc.get('doc_title', 'N/A')} ({doc.get('so_hieu', 'N/A')})")
        print(f"   Content preview: {doc.get('content', '')[:200]}...")
    
    rag.close()


if __name__ == "__main__":
    main()
