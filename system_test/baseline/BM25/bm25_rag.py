"""
BM25 RAG QA System for UIT Law Points Database
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re
from pathlib import Path


class BM25RAG:
    def __init__(self, db_path: str):
        """Initialize BM25 RAG system with database path"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.documents = []
        self.doc_ids = []
        self.bm25 = None
        
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
        # Convert to lowercase and split on whitespace/punctuation
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
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate answer from retrieved documents (simple concatenation for baseline)"""
        if not retrieved_docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        
        # Build answer from top retrieved documents
        answer_parts = []
        
        for i, doc in enumerate(retrieved_docs[:3], 1):
            part = f"\n**Tài liệu {i}:**\n"
            if doc.get('doc_title'):
                part += f"- Văn bản: {doc['doc_title']}\n"
            if doc.get('so_hieu'):
                part += f"- Số hiệu: {doc['so_hieu']}\n"
            if doc.get('title'):
                part += f"- Mục: {doc['title']}\n"
            if doc.get('content'):
                content = doc['content'][:500]  # Limit length
                part += f"- Nội dung: {content}...\n"
            
            answer_parts.append(part)
        
        answer = "\n".join(answer_parts)
        return answer
    
    def answer_question(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """End-to-end QA: retrieve and generate answer"""
        retrieved_docs = self.retrieve(question, top_k)
        answer = self.generate_answer(question, retrieved_docs)
        return answer, retrieved_docs
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main function to demonstrate BM25 RAG system"""
    import sys
    
    # Database path
    db_path = r"E:\Github\uit_chatbot\normailizer\uit_law_points.db"
    
    print("Initializing BM25 RAG system...")
    rag = BM25RAG(db_path)
    
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
    print("\nRetrieving relevant documents...")
    
    answer, docs = rag.answer_question(question, top_k=5)
    
    print("\n" + "="*80)
    print("ANSWER:")
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
