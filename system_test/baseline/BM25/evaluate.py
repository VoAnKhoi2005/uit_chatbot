"""
Evaluate BM25 RAG system on test questions
"""
import pandas as pd
import sqlite3
from bm25_rag import BM25RAG
from typing import List, Dict
import json
from tqdm import tqdm
from pathlib import Path


def load_test_questions(csv_path: str) -> pd.DataFrame:
    """Load test questions from CSV"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"Loaded {len(df)} test questions")
    return df


def evaluate_bm25(
    test_df: pd.DataFrame,
    db_path: str,
    top_k: int = 5,
    output_path: str = None
) -> pd.DataFrame:
    """Evaluate BM25 RAG on test questions"""
    
    print("Initializing BM25 RAG system...")
    rag = BM25RAG(db_path)
    
    print("Loading documents...")
    rag.load_documents()
    
    print("Building BM25 index...")
    rag.build_index()
    
    results = []
    
    print(f"\nProcessing {len(test_df)} questions...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row['question']
        
        # Get answer and retrieved documents
        answer, retrieved_docs = rag.answer_question(question, top_k=top_k)
        
        # Extract top document IDs and scores
        doc_ids = [doc['id'] for doc in retrieved_docs]
        scores = [doc['score'] for doc in retrieved_docs]
        
        result = {
            'question': question,
            'extractive_answer': row.get('extractive_answer', ''),
            'abstractive_answer': row.get('abstractive_answer', ''),
            'predicted_answer': answer,
            'question_type': row.get('question_type', ''),
            'retrieved_doc_ids': json.dumps(doc_ids),
            'retrieval_scores': json.dumps(scores),
            'top_score': scores[0] if scores else 0.0,
            'num_retrieved': len(retrieved_docs)
        }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if output_path:
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_path}")
    
    rag.close()
    
    return results_df


def print_statistics(results_df: pd.DataFrame):
    """Print evaluation statistics"""
    print("\n" + "="*80)
    print("EVALUATION STATISTICS")
    print("="*80)
    
    print(f"\nTotal questions: {len(results_df)}")
    
    # Statistics by question type
    if 'question_type' in results_df.columns:
        print("\nBy Question Type:")
        type_counts = results_df['question_type'].value_counts()
        for qtype, count in type_counts.items():
            print(f"  {qtype}: {count}")
    
    # Retrieval statistics
    print(f"\nRetrieval Statistics:")
    print(f"  Average top score: {results_df['top_score'].mean():.4f}")
    print(f"  Max top score: {results_df['top_score'].max():.4f}")
    print(f"  Min top score: {results_df['top_score'].min():.4f}")
    print(f"  Average documents retrieved: {results_df['num_retrieved'].mean():.2f}")
    
    # Sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 3)")
    print("="*80)
    
    for idx in range(min(3, len(results_df))):
        row = results_df.iloc[idx]
        print(f"\n{idx+1}. Question: {row['question']}")
        print(f"   Type: {row.get('question_type', 'N/A')}")
        print(f"   Top Score: {row['top_score']:.4f}")
        print(f"   Predicted Answer Preview: {row['predicted_answer'][:200]}...")
        print()


def main():
    """Main evaluation function"""
    
    # Paths
    db_path = r"E:\Github\uit_chatbot\normailizer\uit_law_points.db"
    test_csv_path = r"E:\Github\uit_chatbot\system_test\test_results.csv"
    output_path = r"E:\Github\uit_chatbot\system_test\baseline\BM25\bm25_results.csv"
    
    # Load test data
    test_df = load_test_questions(test_csv_path)
    
    # Run evaluation
    results_df = evaluate_bm25(
        test_df=test_df,
        db_path=db_path,
        top_k=5,
        output_path=output_path
    )
    
    # Print statistics
    print_statistics(results_df)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
