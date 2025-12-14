"""
Create and manage OpenAI Batch API jobs for BM25 RAG evaluation
"""
import pandas as pd
import json
import os
from pathlib import Path
from openai import OpenAI
import time
from typing import List, Dict
from bm25_llm import BM25RAG_LLM, ANSWER_SYSTEM_PROMPT


def create_batch_input_file(
    test_csv_path: str,
    db_path: str,
    output_jsonl_path: str,
    top_k: int = 5
) -> int:
    """
    Create JSONL file for OpenAI Batch API
    
    Returns:
        Number of requests created
    """
    print("Initializing BM25 RAG for retrieval...")
    rag = BM25RAG_LLM(db_path)
    rag.load_documents()
    rag.build_index()
    
    print(f"Loading test questions from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path, encoding='utf-8-sig')
    
    batch_requests = []
    
    print(f"\nPreparing {len(test_df)} batch requests...")
    for idx, row in test_df.iterrows():
        question = row['question']
        
        # Retrieve documents
        retrieved_docs = rag.retrieve(question, top_k=top_k)
        context = rag.format_context(retrieved_docs, top_k=top_k)
        
        # Create batch request
        user_prompt = f"Câu hỏi: {question}\n\nNgữ cảnh:\n{context}"
        
        batch_request = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                "messages": [
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            }
        }
        
        batch_requests.append(batch_request)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} questions...")
    
    # Write to JSONL file
    print(f"\nWriting batch requests to {output_jsonl_path}...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    rag.close()
    
    print(f"✓ Created {len(batch_requests)} batch requests")
    return len(batch_requests)


def submit_batch_job(
    input_jsonl_path: str,
    description: str = "BM25 RAG Evaluation"
) -> str:
    """
    Submit batch job to OpenAI
    
    Returns:
        Batch job ID
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print(f"\nUploading batch file: {input_jsonl_path}...")
    
    # Upload file
    with open(input_jsonl_path, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    print(f"✓ File uploaded: {batch_file.id}")
    
    # Create batch job
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )
    
    print(f"✓ Batch job created: {batch.id}")
    print(f"  Status: {batch.status}")
    print(f"  Total requests: {batch.request_counts}")
    
    return batch.id


def check_batch_status(batch_id: str) -> Dict:
    """Check status of a batch job"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    batch = client.batches.retrieve(batch_id)
    
    print(f"\nBatch Job: {batch_id}")
    print(f"  Status: {batch.status}")
    print(f"  Created at: {batch.created_at}")
    print(f"  Request counts: {batch.request_counts}")
    
    if batch.completed_at:
        print(f"  Completed at: {batch.completed_at}")
    if batch.failed_at:
        print(f"  Failed at: {batch.failed_at}")
    if batch.error:
        print(f"  Error: {batch.error}")
    
    return {
        "id": batch.id,
        "status": batch.status,
        "request_counts": batch.request_counts,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id
    }


def download_batch_results(
    batch_id: str,
    output_jsonl_path: str
) -> bool:
    """
    Download batch results
    
    Returns:
        True if successful
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"Batch not completed yet. Current status: {batch.status}")
        return False
    
    if not batch.output_file_id:
        print("No output file available")
        return False
    
    print(f"\nDownloading results from file: {batch.output_file_id}...")
    
    # Download output file
    file_response = client.files.content(batch.output_file_id)
    
    # Save to file
    with open(output_jsonl_path, 'wb') as f:
        f.write(file_response.content)
    
    print(f"✓ Results saved to: {output_jsonl_path}")
    return True


def process_batch_results(
    results_jsonl_path: str,
    test_csv_path: str,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Process batch results and create final CSV
    """
    print(f"\nProcessing batch results from {results_jsonl_path}...")
    
    # Load test data
    test_df = pd.read_csv(test_csv_path, encoding='utf-8-sig')
    
    # Load batch results
    results = {}
    with open(results_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            idx = int(custom_id.replace('request-', ''))
            
            if result['response']['status_code'] == 200:
                answer = result['response']['body']['choices'][0]['message']['content']
                results[idx] = answer
            else:
                results[idx] = f"ERROR: {result['response']['body']}"
    
    # Add predictions to test data
    test_df['predicted_answer'] = test_df.index.map(lambda i: results.get(i, "ERROR: No response"))
    
    # Save results
    test_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Results saved to: {output_csv_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("BATCH RESULTS STATISTICS")
    print("="*80)
    print(f"Total questions: {len(test_df)}")
    print(f"Successful responses: {sum(1 for v in results.values() if not v.startswith('ERROR'))}")
    print(f"Errors: {sum(1 for v in results.values() if v.startswith('ERROR'))}")
    
    return test_df


def wait_for_batch_completion(
    batch_id: str,
    check_interval: int = 60,
    max_wait_time: int = 86400
) -> bool:
    """
    Wait for batch to complete
    
    Args:
        batch_id: Batch job ID
        check_interval: Seconds between status checks (default: 60)
        max_wait_time: Maximum time to wait in seconds (default: 24 hours)
    
    Returns:
        True if completed successfully
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    start_time = time.time()
    
    print(f"\nWaiting for batch {batch_id} to complete...")
    print(f"Checking every {check_interval} seconds (max wait: {max_wait_time/3600:.1f} hours)")
    
    while True:
        batch = client.batches.retrieve(batch_id)
        elapsed = time.time() - start_time
        
        print(f"\n[{elapsed/60:.1f}m] Status: {batch.status}")
        if batch.request_counts:
            print(f"  Progress: {batch.request_counts}")
        
        if batch.status == "completed":
            print(f"\n✓ Batch completed successfully!")
            return True
        
        if batch.status in ["failed", "expired", "cancelled"]:
            print(f"\n❌ Batch {batch.status}")
            if batch.error:
                print(f"Error: {batch.error}")
            return False
        
        if elapsed > max_wait_time:
            print(f"\n❌ Max wait time exceeded")
            return False
        
        time.sleep(check_interval)


def main():
    """Main function for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI Batch API for BM25 RAG Evaluation")
    parser.add_argument("--action", required=True, 
                       choices=["create", "submit", "status", "download", "process", "full"],
                       help="Action to perform")
    parser.add_argument("--test-csv", default=r"E:\Github\uit_chatbot\system_test\test_results.csv",
                       help="Path to test questions CSV")
    parser.add_argument("--db-path", default=r"E:\Github\uit_chatbot\normailizer\uit_law_points.db",
                       help="Path to database")
    parser.add_argument("--input-jsonl", default="batch_input.jsonl",
                       help="Path to batch input JSONL file")
    parser.add_argument("--output-jsonl", default="batch_output.jsonl",
                       help="Path to batch output JSONL file")
    parser.add_argument("--output-csv", default="bm25_llm_results.csv",
                       help="Path to final results CSV")
    parser.add_argument("--batch-id", help="Batch job ID (for status, download actions)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--wait", action="store_true", help="Wait for batch completion")
    
    args = parser.parse_args()
    
    if args.action == "create":
        create_batch_input_file(args.test_csv, args.db_path, args.input_jsonl, args.top_k)
    
    elif args.action == "submit":
        batch_id = submit_batch_job(args.input_jsonl)
        print(f"\nSave this batch ID: {batch_id}")
        
        if args.wait:
            if wait_for_batch_completion(batch_id):
                download_batch_results(batch_id, args.output_jsonl)
                process_batch_results(args.output_jsonl, args.test_csv, args.output_csv)
    
    elif args.action == "status":
        if not args.batch_id:
            print("--batch-id required for status check")
            return
        check_batch_status(args.batch_id)
    
    elif args.action == "download":
        if not args.batch_id:
            print("--batch-id required for download")
            return
        download_batch_results(args.batch_id, args.output_jsonl)
    
    elif args.action == "process":
        process_batch_results(args.output_jsonl, args.test_csv, args.output_csv)
    
    elif args.action == "full":
        # Full workflow
        print("="*80)
        print("STEP 1: Creating batch input file")
        print("="*80)
        create_batch_input_file(args.test_csv, args.db_path, args.input_jsonl, args.top_k)
        
        print("\n" + "="*80)
        print("STEP 2: Submitting batch job")
        print("="*80)
        batch_id = submit_batch_job(args.input_jsonl)
        print(f"\nBatch ID: {batch_id}")
        
        print("\n" + "="*80)
        print("STEP 3: Waiting for completion")
        print("="*80)
        if wait_for_batch_completion(batch_id):
            print("\n" + "="*80)
            print("STEP 4: Downloading results")
            print("="*80)
            if download_batch_results(batch_id, args.output_jsonl):
                print("\n" + "="*80)
                print("STEP 5: Processing results")
                print("="*80)
                process_batch_results(args.output_jsonl, args.test_csv, args.output_csv)
                
                print("\n" + "="*80)
                print("✓ BATCH PROCESSING COMPLETE")
                print("="*80)


if __name__ == "__main__":
    main()
