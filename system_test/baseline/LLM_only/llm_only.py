"""
Create and manage OpenAI Batch API jobs for LLM-only QA evaluation
Baseline system: no retrieval, uses GPT-4o-mini
"""
import pandas as pd
import json
import os
from pathlib import Path
from openai import OpenAI
import time
from typing import List, Dict
# from llm_only_qa import LLMOnlyQA, SYSTEM_PROMPT  # import your baseline LLM QA class?

SYSTEM_PROMPT = """
Bạn là trợ lý trả lời câu hỏi cho sinh viên dựa trên **quy chế UIT**.

MỤC TIÊU:
- Trả lời rõ ràng, thân thiện, dễ hiểu.
- Ưu tiên trả lời thẳng vào ý chính trong 1–2 câu đầu, sau đó có thể giải thích thêm.
- Dùng ngôi xưng phù hợp với ngữ cảnh (có thể dùng "em", "bạn" theo câu hỏi).
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
""".strip()

def create_batch_input_file(
    test_csv_path: str,
    output_jsonl_path: str,
) -> int:
    """
    Create JSONL file for OpenAI Batch API (LLM-only QA)

    Returns:
        Number of requests created
    """
    print(f"Loading test questions from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path, encoding='utf-8-sig')

    batch_requests = []
    print(f"\nPreparing {len(test_df)} batch requests...")

    for idx, row in test_df.iterrows():
        question = row['question']

        # Create user prompt directly for LLM-only QA
        user_prompt = f"Câu hỏi: {question}"

        batch_request = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3
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

    print(f"✓ Created {len(batch_requests)} batch requests")
    return len(batch_requests)


def submit_batch_job(
    input_jsonl_path: str,
    description: str = "LLM-only QA Evaluation"
) -> str:
    """Submit batch job to OpenAI and return batch ID"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\nUploading batch file: {input_jsonl_path}...")
    with open(input_jsonl_path, 'rb') as f:
        batch_file = client.files.create(file=f, purpose="batch")
    print(f"✓ File uploaded: {batch_file.id}")

    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )
    print(f"✓ Batch job created: {batch.id} (status: {batch.status})")
    return batch.id


def check_batch_status(batch_id: str) -> Dict:
    """Check status of a batch job"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)

    info = {
        "id": batch.id,
        "status": batch.status,
        "request_counts": batch.request_counts,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id
    }

    print(f"\nBatch Job: {batch.id} | Status: {batch.status} | Requests: {batch.request_counts}")
    if batch.error:
        print(f"Error: {batch.error}")

    return info


def download_batch_results(batch_id: str, output_jsonl_path: str) -> bool:
    """Download batch results"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Batch not completed yet (status: {batch.status})")
        return False
    if not batch.output_file_id:
        print("No output file available")
        return False

    print(f"\nDownloading results from file: {batch.output_file_id}...")
    file_response = client.files.content(batch.output_file_id)

    with open(output_jsonl_path, 'wb') as f:
        f.write(file_response.content)
    print(f"✓ Results saved to: {output_jsonl_path}")
    return True


def process_batch_results(
    results_jsonl_path: str,
    test_csv_path: str,
    output_csv_path: str
) -> pd.DataFrame:
    """Process batch results and save to CSV"""
    print(f"\nProcessing batch results from {results_jsonl_path}...")
    test_df = pd.read_csv(test_csv_path, encoding='utf-8-sig')

    results = {}
    with open(results_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            idx = int(result['custom_id'].replace('request-', ''))
            if result['response']['status_code'] == 200:
                results[idx] = result['response']['body']['choices'][0]['message']['content']
            else:
                results[idx] = f"ERROR: {result['response']['body']}"

    test_df['predicted_answer'] = test_df.index.map(lambda i: results.get(i, "ERROR: No response"))
    test_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"✓ Results saved to: {output_csv_path}")
    print("\nBatch Results Statistics:")
    print(f"Total questions: {len(test_df)}")
    print(f"Successful responses: {sum(1 for v in results.values() if not v.startswith('ERROR'))}")
    print(f"Errors: {sum(1 for v in results.values() if v.startswith('ERROR'))}")

    return test_df


def wait_for_batch_completion(batch_id: str, check_interval: int = 60, max_wait_time: int = 86400) -> bool:
    """Wait for batch to complete"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.time()

    print(f"\nWaiting for batch {batch_id} to complete...")
    while True:
        batch = client.batches.retrieve(batch_id)
        elapsed = time.time() - start_time
        print(f"[{elapsed/60:.1f}m] Status: {batch.status}")
        if batch.status == "completed":
            print("✓ Batch completed successfully!")
            return True
        if batch.status in ["failed", "expired", "cancelled"]:
            print(f"❌ Batch {batch.status}")
            return False
        if elapsed > max_wait_time:
            print("❌ Max wait time exceeded")
            return False
        time.sleep(check_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI Batch API for LLM-only QA Evaluation")
    parser.add_argument("--action", required=True,
                        choices=["create", "submit", "status", "download", "process", "full"])
    parser.add_argument("--test-csv", required=True, help="Path to test questions CSV")
    parser.add_argument("--input-jsonl", default="batch_input.jsonl")
    parser.add_argument("--output-jsonl", default="batch_output.jsonl")
    parser.add_argument("--output-csv", default="llm_only_results.csv")
    parser.add_argument("--batch-id", help="Batch job ID (for status/download)")
    parser.add_argument("--wait", action="store_true", help="Wait for batch completion")
    args = parser.parse_args()

    if args.action == "create":
        create_batch_input_file(args.test_csv, args.input_jsonl)
    elif args.action == "submit":
        batch_id = submit_batch_job(args.input_jsonl)
        print(f"Batch ID: {batch_id}")
        if args.wait:
            if wait_for_batch_completion(batch_id):
                download_batch_results(batch_id, args.output_jsonl)
                process_batch_results(args.output_jsonl, args.test_csv, args.output_csv)
    elif args.action == "status":
        if args.batch_id:
            check_batch_status(args.batch_id)
    elif args.action == "download":
        if args.batch_id:
            download_batch_results(args.batch_id, args.output_jsonl)
    elif args.action == "process":
        process_batch_results(args.output_jsonl, args.test_csv, args.output_csv)
    elif args.action == "full":
        create_batch_input_file(args.test_csv, args.input_jsonl)
        batch_id = submit_batch_job(args.input_jsonl)
        if wait_for_batch_completion(batch_id):
            download_batch_results(batch_id, args.output_jsonl)
            process_batch_results(args.output_jsonl, args.test_csv, args.output_csv)
