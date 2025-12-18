"""
Script to call real backend pipeline for each test question and create batch API requests
based on ACTUAL retrieval results (evidence + ontology facts).

This tests the real retrieval & classification pipeline, not simulated data.
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

from llm.gpt_client import GPTLLMClient
from llm.orchestrator import ChatPipeline

async def main():
    # Read test questions
    csv_path = Path('test/test_question.csv')
    questions = []

    print('Reading test questions...')
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            question = row.get('question') or row.get('\ufeffquestion', '')
            if question:
                questions.append({
                    'id': idx,
                    'question': question.strip()
                })

    print(f'Loaded {len(questions)} questions')

    # Initialize real backend pipeline
    print('\nInitializing ChatPipeline...')
    gpt_client = GPTLLMClient()
    pipeline = ChatPipeline(llm_client=gpt_client)
    print('Pipeline initialized')

    # Process each question through real pipeline
    batch_requests = []
    results_log = []

    print('\nProcessing questions through real pipeline...')
    for i, q in enumerate(questions, 1):
        print(f'[{i}/{len(questions)}] Processing: {q["question"][:60]}...')

        result = await pipeline.get_context(question=q['question'], debug=True)
        if not result:
            print(f'No result for question ID {q["id"]}')
            continue

        # Extract system prompt, context, and debug info
        system_prompt = result.get('system_prompt', '')
        context = result.get('context', '')
        debug_info = result.get('debug', {})

        # Create full prompt with context
        full_user_prompt = f"{q['question']}\n\nContext:\n{context}"

        # Create batch request in OpenAI format
        batch_request = {
            "custom_id": f"request-{q['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "max_tokens": 1000
            }
        }
        batch_requests.append(batch_request)

        # Log result for debugging
        results_log.append({
            "question_id": q['id'],
            "question": q['question'],
            "context": context,
            "system_prompt": system_prompt,
            "debug_info": debug_info
        })

        print(f'  ✓ Context length: {len(context)} chars')

    # Save batch requests to JSONL file
    output_file = Path('test_uit_question.jsonl')
    print(f'\nSaving {len(batch_requests)} batch requests to {output_file}...')
    with open(output_file, 'w', encoding='utf-8') as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    print(f'✓ Saved to {output_file}')

    # Save debug log
    log_file = Path('batch_creation_log.json')
    print(f'\nSaving debug log to {log_file}...')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f'✓ Saved debug log to {log_file}')

    print(f'\n=== Summary ===')
    print(f'Total questions processed: {len(batch_requests)}')
    print(f'Batch file ready: {output_file}')

if __name__ == '__main__':
    asyncio.run(main())
