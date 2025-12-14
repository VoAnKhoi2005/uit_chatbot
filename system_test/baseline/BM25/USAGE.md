# BM25 RAG with LLM Answer Generation - Usage Guide

## Overview

This directory contains a BM25-based RAG system with two answer generation modes:
1. **Baseline**: Simple document concatenation (bm25_rag.py)
2. **LLM-Enhanced**: GPT-4o-mini answer generation (bm25_llm.py + batch_api.py)

## Files

- `bm25_rag.py`: Baseline BM25 RAG (no LLM)
- `bm25_llm.py`: BM25 RAG with LLM answer generation
- `batch_api.py`: OpenAI Batch API for efficient bulk processing
- `evaluate.py`: Evaluation script (baseline)
- `requirements.txt`: Dependencies
- `README.md`: General documentation
- `USAGE.md`: This file

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

Create a `.env` file or set environment variable:

```bash
# .env file
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Or on Windows:
```cmd
set OPENAI_API_KEY=your_api_key_here
```

## Usage

### Option 1: Single Question (Interactive)

Test with a single question using LLM:

```bash
python bm25_llm.py "Người học bồi dưỡng nâng cao trình độ học vấn, nghề nghiệp thì có được cấp chứng chỉ không?"
```

### Option 2: Batch Processing (Recommended for 500 questions)

OpenAI Batch API is **50% cheaper** and more efficient for bulk processing.

#### Full Workflow (Automated)

```bash
python batch_api.py --action full --wait
```

This will:
1. Create batch input file with retrieved contexts
2. Submit to OpenAI Batch API
3. Wait for completion (polls every 60 seconds)
4. Download results
5. Generate final CSV

#### Step-by-Step Workflow (Manual Control)

**Step 1: Create batch input file**
```bash
python batch_api.py --action create --input-jsonl batch_input.jsonl
```

**Step 2: Submit batch job**
```bash
python batch_api.py --action submit --input-jsonl batch_input.jsonl
```
Save the returned `batch_id` (e.g., `batch_abc123`)

**Step 3: Check status**
```bash
python batch_api.py --action status --batch-id batch_abc123
```

**Step 4: Download results (when completed)**
```bash
python batch_api.py --action download --batch-id batch_abc123 --output-jsonl batch_output.jsonl
```

**Step 5: Process results into CSV**
```bash
python batch_api.py --action process --output-jsonl batch_output.jsonl --output-csv bm25_llm_results.csv
```

### Option 3: Baseline (No LLM)

For comparison, run baseline without LLM:

```bash
python evaluate.py
```

## Command Line Arguments

### batch_api.py

```
--action         Action to perform: create, submit, status, download, process, full
--test-csv       Path to test questions CSV (default: test_results.csv)
--db-path        Path to database (default: uit_law_points.db)
--input-jsonl    Batch input JSONL file (default: batch_input.jsonl)
--output-jsonl   Batch output JSONL file (default: batch_output.jsonl)
--output-csv     Final results CSV (default: bm25_llm_results.csv)
--batch-id       Batch job ID (required for status/download)
--top-k          Number of documents to retrieve (default: 5)
--wait           Wait for batch completion before returning
```

## Examples

### Custom paths

```bash
python batch_api.py \
  --action full \
  --test-csv my_questions.csv \
  --db-path my_database.db \
  --output-csv my_results.csv \
  --top-k 10
```

### Monitor existing batch

```bash
python batch_api.py --action status --batch-id batch_abc123
```

### Download only

```bash
python batch_api.py --action download --batch-id batch_abc123
```

## OpenAI Batch API Benefits

1. **Cost Savings**: 50% cheaper than regular API
2. **Rate Limits**: Higher throughput for bulk requests
3. **Reliability**: Automatic retries and error handling
4. **Async Processing**: No need to keep connection open

## Typical Processing Time

- **500 questions**: 10-30 minutes (depends on queue)
- **Status checks**: Every 60 seconds by default
- **Max wait time**: 24 hours (configurable)

## Output Format

The final CSV (`bm25_llm_results.csv`) contains:

| Column | Description |
|--------|-------------|
| question | Original question |
| extractive_answer | Ground truth (extractive) |
| abstractive_answer | Ground truth (abstractive) |
| predicted_answer | **LLM-generated answer** |
| question_type | EXACT_RULE, NEAR_RULE, OUT_OF_SCOPE, ERROR |

## Troubleshooting

### API Key Error
```
EnvironmentError: OPENAI_API_KEY is required
```
**Solution**: Set `OPENAI_API_KEY` environment variable

### Batch Still Processing
```
❌ Batch not completed yet. Current status: in_progress
```
**Solution**: Wait longer or use `--wait` flag

### Rate Limit Errors
**Solution**: Use Batch API instead of direct calls

## Cost Estimation

For 500 questions with top-5 retrieval:
- Input tokens: ~500 questions × ~2000 tokens = 1M tokens
- Output tokens: ~500 questions × ~500 tokens = 250K tokens
- Estimated cost: ~$0.05-0.10 with Batch API (50% discount)

## Comparison: Baseline vs LLM

| Feature | Baseline | LLM-Enhanced |
|---------|----------|--------------|
| Speed | Very fast (~2 sec) | Slower (10-30 min for batch) |
| Cost | Free | ~$0.05-0.10 |
| Answer Quality | Concatenated docs | Natural language |
| Formatting | Basic | Markdown formatted |
| Context Understanding | None | Strong |

## Next Steps

1. Run baseline: `python evaluate.py`
2. Run LLM batch: `python batch_api.py --action full --wait`
3. Compare results in CSV files
4. Analyze answer quality improvements

## Support

For issues:
1. Check OpenAI API status: https://status.openai.com/
2. Verify API key and quota
3. Check batch status with `--action status`
