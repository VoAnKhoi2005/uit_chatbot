# BM25 Baseline RAG QA System

This is a baseline BM25-based Retrieval-Augmented Generation (RAG) system for answering questions about UIT law points.

## System Architecture

1. **Document Loading**: Loads legal documents from SQLite database (`uit_law_points.db`)
2. **Indexing**: Builds BM25 index using rank-bm25 library
3. **Retrieval**: Retrieves top-k relevant documents for each query
4. **Answer Generation**: Concatenates retrieved documents to form an answer

## Files

- `bm25_rag.py`: Main BM25 RAG implementation
- `evaluate.py`: Evaluation script for test questions
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Test with a single question:

```bash
python bm25_rag.py "Người học bồi dưỡng nâng cao trình độ học vấn, nghề nghiệp thì có được cấp chứng chỉ không?"
```

### Run full evaluation on test set:

```bash
python evaluate.py
```

This will:
1. Load test questions from `E:\Github\uit_chatbot\system_test\test_results.csv`
2. Process each question through the BM25 RAG system
3. Save results to `bm25_results.csv`
4. Print evaluation statistics

## Database Schema

The system uses the following tables from `uit_law_points.db`:

- **items**: Contains legal document items with:
  - id, doc_id, parent_id
  - level, title, heading, content
  - path, ordinal
  - created_at, updated_at

- **documents**: Contains document metadata:
  - id, so_hieu, title
  - issued_date, effective_date
  - unit, status

## BM25 Parameters

- Default top-k retrieval: 5 documents
- Tokenization: Simple whitespace and punctuation-based
- Scoring: BM25Okapi algorithm from rank-bm25

## Output Format

Results CSV contains:
- question: Original question
- extractive_answer: Ground truth extractive answer
- abstractive_answer: Ground truth abstractive answer
- predicted_answer: System-generated answer
- question_type: Type of question (EXACT_RULE, NEAR_RULE, etc.)
- retrieved_doc_ids: List of retrieved document IDs
- retrieval_scores: BM25 scores for retrieved documents
- top_score: Highest BM25 score
- num_retrieved: Number of documents retrieved

## Performance Metrics

The evaluation script provides:
- Question type distribution
- Retrieval score statistics (mean, max, min)
- Average number of documents retrieved
- Sample predictions

## Limitations

This is a baseline system with simple answer generation (document concatenation). More advanced systems may use:
- Better tokenization (Vietnamese-specific)
- Query expansion
- Re-ranking
- Generative models for answer synthesis
- Hybrid retrieval methods

## Future Improvements

1. Vietnamese word segmentation
2. Query preprocessing and expansion
3. Answer post-processing
4. Hybrid retrieval (BM25 + semantic)
5. Fine-tuned answer generation
