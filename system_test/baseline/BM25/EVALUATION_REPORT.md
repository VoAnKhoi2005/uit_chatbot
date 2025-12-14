# BM25 Baseline RAG QA System - Evaluation Report

**Date:** December 14, 2025  
**Database:** E:\Github\uit_chatbot\normailizer\uit_law_points.db  
**Test Set:** E:\Github\uit_chatbot\system_test\test_results.csv  

---

## Executive Summary

Successfully built and evaluated a BM25-based Retrieval-Augmented Generation (RAG) system for answering questions about UIT law and regulations. The system processes 500 test questions across multiple question types.

---

## System Components

### 1. Document Corpus
- **Total Documents:** 1,352 items loaded from database
- **Source Tables:** items, documents
- **Document Fields:** title, heading, content, doc_title, so_hieu

### 2. Retrieval System
- **Algorithm:** BM25Okapi (from rank-bm25 library)
- **Tokenization:** Simple whitespace and punctuation-based
- **Top-K Retrieval:** 5 documents per query

### 3. Answer Generation
- **Method:** Document concatenation
- **Format:** Top 3 retrieved documents with metadata

---

## Evaluation Results

### Overall Statistics
- **Total Questions:** 500
- **Documents Retrieved per Question:** 5.00 (average)

### Question Type Distribution
| Question Type | Count | Percentage |
|--------------|-------|------------|
| EXACT_RULE | 252 | 50.4% |
| OUT_OF_SCOPE | 217 | 43.4% |
| NEAR_RULE | 27 | 5.4% |
| ERROR | 4 | 0.8% |

### Retrieval Performance
| Metric | Value |
|--------|-------|
| Average Top Score | 27.76 |
| Max Top Score | 99.79 |
| Min Top Score | 8.07 |
| Score Range | 91.72 |

---

## Sample Results

### Example 1: EXACT_RULE Question
**Question:** "Người học bồi dưỡng nâng cao trình độ học vấn, nghề nghiệp thì có được cấp chứng chỉ không?"

**Top Score:** 45.93

**Top Retrieved Document:**
- Document: 172-Qd-Dhcntt 08-3-2023 Quy Che Van Bang Chung Chi
- Section: Khoản 2
- Content: Hiệu trưởng hoặc thủ trưởng đơn vị trực thuộc được ủy quyền có trách nhiệm cấp chứng chỉ cho người học chậm nhất là 30 ngày kể từ ngày kết thúc khóa đào tạo, bồi dưỡng nâng cao trình độ học vấn, nghề nghiệp.

**Assessment:** System correctly retrieved relevant regulation about certificate issuance for vocational training.

---

### Example 2: OUT_OF_SCOPE Question
**Question:** "Phương thức dạy học kết hợp là sự kết hợp giữa những hình thức nào?"

**Top Score:** 27.05

**Top Retrieved Document:**
- Document: 507-Qd-Dhcntt-27-5-2024 Quy Che Dao Tao Cho Sinh Vien
- Section: Khoản 1
- Content: Related to distance education format

**Assessment:** System retrieved related documents about education formats, though specific answer may not be directly stated.

---

### Example 3: NEAR_RULE Question
**Question:** "Hội đồng bảo vệ KLTN sẽ đánh giá theo điểm cho từng sinh viên trên đâu?"

**Top Score:** 35.39

**Top Retrieved Document:**
- Document: 159-Qd-Dhcntt 05-03-2024 Ban Hanh Quy Dinh Kltn
- Section: Điều 12
- Content: About thesis defense council procedures

**Assessment:** Retrieved relevant regulation about thesis defense evaluation.

---

## Strengths

1. ✅ **Fast Processing:** Evaluated 500 questions in ~2 seconds
2. ✅ **Consistent Retrieval:** Retrieved exactly 5 documents per query
3. ✅ **Document Coverage:** Uses 1,352 documents from comprehensive database
4. ✅ **Metadata Preservation:** Maintains document structure (title, section, content)
5. ✅ **Simple Pipeline:** Easy to understand and modify

---

## Limitations

1. ❌ **Basic Tokenization:** No Vietnamese word segmentation
2. ❌ **No Semantic Understanding:** Relies only on lexical matching
3. ❌ **Simple Answer Generation:** Just concatenates documents
4. ❌ **No Query Processing:** No query expansion or reformulation
5. ❌ **Fixed Ranking:** BM25 only, no re-ranking or filtering
6. ❌ **No Answer Synthesis:** Doesn't generate concise answers

---

## Recommendations for Improvement

### Short-term (Quick Wins)
1. Add Vietnamese word segmentation (pyvi, underthesea)
2. Implement answer post-processing (extract key sentences)
3. Add query preprocessing (remove stop words, normalize)
4. Tune BM25 parameters (k1, b)

### Medium-term
1. Implement hybrid retrieval (BM25 + embedding-based)
2. Add re-ranking layer using cross-encoder
3. Implement query expansion using synonyms
4. Add answer confidence scoring

### Long-term
1. Fine-tune Vietnamese language model for answer generation
2. Implement multi-hop reasoning for complex questions
3. Add citation and source tracking
4. Build feedback loop for continuous improvement

---

## File Outputs

1. **bm25_results.csv:** Full evaluation results with predictions
2. **bm25_rag.py:** Main RAG implementation
3. **evaluate.py:** Evaluation script
4. **README.md:** System documentation
5. **requirements.txt:** Python dependencies

---

## Conclusion

The BM25 baseline RAG system provides a solid foundation for answering UIT regulation questions. With an average retrieval score of 27.76 and consistent 5-document retrieval, it demonstrates reliable basic functionality. However, significant improvements are possible through better Vietnamese NLP, semantic understanding, and answer generation techniques.

This baseline establishes performance metrics for comparing future enhanced systems.

---

## Next Steps

1. Review sample predictions for quality assessment
2. Analyze failure cases (low scores, wrong retrievals)
3. Implement Vietnamese tokenization
4. Experiment with semantic retrieval methods
5. Build answer generation module

---

**Report Generated:** 2025-12-14  
**System:** BM25 Baseline RAG v1.0
