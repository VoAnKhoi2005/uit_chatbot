# Vector Database Rebuild Guide

## 📊 Overview

The backend uses **2 vector databases** for hybrid retrieval:

1. **Text Chunks Vector DB** - Article text with embeddings
2. **Knowledge Graph Vector DB** - Concepts & relations with embeddings

---

## 🔹 Vector DB #1: Text Chunks (Primary)

### Location
```
backend/retrieval/text_rag/vector_store.db
```

### Rebuild Command
```bash
cd backend
python -m retrieval.text_rag.build_index
```

### Source Data
```
backend/graph/mongo_export_uit/KB_UIT.items.json
```

### Environment Variables (Optional)
```bash
UIT_CONTENT_JSON=backend/graph/mongo_export_uit/KB_UIT.items.json
UIT_VECTOR_DB=backend/retrieval/text_rag/vector_store.db
UIT_CHUNK_MAX_CHARS=800
```

### What It Does
1. Reads `KB_UIT.items.json` (article data)
2. Splits articles into ~800 character chunks
3. Generates Vietnamese SBERT embeddings
4. Stores in SQLite: chunk_id, article_id, clause_id, text, metadata, embeddings

### Current Stats
- **Records**: 1,562 text chunks
- **Size**: 7.39 MB
- **Model**: Vietnamese SBERT

---

## 🔹 Vector DB #2: Knowledge Graph (Concepts & Relations)

### Location
```
backend/retrieval/src/db/vector.db
```

### Rebuild Command
```bash
cd backend/retrieval/src/db
python import_vector_db.py
```

### ⚠️ Prerequisites
**Requires MongoDB running** with populated collections:
- `KB_UIT.concepts`
- `KB_UIT.relations`

### Environment Variables (Optional)
```bash
VECTOR_DB_PATH=backend/retrieval/src/db/vector.db
```

### What It Does
1. Connects to MongoDB (KB_UIT database)
2. Reads concepts and relations collections
3. Generates embeddings for entities + synonyms
4. Stores in SQLite with parent_id mappings

### Current Stats
- **Records**: 899 concepts + 1,213 relations = 2,112 entities
- **Size**: 8.77 MB
- **Model**: keepitreal/vietnamese-sbert

---

## 🔄 Complete Rebuild Process

### Step 1: Rebuild Text Vector DB (Easy)
```bash
cd backend
python -m retrieval.text_rag.build_index
```
✅ No MongoDB required - uses JSON file

### Step 2: Rebuild Graph Vector DB (Advanced)
```bash
# Start MongoDB first
cd backend/retrieval/src/db
python import_vector_db.py
```
⚠️ Requires MongoDB with KB_UIT database

### Step 3: Verify
```bash
cd backend
python -c "from retrieval.text_rag.vector_store import ChunkVectorStore; print(f'Text chunks: {ChunkVectorStore(\"retrieval/text_rag/vector_store.db\").count_chunks()}')"
```

Or check with SQLite:
```bash
sqlite3 backend/retrieval/text_rag/vector_store.db "SELECT COUNT(*) FROM chunk_vectors;"
sqlite3 backend/retrieval/src/db/vector.db "SELECT COUNT(*) FROM concepts; SELECT COUNT(*) FROM relations;"
```

---

## 📝 When to Rebuild

### Rebuild Text Vector DB when:
- ✅ You update `KB_UIT.items.json` (add/edit articles)
- ✅ You want to change chunking parameters
- ✅ Articles are not being retrieved correctly

### Rebuild Graph Vector DB when:
- ✅ You update concepts/relations in MongoDB
- ✅ You add new synonyms
- ✅ Triplet-based retrieval is not working

---

## 💡 Tips

- **First time setup**: Only rebuild DB #1 (text chunks) - it's sufficient for basic RAG
- **DB #2 (graph)**: Optional - only needed for advanced hybrid retrieval
- **Time**: Rebuilding takes 2-5 minutes depending on hardware
- **Backup**: Always backup `.db` files before rebuilding!
- **After rebuild**: Restart the backend server

---

## 🔍 Quick Status Check

```bash
# Check text chunks count
sqlite3 backend/retrieval/text_rag/vector_store.db "SELECT COUNT(*) FROM chunk_vectors;"

# Check graph entities count
sqlite3 backend/retrieval/src/db/vector.db "SELECT 'Concepts:', COUNT(*) FROM concepts UNION SELECT 'Relations:', COUNT(*) FROM relations;"

# Check file sizes
ls -lh backend/retrieval/text_rag/vector_store.db
ls -lh backend/retrieval/src/db/vector.db
```

---

## 🛠️ Troubleshooting

### Error: "No such file or directory: KB_UIT.items.json"
```bash
# Make sure source file exists
ls backend/graph/mongo_export_uit/KB_UIT.items.json
```

### Error: "Cannot connect to MongoDB"
```bash
# For graph DB rebuild only - start MongoDB first
# Or skip graph DB rebuild if not needed
```

### Low chunk count after rebuild
```bash
# Check source JSON has data
python -c "import json; print(len(json.load(open('backend/graph/mongo_export_uit/KB_UIT.items.json'))))"
```

---

## 📚 Related Files

- **Text chunker**: `backend/retrieval/text_rag/chunker.py`
- **Vector store**: `backend/retrieval/text_rag/vector_store.py`
- **Graph DB**: `backend/retrieval/src/db/vector_db.py`
- **Build script**: `backend/retrieval/text_rag/build_index.py`
- **Import script**: `backend/retrieval/src/db/import_vector_db.py`
