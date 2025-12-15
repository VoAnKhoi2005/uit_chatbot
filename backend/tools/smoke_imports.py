# Smoke test for critical imports and class wiring

try:
    from backend.api.main_gpt import app
    print("[OK] FastAPI app import: backend.api.main_gpt:app")
except Exception as e:
    print("[FAIL] FastAPI app import:", e)

try:
    from backend.llm.orchestrator import ChatPipeline
    print("[OK] ChatPipeline import")
except Exception as e:
    print("[FAIL] ChatPipeline import:", e)

try:
    from retrieval.src.retrieval.hybrid_orchestrator import HybridOrchestrator
    print("[OK] HybridOrchestrator import")
except Exception as e:
    print("[FAIL] HybridOrchestrator import:", e)

try:
    from retrieval.src.retrieval.triplet_retriever import TripletRetriever
    print("[OK] TripletRetriever import")
except Exception as e:
    print("[FAIL] TripletRetriever import:", e)

print("Smoke import test completed.")
