from typing import Optional, Dict

class ConversationStateStore:
    """
    In-memory conversation state keyed by user_id/session_id.
    Stores last question and last grounding article_id for each user.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, Optional[str]]] = {}

    def get_last_question(self, user_id: str) -> Optional[str]:
        return self._store.get(user_id, {}).get("last_question")

    def get_last_grounding(self, user_id: str) -> Optional[str]:
        return self._store.get(user_id, {}).get("last_grounding")

    def update(self, user_id: str, question: str, grounding_article_id: Optional[str]):
        self._store[user_id] = {
            "last_question": question,
            "last_grounding": grounding_article_id,
        }
