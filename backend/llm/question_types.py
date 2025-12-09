from enum import Enum


class QuestionType(str, Enum):
    EXACT_RULE = "EXACT_RULE"
    NEAR_RULE = "NEAR_RULE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

