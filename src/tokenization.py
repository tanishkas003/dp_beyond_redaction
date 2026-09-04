"""
tokenization.py
===============

Exact reversible tokenization for entities that must be recovered
exactly after downstream LLM processing.
"""

from dataclasses import dataclass, field


@dataclass
class Tokenizer:
    """
    Generates unique typed placeholders for exact-recovery entities.

    Examples:
        SSN         -> [SSN_1]
        SSN         -> [SSN_2]
        EMAIL       -> [EMAIL_1]
        PERSON      -> [PERSON_1]
    """

    counters: dict[str, int] = field(default_factory=dict)

    def tokenize(self, entity_type: str) -> str:
        entity_type = entity_type.upper()

        count = self.counters.get(entity_type, 0) + 1
        self.counters[entity_type] = count

        return f"[{entity_type}_{count}]"

    def reset(self) -> None:
        """Reset all session-local token counters."""
        self.counters.clear()