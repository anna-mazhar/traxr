"""Context window budget allocator for synthesizer prompts.

Prevents context overflow on long episodes by allocating character budgets
to memory entries based on priority weights and recency. Higher-priority
entry types (e.g., analysis, calculation) get more space; within the same
priority, more recent entries are preferred.

Usage:
    budget = ContextBudget(max_tokens=12000)
    allocations = budget.allocate(entries)
    for entry, max_chars in allocations:
        truncated = entry.content[:max_chars]
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from .state import MemoryEntry


# Default priority weights per entry_type.
# Higher weight = more important = gets more of the budget.
DEFAULT_PRIORITIES: Dict[str, float] = {
    "analysis": 1.0,
    "calculation": 0.9,
    "visual_analysis": 0.9,
    "audio_analysis": 0.9,
    "document_summary": 0.9,
    "web_research": 0.8,
    "fact_check": 0.7,
    "note": 0.6,
    "critique": 0.5,
    "plan": 0.3,
}

# Status quality multipliers — verified/final entries are more valuable.
STATUS_MULTIPLIERS: Dict[str, float] = {
    "verified": 1.2,
    "final": 1.1,
    "preliminary": 1.0,
    "failed_attempt": 0.3,
}


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (average ~4 chars per token for English)."""
    return max(1, len(text) // 4)


@dataclass
class EntryAllocation:
    """A memory entry with its allocated character budget."""
    entry: MemoryEntry
    max_chars: int
    priority_score: float


class ContextBudget:
    """Allocates a character budget across memory entries by priority.

    Args:
        max_tokens: Maximum token budget for the synthesizer prompt.
            The allocator reserves ~30% for the prompt template and
            instructions, distributing the rest among entries.
        priorities: Mapping of entry_type -> weight. Defaults to
            DEFAULT_PRIORITIES.
    """

    def __init__(
        self,
        max_tokens: int = 12000,
        priorities: Optional[Dict[str, float]] = None,
    ):
        self.max_tokens = max_tokens
        self.priorities = priorities or DEFAULT_PRIORITIES

        # Reserve ~30% of the budget for prompt template, instructions, etc.
        usable_tokens = int(max_tokens * 0.70)
        # Convert to chars (roughly 4 chars/token)
        self._max_chars = usable_tokens * 4

    def _score_entry(self, entry: MemoryEntry) -> float:
        """Compute priority score for a single entry.

        Score = type_weight * status_multiplier * confidence * recency_bonus
        """
        type_weight = self.priorities.get(entry.entry_type, 0.5)
        status_mult = STATUS_MULTIPLIERS.get(entry.status, 1.0)
        confidence = max(0.1, entry.confidence)  # floor at 0.1

        # Recency bonus: step number normalized (higher step = more recent)
        # We don't know max step here, so just use raw step as a tiebreaker.
        # The actual ordering happens in allocate() after computing all scores.
        recency = entry.step * 0.01  # Small additive bonus

        return type_weight * status_mult * confidence + recency

    def allocate(
        self,
        entries: List[MemoryEntry],
    ) -> List[EntryAllocation]:
        """Allocate character budgets to entries based on priority.

        Returns a list of EntryAllocation sorted by the original entry order
        (preserving chronological ordering for prompt construction). Each
        allocation includes the maximum characters to include from that entry.

        Entries with status "failed_attempt" and low confidence get minimal
        allocations (just enough for a warning snippet).
        """
        if not entries:
            return []

        # Score each entry
        scored = []
        for i, entry in enumerate(entries):
            score = self._score_entry(entry)
            scored.append((i, entry, score))

        # Calculate total weighted demand
        total_score = sum(s for _, _, s in scored)
        if total_score == 0:
            total_score = 1.0  # avoid division by zero

        # Distribute budget proportionally to score
        allocations: List[EntryAllocation] = []
        remaining_budget = self._max_chars

        for original_idx, entry, score in scored:
            content_len = len(entry.content)

            # Proportional allocation
            share = score / total_score
            allocated_chars = int(self._max_chars * share)

            # Clamp: don't allocate more than the entry actually has
            allocated_chars = min(allocated_chars, content_len)

            # Failed attempts get at most 200 chars (just a snippet)
            if entry.status == "failed_attempt":
                allocated_chars = min(allocated_chars, 200)

            # Minimum: at least 50 chars so entries aren't completely invisible
            allocated_chars = max(allocated_chars, min(50, content_len))

            allocations.append(EntryAllocation(
                entry=entry,
                max_chars=allocated_chars,
                priority_score=score,
            ))

        # Sort back to original order for prompt construction
        # (entries were already in chronological order from memory)
        return allocations

    def fits_in_budget(self, entries: List[MemoryEntry]) -> bool:
        """Check if all entries fit within the token budget without truncation."""
        total_chars = sum(len(e.content) for e in entries)
        return total_chars <= self._max_chars

    def total_allocated_chars(self, allocations: List[EntryAllocation]) -> int:
        """Sum of all allocated characters."""
        return sum(a.max_chars for a in allocations)

    def total_allocated_tokens(self, allocations: List[EntryAllocation]) -> int:
        """Estimated token count for all allocations."""
        return estimate_tokens("x" * self.total_allocated_chars(allocations))
