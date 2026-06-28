"""Keyword-based retrieval implementation using in-memory storage."""

from typing import List, Optional
import random

from .base import RetrievalComponent
from .items import RetrievalItem, RetrievalResult


class InMemoryRetrieval(RetrievalComponent):
    """In-memory retrieval using keyword matching.

    Scores items by word overlap between query and content.
    Suitable for experiments where simple retrieval is sufficient.
    """

    def __init__(self, seed: Optional[int] = None):
        self._items: List[RetrievalItem] = []
        self._seed = seed
        self._rng = random.Random(seed)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Retrieve items using simple keyword matching."""
        if not self._items:
            return RetrievalResult(
                query=query,
                items=[],
                total_available=0,
            )

        # Simple scoring based on word overlap
        query_words = set(query.lower().split())
        scored_items = []

        for item in self._items:
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)
            # Add some randomness for variety
            score = overlap / max(len(query_words), 1) + self._rng.random() * 0.1
            scored_items.append((item, min(score, 1.0)))

        # Sort by score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Apply filters if any
        if filters:
            source_filter = filters.get("source")
            if source_filter:
                scored_items = [
                    (item, score)
                    for item, score in scored_items
                    if item.source == source_filter
                ]

        # Take top_k
        top_items = []
        for item, score in scored_items[:top_k]:
            # Create new item with computed score
            new_item = RetrievalItem(
                content=item.content,
                score=score,
                source=item.source,
                metadata=item.metadata.copy(),
            )
            new_item.is_injected = item.is_injected
            new_item.is_oracle = item.is_oracle
            top_items.append(new_item)

        return RetrievalResult(
            query=query,
            items=top_items,
            total_available=len(self._items),
        )

    def get_total_items(self) -> int:
        """Get total number of items."""
        return len(self._items)

    def add_item(self, item: RetrievalItem) -> None:
        """Add an item to the index."""
        self._items.append(item)

    def add_items(self, items: List[RetrievalItem]) -> None:
        """Add multiple items to the index."""
        self._items.extend(items)

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()

    def seed_with_default_documents(self, task_id: str) -> None:
        """Seed with some default documents for a task."""
        default_docs = [
            RetrievalItem(
                content=f"Document about {task_id}: This contains relevant information "
                f"that can help answer questions about the topic.",
                score=0.0,
                source=f"doc_{task_id}_1",
            ),
            RetrievalItem(
                content=f"Background information for {task_id}: Historical context and "
                f"foundational knowledge about the subject matter.",
                score=0.0,
                source=f"doc_{task_id}_2",
            ),
            RetrievalItem(
                content=f"Expert analysis on {task_id}: Detailed examination of the key "
                f"aspects and their implications.",
                score=0.0,
                source=f"doc_{task_id}_3",
            ),
            RetrievalItem(
                content=f"Case study related to {task_id}: Real-world example demonstrating "
                f"practical applications and outcomes.",
                score=0.0,
                source=f"doc_{task_id}_4",
            ),
            RetrievalItem(
                content=f"Summary of {task_id}: Overview of main points and conclusions "
                f"drawn from comprehensive research.",
                score=0.0,
                source=f"doc_{task_id}_5",
            ),
        ]
        self.add_items(default_docs)
