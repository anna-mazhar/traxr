"""Base retrieval component interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .items import RetrievalItem, RetrievalResult


class RetrievalComponent(ABC):
    """Abstract base class for retrieval components."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Retrieve items matching the query.

        Args:
            query: The search query
            top_k: Maximum number of items to return
            filters: Optional filters to apply

        Returns:
            RetrievalResult with matching items
        """
        pass

    @abstractmethod
    def get_total_items(self) -> int:
        """Get total number of items in the index."""
        pass

    @abstractmethod
    def add_item(self, item: RetrievalItem) -> None:
        """Add an item to the index."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from the index."""
        pass
