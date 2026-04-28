from __future__ import annotations

from typing import Protocol, TypeVar

from dsa_in_python.common.types import SupportsLessThan

K = TypeVar('K', bound=SupportsLessThan)
V = TypeVar('V')


class PriorityQueue(Protocol[K, V]):
    """Protocol representing the Priority Queue ADT public API."""

    def __len__(self) -> int:
        """Return the number of items in the Priority Queue."""
        ...

    def is_empty(self) -> bool:
        """Return True if the Priority Queue does not contain any items."""
        ...

    def add(self, k: K, v: V) -> None:
        """Insert item with key `k` and value `v` into the Priority Queue."""
        ...

    def min(self) -> tuple[K, V]:
        """
        Return a tuple, (k,v), representing the key and value of the item
        in the Priority Queue with the minimum key. (k,v) is not removed.

        Raises:
            IndexError if the Priority Queue is empty.
        """
        ...

    def remove_min(self) -> tuple[K, V]:
        """
        Remove and return a tuple, (k,v), representing the key and value of
        the item in the Priority Queue with the minimum key.

        Raises:
            IndexError if the Priority Queue is empty
        """
        ...
