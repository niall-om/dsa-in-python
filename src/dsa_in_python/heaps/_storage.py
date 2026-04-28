from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

HeapItem = TypeVar('HeapItem')
HeapHandle = TypeVar('HeapHandle')


# TODO: fix inconsisten doc strings in BinaryHeapStore


class BinaryHeapStore(Protocol[HeapItem, HeapHandle]):
    """
    Storage backend for a binary heap.

    This protocol defines the structural and navigational operations required
    by a binary heap implementation. The store is responsible for maintaining
    the *complete binary tree* shape property. It does NOT enforce the heap
    ordering property — that responsibility belongs to the heap algorithm.

    The store operates using opaque handles (`HeapHandle`) that identify positions
    within the structure. The heap algorithm must treat handles as opaque
    tokens and only interact with them via the methods defined here.

    Implementations may be backed by arrays, linked trees, or other
    complete-binary-tree representations.
    """

    def __init__(self, items: Iterable[HeapItem] | None) -> None:
        """
        Constructs a HeapStore.

        If optional arg `items` is provided, constructs a heap satisifying
        the *complete binary tree* shape property. It does NOT enforce the heap
        ordering property - that responsibility belongs to the heap algorithm.
        """
        ...

    def __len__(self) -> int:
        """Return the number of elements currently stored."""
        ...

    def is_empty(self) -> bool:
        """Return True if the store is empty."""
        ...

    # ------------- Accessors ------------------
    def root(self) -> HeapHandle:
        """
        Return the handle of the root element.

        Raises:
            IndexError: if the store is empty.
        """
        ...

    def last(self) -> HeapHandle:
        """
        Return the handle of the last (most recently inserted) position
        in level-order.

        This corresponds to the deepest, rightmost node in the complete tree.

        Raises:
            IndexError: if the store is empty.
        """
        ...

    def parent(self, h: HeapHandle) -> HeapHandle | None:
        """
        Return the handle of the parent of the position given by handle `h`.

        Returns:
            The parent handle, or None if the position is the root.

        Raises:
            ValurError: if handle `h` is invalid or no longer exists.
        """
        ...

    def left_child(self, h: HeapHandle) -> HeapHandle | None:
        """
        Return the handle of the left child of the given position.

        Returns:
            The left child handle, or None if no left child exists.

        Raises:
            ValueError: if handle `h` is invalid or no longer exists.
        """
        ...

    def right_child(self, h: HeapHandle) -> HeapHandle | None:
        """
        Return the handle of the right child of the given position.

        Returns:
            The right child handle, or None if no right child exists.

        Raises:
            ValueError: if handle `h` is invalid or no longer exists.
        """
        ...

    def has_left_child(self, h: HeapHandle) -> bool:
        """Return True if position `h` has a left child."""
        ...

    def has_right_child(self, h: HeapHandle) -> bool:
        """Return True if position `h` has a right child."""
        ...

    def get(self, h: HeapHandle) -> HeapItem:
        """
        Return the entry stored at position `h` in the heap.

        Raises:
            ValueError: if the handle `h` is invalid or no longer exists.
        """
        ...

    def handles_level_order(self) -> Iterable[HeapHandle]:
        """Yield handles in level-order (breadth first), from root to last."""
        ...

    def handles_reverse_level_order(self) -> Iterable[HeapHandle]:
        """Yiled handles in reverse level-order, from last to root."""
        ...

    # -------------- Mutators -------------------
    def set(self, h: HeapHandle, item: HeapItem) -> None: ...

    def insert_last(self, item: HeapItem) -> HeapHandle:
        """
        Insert a new element at the last position in the complete tree.

        This preserves the structural (completeness) property.

        Returns:
            The handle of the newly inserted position.
        """
        ...

    def remove_last(self) -> HeapItem:
        """
        Remove and return the value stored at the last position
        in the complete tree.

        This preserves the structural (completeness) property.

        Raises:
            IndexError: if the store is empty.
        """
        ...

    def swap(self, h1: HeapHandle, h2: HeapHandle) -> None:
        """
        Exchange the items stored at two positions `h1` and `h2`.

        This operation is used by the heap algorithm to restore
        the heap-order property during upheap and downheap operations.

        Returns: None

        Raises:
            ValueEror: if either handle `h1` or `h2` is invalid or no longer exists.
        """
        ...
