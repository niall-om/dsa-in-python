from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from dsa_in_python.heaps._storage import BinaryHeapStore

HeapItem = TypeVar('HeapItem')


class ArrayBinaryHeapStore(BinaryHeapStore[HeapItem, int]):
    """
    Array-backed implementation of :class:`BinaryHeapStore`.

    This store represents the complete binary tree using a Python list where
    each element corresponds to a position in **level-order**.

    Handles
    -------
    In this implementation, a handle is simply an integer index into the
    underlying array.

    The standard binary heap index relationships are used:

        parent(i)      = (i - 1) // 2
        left_child(i)  = 2*i + 1
        right_child(i) = 2*i + 2

    The array representation naturally preserves the **complete binary tree
    shape property**, making insertion at the last position and navigation
    between parent/child positions efficient.

    Notes
    -----
    This class does **not enforce the heap ordering property**. It only
    maintains structural relationships between positions. Heap ordering
    is managed by the heap algorithm that uses this store.
    """

    __slots__ = ('_array',)
    _array: list[HeapItem]

    def __init__(self, items: Iterable[HeapItem] | None) -> None:
        """
        Initialize the heap store.

        If ``items`` is provided, the underlying array is populated in the order
        given by the iterable. This corresponds to **level-order placement** in
        the conceptual complete binary tree.

        The heap-order property is **not enforced** during construction.

        Parameters
        ----------
        items:
            Optional iterable of heap entries used to initialize the store.
        """
        self._array = list(items) if items is not None else []

    def __len__(self) -> int:
        """
        Return the number of entries stored in the heap.

        Returns
        -------
        int
            Number of elements currently stored.

        Complexity
        ----------
        O(1)
        """
        return len(self._array)

    def is_empty(self) -> bool:
        """
        Determine whether the store contains any elements.

        Returns
        -------
        bool
            True if the store contains no entries, otherwise False.

        Complexity
        ----------
        O(1)
        """
        return len(self._array) == 0

    # ------------- Accessors ------------------
    def root(self) -> int:
        """
        Return the handle of the root position.

        In the array representation, the root is always located at index ``0``.

        Returns
        -------
        int
            Handle identifying the root position.

        Raises
        ------
        IndexError
            If the store is empty.
        """
        if not self._array:
            raise IndexError('heap store is empty')
        return 0

    def last(self) -> int:
        """
        Return the handle of the last position in level-order.

        In the array representation this corresponds to the final index of
        the underlying list.

        Returns
        -------
        int
            Handle of the deepest, rightmost position.

        Raises
        ------
        IndexError
            If the store is empty.
        """
        if not self._array:
            raise IndexError('heap store is empty')
        return len(self._array) - 1

    def parent(self, h: int) -> int | None:
        """
        Return the handle of the parent of position ``h``.

        Parameters
        ----------
        h:
            Handle identifying a position in the store.

        Returns
        -------
        int | None
            The parent handle, or ``None`` if ``h`` refers to the root.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle for this store.
        """
        self._validate_handle(h)
        if h == 0:
            return None
        return (h - 1) // 2

    def left_child(self, h: int) -> int | None:
        """
        Return the handle of the left child of position ``h``.

        Parameters
        ----------
        h:
            Handle identifying a position in the store.

        Returns
        -------
        int | None
            Handle of the left child if it exists, otherwise ``None``.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle.
        """
        self._validate_handle(h)
        c = 2 * h + 1
        return c if c < len(self._array) else None

    def right_child(self, h: int) -> int | None:
        """
        Return the handle of the right child of position ``h``.

        Parameters
        ----------
        h:
            Handle identifying a position in the store.

        Returns
        -------
        int | None
            Handle of the right child if it exists, otherwise ``None``.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle.
        """
        self._validate_handle(h)
        c = 2 * h + 2
        return c if c < len(self._array) else None

    def has_left_child(self, h: int) -> bool:
        """
        Determine whether position ``h`` has a left child.

        Parameters
        ----------
        h:
            Handle identifying a position in the store.

        Returns
        -------
        bool
            True if a left child exists, otherwise False.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle.
        """
        self._validate_handle(h)
        c = 2 * h + 1
        return c < len(self._array)

    def has_right_child(self, h: int) -> bool:
        """
        Determine whether position ``h`` has a right child.

        Parameters
        ----------
        h:
            Handle identifying a position in the store.

        Returns
        -------
        bool
            True if a right child exists, otherwise False.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle.
        """
        self._validate_handle(h)
        c = 2 * h + 2
        return c < len(self._array)

    def get(self, h: int) -> HeapItem:
        """
        Return the entry stored at position ``h``.

        Parameters
        ----------
        h:
            Handle identifying a position in the store.

        Returns
        -------
        HeapEntry
            The entry stored at the given position.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle.
        """
        self._validate_handle(h)
        return self._array[h]

    def handles_level_order(self) -> Iterable[int]:
        """
        Yield handles in level-order traversal.

        For an array-backed heap store this corresponds to iterating
        over the array indices from ``0`` to ``n - 1``.

        Yields
        ------
        int
            Handles representing positions in level-order.
        """
        yield from range(len(self._array))

    def handles_reverse_level_order(self) -> Iterable[int]:
        """
        Yield handles in reverse level-order traversal.

        For the array representation this corresponds to iterating
        from the last index down to ``0``.

        Yields
        ------
        int
            Handles representing positions from the deepest, rightmost
            node back toward the root.
        """
        yield from range(len(self._array) - 1, -1, -1)

    # -------------- Mutators -------------------
    def set(self, h: int, item: HeapItem) -> None:
        """
        Replace the entry stored at position ``h``.

        Parameters
        ----------
        h:
            Handle identifying the position to update.

        item:
            The new entry to store at the given position.

        Raises
        ------
        ValueError
            If ``h`` is not a valid handle.
        """
        self._validate_handle(h)
        self._array[h] = item

    def insert_last(self, item: HeapItem) -> int:
        """
        Insert a new entry at the last position of the heap.

        The new entry is appended to the underlying array, preserving
        the **complete binary tree shape property**.

        Parameters
        ----------
        item:
            Entry to insert.

        Returns
        -------
        int
            Handle identifying the newly inserted position.

        Complexity
        ----------
        Amortized O(1).
        """
        self._array.append(item)
        return len(self._array) - 1

    def remove_last(self) -> HeapItem:
        """
        Remove and return the entry stored at the last position.

        Returns
        -------
        HeapEntry
            The removed entry.

        Raises
        ------
        IndexError
            If the store is empty.

        Complexity
        ----------
        O(1)
        """
        if not self._array:
            raise IndexError('heap store is empty')
        return self._array.pop()

    def swap(self, h1: int, h2: int) -> None:
        """
        Exchange the entries stored at two positions.

        Parameters
        ----------
        h1:
            Handle of the first position.

        h2:
            Handle of the second position.

        Raises
        ------
        ValueError
            If either handle is invalid.
        """
        self._validate_handle(h1)
        self._validate_handle(h2)
        self._array[h1], self._array[h2] = self._array[h2], self._array[h1]

    # ------------- Internal Helpers -------------
    def _validate_handle(self, h: int) -> None:
        """
        Validate that ``h`` is a valid handle foqr this store.

        A handle is valid if it is an integer index within the bounds of
        the underlying array.

        Parameters
        ----------
        h:
            Handle to validate.

        Raises
        ------
        ValueError
            If the handle type is incorrect or if the index is outside the
            bounds of the array.
        """
        if not isinstance(h, int):
            raise ValueError('Invalid handle type')
        if h < 0 or h >= len(self._array):
            raise ValueError('Invalid handle: out of range')
