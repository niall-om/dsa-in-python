"""
Binary heap implementation of the Priority Queue ADT.

This module provides a concrete ``BinaryHeap`` implementation of the
:class:`PriorityQueue` interface. The heap stores ``(key, value)`` pairs and
maintains the **min-heap property**, meaning the entry with the smallest key
is always located at the root.

Design
------
The heap logic is intentionally separated from the physical storage of the
underlying complete binary tree. Storage is delegated to implementations of
:class:`BinaryHeapStore`, which provide structural operations such as
navigation between parent and child positions, insertion at the last position,
and swapping stored entries.

The ``BinaryHeap`` class itself is responsible only for enforcing the
**heap-order property** using standard heap algorithms such as *up-heap*
(percolate up) and *down-heap* (percolate down).

This separation allows different storage strategies to be used, including:

• array-backed stores
• linked-node tree stores
• alternative complete-tree representations

Performance characteristics may therefore depend on the chosen storage
implementation, although heap-ordering operations always follow the
same algorithmic structure.

Key Operations
--------------
``add(k, v)``
    Insert a new key-value pair into the heap.

``min()``
    Return the entry with the smallest key without removing it.

``remove_min()``
    Remove and return the entry with the smallest key.

``heapify(items)``
    Construct a heap from an iterable of entries using a bottom-up algorithm.

Type Parameters
---------------
K
    Key type. Must support the ``<`` comparison operator.

V
    Value type associated with each key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Self, TypeAlias, TypeVar

from dsa_in_python.common.types import SupportsLessThan
from dsa_in_python.heaps._base import PriorityQueue
from dsa_in_python.heaps._storage import BinaryHeapStore

K = TypeVar('K', bound=SupportsLessThan)
V = TypeVar('V')

# private: internal aliases for type hinting convenience
_HeapItem: TypeAlias = tuple[K, V]
_HeapHandle: TypeAlias = Any


class BinaryHeap(PriorityQueue[K, V]):
    """
    A concrete implementation of the Priority QueueADT using a logical
    *complete binary tree* structure for underlying storage.

    Stores key-value pairs, (K,V), as items on the heap.
    Keys, K, must be comparable (support `<` operator).

    Enforces a min heap order property - the key-value pair (K,V)
    at the top of the heap is guaranteed to be that which has the
    minumum key K.

    Performance:
        add(k, v): O(log*n)
        min(): O(1)
        remove_min(): O(log*n)
        heapify(items): O(n)
    """

    __slots__ = ('_heap_store',)
    _heap_store: BinaryHeapStore[_HeapItem[K, V], _HeapHandle]

    def __init__(self, store_factory: Callable[..., BinaryHeapStore[tuple[K, V], Any]]) -> None:
        """
        Initialize an empty binary heap.

        A new underlying :class:`BinaryHeapStore` is created using the supplied
        ``store_factory``. The store is responsible only for maintaining the
        **complete binary tree shape property**; heap ordering is enforced by the
        heap algorithms implemented in this class.

        The resulting heap contains no elements and therefore trivially satisfies
        the min-heap property.

        Parameters
        ----------
        store_factory:
            A callable that constructs a ``BinaryHeapStore`` implementation used
            to manage the underlying complete binary tree structure.

        Notes
        -----
        Construction is typically **O(1)**, though the exact cost depends on the
        store implementation.
        """
        self._heap_store = store_factory()

    # ----------------- Public API (Priority Queue ADT) ----------------------
    def __len__(self) -> int:
        """
        Return the number of entries currently stored in the heap.

        This method delegates directly to the underlying heap store.

        Returns
        -------
        int
            The number of key-value pairs contained in the heap.

        Complexity
        ----------
        Typically **O(1)** depending on the store implementation.
        """
        return len(self._heap_store)

    def __str__(self) -> str:
        """
        Return a sideways ASCII representation of the heap.

        The root is printed on the left, with right subtrees above and
        left subtrees below. This format makes it easy to visually inspect
        heap structure and ordering.

        Example
        -------
            ┌── 9:a
        ┌── 7:b
        │   └── 8:c
        5:d
        │   ┌── 6:e
        └── 3:f
            └── 4:g
        """
        if self.is_empty():
            return 'BinaryHeap(<empty>)'

        def fmt(entry: _HeapItem[K, V]) -> str:
            """Format a heap entry for display."""
            k, v = entry
            return f'{k}:{v}'

        def render(h: _HeapHandle, prefix: str = '', is_left: bool = True) -> list[str]:
            lines: list[str] = []

            right = self._heap_store.right_child(h)
            if right is not None:
                lines.extend(render(right, prefix + ('│   ' if is_left else '    '), False))

            connector = '└── ' if is_left else '┌── '
            lines.append(prefix + connector + fmt(self._heap_store.get(h)))

            left = self._heap_store.left_child(h)
            if left is not None:
                lines.extend(render(left, prefix + ('    ' if is_left else '│   '), True))

            return lines

        root = self._heap_store.root()
        return '\n'.join(render(root))

    def is_empty(self) -> bool:
        """
        Determine whether the heap contains any entries.

        Returns
        -------
        bool
            ``True`` if the heap contains no elements, otherwise ``False``.

        Complexity
        ----------
        Typically **O(1)**.
        """
        return len(self._heap_store) == 0

    def add(self, k: K, v: V) -> None:
        """
        Insert a new entry into the heap.

        The new ``(key, value)`` pair is first inserted at the last position
        of the underlying complete binary tree to preserve the shape property.
        An *up-heap* operation is then performed to restore the min-heap order.

        Parameters
        ----------
        k:
            Key associated with the entry. Keys must support comparison
            using the ``<`` operator.

        v:
            Value associated with the key.

        Complexity
        ----------
        The up-heap procedure may traverse the height of the tree, giving a
        typical complexity of **O(log n)** where ``n`` is the number of elements
        in the heap. The exact performance depends on the store implementation.
        """

        # insert a new entry in last position
        entry: _HeapItem[K, V] = (k, v)
        last: _HeapHandle = self._heap_store.insert_last(entry)

        # upheap from last position to restore min heap property
        self._up_heap(last)

    def min(self) -> tuple[K, V]:
        """
        Return the entry with the minimum key without removing it.

        The minimum element in a min-heap is always stored at the root position.

        Returns
        -------
        tuple[K, V]
            The ``(key, value)`` pair with the smallest key.

        Raises
        ------
        IndexError
            If the heap is empty.

        Complexity
        ----------
        **O(1)** assuming constant-time access to the root position in the
        underlying store.
        """

        if self.is_empty():
            raise IndexError('min from empty heap')
        root: _HeapHandle = self._heap_store.root()
        return self._heap_store.get(root)

    def remove_min(self) -> tuple[K, V]:
        """
        Remove and return the entry with the smallest key.

        The root entry is swapped with the last position in the heap and then
        removed. A *down-heap* operation is subsequently performed from the root
        to restore the min-heap ordering property.

        Returns
        -------
        tuple[K, V]
            The removed ``(key, value)`` pair with the minimum key.

        Raises
        ------
        IndexError
            If the heap is empty.

        Complexity
        ----------
        The down-heap operation may traverse the height of the tree,
        resulting in **O(log n)** time in typical implementations.
        """

        if self.is_empty():
            raise IndexError('remove_min from empty heap')

        # swap items stored at root and last position
        root: _HeapHandle = self._heap_store.root()
        last: _HeapHandle = self._heap_store.last()
        self._heap_store.swap(root, last)

        # remove last position (store entry for return)
        min_entry: _HeapItem[K, V] = self._heap_store.remove_last()

        # down heap from root to restore min heap property if needed
        if not self.is_empty():
            self._down_heap(root)

        return min_entry

    @classmethod
    def heapify(
        cls,
        store_factory: Callable[[Iterable[tuple[K, V]] | None], BinaryHeapStore[tuple[K, V], Any]],
        items: Iterable[tuple[K, V]],
    ) -> Self:
        """
        Construct a BinaryHeap from an iterable of (key, value) pairs.

        This method builds a heap in two phases:

        1. The provided ``store_factory`` is used to construct a heap store containing
        all items arranged in *level-order* so that the **complete binary tree
        shape property** is satisfied, but without enforcing the heap-order
        property.

        2. The heap-order property is then restored using a bottom-up procedure:
        positions are visited in reverse level-order and ``_down_heap`` is applied
        to each position. This is equivalent to Floyd's bottom-up heap construction
        algorithm.

        Performance
        -----------
        The overall running time depends on the capabilities and performance
        characteristics of the underlying ``BinaryHeapStore`` implementation.

        Typical cases:

        • **Array-backed stores**
        - Building the store from ``items`` is usually O(n).
        - Reverse level-order traversal is O(n).
        - The bottom-up heap construction is O(n).
        - Overall complexity: **O(n)**.

        • **Linked-tree stores**
        - Constructing the complete tree shape may require repeated navigation
            to locate insertion positions (e.g., O(log n) per insertion).
        - In such implementations the overall complexity may degrade to
            **O(n log n)**.

        This method guarantees correct heap construction regardless of store
        implementation, but does **not guarantee O(n)** performance unless the
        store supports efficient level-order construction and traversal.

        Parameters
        ----------
        store_factory:
            A callable that constructs a ``BinaryHeapStore``. When provided with
            ``items``, the store must arrange them in level-order so the complete
            binary tree shape property holds.

        items:
            An iterable of ``(key, value)`` pairs to be inserted into the heap.

        Returns
        -------
        BinaryHeap
            A new heap containing all provided items and satisfying the min-heap
            ordering property.
        """

        heap = cls.__new__(cls)
        heap._heap_store = store_factory(items)  # populate heap store in level-order

        if heap.is_empty():
            return heap

        # enforce min heap order property
        for h in heap._heap_store.handles_reverse_level_order():
            heap._down_heap(h)

        return heap

    # -------------------- Internal Helpers -------------------------
    def _down_heap(self, h: _HeapHandle) -> None:
        """
        Restore the heap-order property by moving an entry downward.

        Starting from position ``h``, the entry is repeatedly compared with
        its children. If either child has a smaller key, the entry is swapped
        with the smaller child and the process continues from the child
        position.

        The procedure stops once the entry is less than or equal to its
        children or when a leaf node is reached.

        Parameters
        ----------
        h:
            Handle identifying the starting position within the heap store.

        Complexity
        ----------
        In the worst case this operation traverses the height of the heap,
        giving **O(log n)** time.
        """

        cur: _HeapHandle = h
        while self._heap_store.has_left_child(cur):
            left = self._heap_store.left_child(cur)
            assert left is not None  # for static type checkers

            small_child = left
            small_key, _ = self._heap_store.get(left)

            right = self._heap_store.right_child(cur)
            if right is not None:
                right_key, _ = self._heap_store.get(right)
                if right_key < small_key:
                    small_child = right
                    small_key = right_key

            cur_key, _ = self._heap_store.get(cur)
            if small_key < cur_key:
                self._heap_store.swap(cur, small_child)
                cur = small_child
            else:
                break

        return

    def _up_heap(self, h: _HeapHandle) -> None:
        """
        Restore the heap-order property by moving an entry upward.

        Starting from position ``h``, the entry is repeatedly compared with
        its parent. If the entry's key is smaller than the parent's key,
        the two entries are swapped and the process continues from the
        parent position.

        The procedure stops once the entry is greater than or equal to its
        parent or when the root is reached.

        Parameters
        ----------
        h:
            Handle identifying the starting position within the heap store.

        Complexity
        ----------
        At most the height of the heap is traversed, giving **O(log n)** time.
        """

        cur: _HeapHandle = h
        while True:
            parent = self._heap_store.parent(cur)
            if parent is None:
                break
            cur_key, _ = self._heap_store.get(cur)
            parent_key, _ = self._heap_store.get(parent)
            if cur_key < parent_key:
                self._heap_store.swap(cur, parent)
                cur = parent
            else:
                break

        return
