"""
Linked-tree implementation of the BinaryHeapStore protocol.

This module provides a concrete storage backend for a binary heap using a
linked binary tree as the underlying data structure. The implementation
satisfies the structural requirements of the ``BinaryHeapStore`` protocol:
it maintains the *complete binary tree* shape property but does not enforce
any heap ordering. Enforcement of the heap-order property is delegated to
the heap algorithm itself.

The storage layer is implemented using a ``LinkedBinaryTree``. Because linked
trees do not provide direct index-based access to nodes, this implementation
maintains an auxiliary ``_level_order`` list that stores the tree positions
in breadth-first order. This list acts as an address table that allows the
heap algorithm to efficiently access the root, the last node, and perform
level-order traversals required by heap construction and maintenance.

Heap positions are exposed through opaque ``_HeapHandle`` objects rather than
directly exposing the tree's ``Position`` objects. This provides a clean
abstraction boundary between the heap storage layer and the underlying tree
implementation. Each handle internally wraps a tree position and includes a
reference to its owning store so that handle validity can be verified.

The store enforces the following invariant:

    ``_level_order[i]`` corresponds to the i-th node in the tree's level-order
    traversal.

Structural mutations such as ``insert_last`` and ``remove_last`` update both
the underlying tree and the ``_level_order`` list so that this invariant
remains valid.

This design favors clarity and correctness over minimal memory usage.
The auxiliary level-order index requires O(n) additional space but allows
efficient structural operations on top of a pointer-based tree representation.

Future improvements may extract the complete-tree maintenance logic into a
dedicated ``CompleteLinkedBinaryTree`` abstraction that manages both the
linked structure and level-order indexing in a unified data structure.
"""

# TODO: Refactor to CompleteLinkedBinaryTree abstraction (see DESIGN note below).
# TODO: Normalize error messages for the BinaryHeapStore protocol.

# DESIGN:
# This store currently maintains two coupled structures:
#   1) `_tree`        — the underlying LinkedBinaryTree storing heap elements
#   2) `_level_order` — an auxiliary list of tree positions in breadth-first order
#
# The `_level_order` list acts as an address table that enables efficient
# heap-style operations (root, last, level-order traversal) on top of the
# pointer-based tree structure.
#
# However, `_tree` and `_level_order` conceptually represent a single logical
# data structure: a *complete binary tree*. Maintaining them separately means
# that structural operations must carefully update both structures to preserve
# the invariant:
#
#     _level_order[i] == the i-th node in the tree's level-order traversal
#
# A cleaner design would introduce a dedicated `CompleteLinkedBinaryTree`
# abstraction that encapsulates both the linked node structure and the
# level-order index as a single atomic data structure. The heap store would
# then depend on that abstraction instead of directly coordinating the two
# structures itself.
#
# Such a refactor would:
#   - centralize complete-tree invariants
#   - eliminate the risk of `_tree` and `_level_order` becoming inconsistent
#   - simplify the heap store implementation
#   - make the complete binary tree reusable for other algorithms

# DESIGN:
# This store currently depends directly on the concrete LinkedBinaryTree
# implementation. This is acceptable for the initial design, but it creates
# tighter coupling than strictly necessary.
#
# In the future, this class could be generalized to depend only on the
# `_MutableBinaryTreeABC` abstraction, allowing any compatible mutable
# binary tree implementation to serve as the underlying storage.
#
# This would improve flexibility and decouple the heap storage layer from
# a specific tree implementation, but is not required for correctness and
# therefore remains a potential refactor rather than an immediate `TODO`.

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Generic, TypeVar, overload

from dsa_in_python.heaps._storage import BinaryHeapStore
from dsa_in_python.trees._base import Position
from dsa_in_python.trees._linked_binary_tree import LinkedBinaryTree

HeapItem = TypeVar('HeapItem')


class _HeapHandle(Generic[HeapItem]):
    __slots__ = ('_store', '_position')
    _store: LinkedTreeBinaryHeapStore[HeapItem]
    _position: Position[HeapItem]

    def __init__(self, store: LinkedTreeBinaryHeapStore[HeapItem], position: Position[HeapItem]) -> None:
        self._store = store
        self._position = position


class LinkedTreeBinaryHeapStore(BinaryHeapStore[HeapItem, _HeapHandle[HeapItem]]):
    """
    Binary heap storage backend implemented using a linked binary tree.

    This class provides a concrete implementation of the ``BinaryHeapStore``
    protocol where heap elements are stored in a ``LinkedBinaryTree`` while
    maintaining the *complete binary tree* structural property required by
    heap data structures.

    The store itself is responsible only for maintaining the tree shape.
    It does **not** enforce the heap-order property; that responsibility
    belongs to the heap algorithm that uses this store.

    Because linked trees do not support direct index-based navigation,
    the implementation maintains an auxiliary list ``_level_order`` that
    stores tree positions in breadth-first order. This list serves as an
    addressing structure that allows efficient access to:

    - the root node
    - the last node in level order
    - level-order and reverse level-order traversals

    Structural operations update both the underlying tree and the level-order
    index to maintain the invariant that ``_level_order`` mirrors the tree's
    breadth-first traversal.

    Heap positions are exposed as opaque ``_HeapHandle`` objects rather than
    raw tree positions. A handle wraps a tree position and includes a reference
    to the owning store, allowing the store to validate that handles:

    - are of the correct type
    - belong to this store instance
    - reference a still-valid tree position

    This encapsulation prevents external code from directly manipulating the
    underlying tree structure and ensures that heap operations interact with
    the storage layer through a well-defined interface.

    Complexity
    ----------
    Structural operations have the following asymptotic costs:

    - ``root`` / ``last`` : O(1)
    - ``insert_last``     : O(log n)
    - ``remove_last``     : O(1)
    - navigation methods  : O(1)

    The ``insert_last`` operation locates the parent of the next insertion
    point by interpreting the binary representation of the next node index
    and performing a corresponding walk from the root of the tree.

    Notes
    -----
    This implementation intentionally favors simplicity and clarity over
    optimal memory usage. The additional O(n) storage required for the
    level-order index simplifies navigation and helps maintain the complete
    binary tree invariant for a pointer-based tree representation.
    """

    __slots__ = ('_tree', '_level_order')
    _tree: LinkedBinaryTree[HeapItem]
    _level_order: list[Position[HeapItem]]

    def __init__(self, items: Iterable[HeapItem] | None) -> None:
        """
        Constructs a HeapStore.

        If optional arg `items` is provided, constructs a heap satisfying
        the *complete binary tree* shape property. It does NOT enforce the heap
        ordering property - that responsibility belongs to the heap algorithm.
        """

        self._tree = LinkedBinaryTree()
        self._level_order = []

        if items is not None:
            self._build_from_array(items)

    def __len__(self) -> int:
        """Return the number of elements currently stored."""
        return len(self._level_order)

    def is_empty(self) -> bool:
        """Return True if the store is empty."""
        return len(self._level_order) == 0

    # ------------- Accessors ------------------
    def root(self) -> _HeapHandle[HeapItem]:
        """
        Return the handle of the root position.

        Returns
        -------
        _HeapHandle
            Handle identifying the root position.

        Raises
        ------
        IndexError
            If the store is empty.
        """
        if not self._level_order:
            raise IndexError('heap store is empty')

        root = self._tree.root()
        if root is None:
            raise RuntimeError('heap store invariant broken: non-empty store has no root')
        return self._make_handle(root)

    def last(self) -> _HeapHandle[HeapItem]:
        """
        Return the handle of the last (most recently inserted) position
        in level-order.

        This corresponds to the deepest, rightmost node in the complete tree.

        Raises:
            IndexError: if the store is empty.
        """
        if not self._level_order:
            raise IndexError('heap store is empty')
        return self._make_handle(self._level_order[-1])

    def parent(self, h: _HeapHandle[HeapItem]) -> _HeapHandle[HeapItem] | None:
        """
        Return the handle of the parent of the position given by handle `h`.

        Returns:
            The parent handle, or None if the position is the root.

        Raises:
            ValueError: if handle `h` is invalid or no longer exists.
        """
        # delegate validation; validator raises ValueError for invalid handles
        pos: Position[HeapItem] = self._validate_handle(h)
        return self._make_handle(self._tree.parent(pos))

    def left_child(self, h: _HeapHandle[HeapItem]) -> _HeapHandle[HeapItem] | None:
        """
        Return the handle of the left child of the given position.

        Returns:
            The left child handle, or None if no left child exists.

        Raises:
            ValueError: if handle `h` is invalid or no longer exists.
        """
        # delegate validation; validator raises ValueError for invalid handles
        pos: Position[HeapItem] = self._validate_handle(h)
        return self._make_handle(self._tree.left(pos))

    def right_child(self, h: _HeapHandle[HeapItem]) -> _HeapHandle[HeapItem] | None:
        """
        Return the handle of the right child of the given position.

        Returns:
            The right child handle, or None if no right child exists.

        Raises:
            ValueError: if handle `h` is invalid or no longer exists.
        """
        # delegate validation; validator raises ValueError for invalid handles
        pos: Position[HeapItem] = self._validate_handle(h)
        return self._make_handle(self._tree.right(pos))

    def has_left_child(self, h: _HeapHandle[HeapItem]) -> bool:
        """Return True if position `h` has a left child."""
        pos = self._validate_handle(h)
        return self._tree.left(pos) is not None

    def has_right_child(self, h: _HeapHandle[HeapItem]) -> bool:
        """Return True if position `h` has a right child."""
        pos = self._validate_handle(h)
        return self._tree.right(pos) is not None

    def get(self, h: _HeapHandle[HeapItem]) -> HeapItem:
        """
        Return the entry stored at position `h` in the heap.

        Raises:
            ValueError: if the handle `h` is invalid or no longer exists.
        """
        pos = self._validate_handle(h)
        return pos.element

    def handles_level_order(self) -> Iterable[_HeapHandle[HeapItem]]:
        """Yield handles in level-order (breadth first), from root to last."""
        for pos in self._level_order:
            yield self._make_handle(pos)

    def handles_reverse_level_order(self) -> Iterable[_HeapHandle[HeapItem]]:
        """Yield handles in reverse level-order, from last to root."""
        for pos in reversed(self._level_order):
            yield self._make_handle(pos)

    # -------------- Mutators -------------------
    def set(self, h: _HeapHandle[HeapItem], item: HeapItem) -> None:
        pos = self._validate_handle(h)
        self._tree.replace(pos, item)

    def insert_last(self, item: HeapItem) -> _HeapHandle[HeapItem]:
        """
        Insert a new element at the last position in the complete tree.

        This preserves the structural (completeness) property.

        Returns:
            The handle of the newly inserted position.
        """
        return self._insert_last(item)

    def remove_last(self) -> HeapItem:
        """
        Remove and return the value stored at the last position
        in the complete tree.

        This preserves the structural (completeness) property.

        Raises:
            IndexError: if the store is empty.
        """
        if not self._level_order:
            raise IndexError('heap store is empty')
        return self._remove_last()

    def swap(self, h1: _HeapHandle[HeapItem], h2: _HeapHandle[HeapItem]) -> None:
        """
        Exchange the items stored at two positions `h1` and `h2`.

        This operation is used by the heap algorithm to restore
        the heap-order property during upheap and downheap operations.

        Returns: None

        Raises:
            ValueError: if either handle `h1` or `h2` is invalid or no longer exists.
        """

        p1 = self._validate_handle(h1)
        p2 = self._validate_handle(h2)

        # fast return path
        if p1 == p2:
            return

        p2_old_item = self._tree.replace(p2, p1.element)
        self._tree.replace(p1, p2_old_item)

    @classmethod
    def from_array(cls, items: Iterable[HeapItem]) -> LinkedTreeBinaryHeapStore[HeapItem]:
        """
        Build a complete binary tree store from ``items`` in level order.

        Time complexity: O(n)
        Auxiliary space: O(n)
        """

        heap = cls.__new__(cls)
        heap._tree = LinkedBinaryTree()
        heap._level_order = []
        heap._build_from_array(items)
        return heap

    # ----------------- Internal Helpers ------------------------
    def _validate_handle(self, h: _HeapHandle[HeapItem]) -> Position[HeapItem]:
        if not isinstance(h, _HeapHandle):
            raise ValueError('h must be a proper Handle type')
        if h._store is not self:
            raise ValueError('handle does not belong to this heap store')

        # ask underlying tree to validate the wrapped position
        if not self._tree.is_valid_position(h._position):
            raise ValueError('handle is no longer valid')

        return h._position

    @overload
    def _make_handle(self, p: Position[HeapItem]) -> _HeapHandle[HeapItem]: ...
    @overload
    def _make_handle(self, p: None) -> None: ...

    def _make_handle(self, p: Position[HeapItem] | None) -> _HeapHandle[HeapItem] | None:
        return _HeapHandle(self, p) if p is not None else None

    def _insert_last(self, item: HeapItem) -> _HeapHandle[HeapItem]:
        """
        Insert ``item`` at the last position of the complete binary tree.

        The insertion position is determined by interpreting the 1-based index
        of the next node in binary and following the corresponding path from
        the root to the parent of that node.

        The newly created position is appended to ``_level_order`` so that the
        level-order index structure remains consistent with the underlying tree.

        Raises:
            RuntimeError: if heap store invariant (complete binary tree shape) is broken.
        """

        if not self._level_order:
            new_root = self._tree.add_root(item)
            self._level_order.append(new_root)
            return self._make_handle(new_root)

        # start search from root
        root = self._tree.root()
        if root is None:
            raise RuntimeError('heap store invariant broken: non-empty store has no root')
        cur: Position[HeapItem] = root

        # 1-based index of the next node to insert
        next_idx = len(self._level_order) + 1

        # Find path to parent of next_idx
        k = next_idx.bit_length() - 1
        power = 1 << k
        remainder = next_idx - power
        power >>= 1  # skip MSB; root is already selected

        # stop one step early: we want the parent, not the insertion node itself
        while power > 1:
            if remainder < power:  # go left
                left = self._tree.left(cur)
                if left is None:
                    raise RuntimeError('heap store invariant broken: expected left child on insertion path')
                cur = left
            else:
                remainder -= power
                right = self._tree.right(cur)
                if right is None:
                    raise RuntimeError('heap store invariant broken: expected right child on insertion path')
                cur = right
            power >>= 1

        if self._tree.left(cur) is None:
            last = self._tree.add_left(cur, item)
        elif self._tree.right(cur) is None:
            last = self._tree.add_right(cur, item)
        else:
            raise RuntimeError('heap store invariant broken: insertion parent already has two children')

        self._level_order.append(last)
        return self._make_handle(last)

    def _remove_last(self) -> HeapItem:
        last = self._level_order.pop()
        return self._tree.delete(last)

    def _build_from_array(self, items: Iterable[HeapItem]) -> None:
        """
        Build a complete binary tree store from ``items`` in level order.

        Items are inserted top-down, left-to-right, so the resulting linked tree
        satisfies the complete binary tree shape property

        Time complexity: O(n)
        Auxiliary space: O(n)
        """

        if self._level_order:
            raise ValueError('heap store is not empty')

        parents: deque[Position[HeapItem]] = deque()

        for item in items:
            if not parents:
                root = self._tree.add_root(item)
                self._level_order.append(root)
                parents.append(root)
            else:
                parent = parents[0]
                if self._tree.left(parent) is None:
                    child = self._tree.add_left(parent, item)

                else:
                    child = self._tree.add_right(parent, item)
                    parents.popleft()

                self._level_order.append(child)
                parents.append(child)

        return
