"""
This module provides a concrete implementation of a Binary Tree ADT
using linked nodes as the underlying storage structure.
"""

from __future__ import annotations

from typing import Generic, TypeVar, overload

from dsa_in_python.trees._base import Position
from dsa_in_python.trees._mutable_binary_tree import _MutableBinaryTreeABC

T = TypeVar('T')


class _BinaryTreeNode(Generic[T]):
    __slots__ = ('_element', '_parent', '_left_child', '_right_child')
    _element: T
    _parent: _BinaryTreeNode[T] | None
    _left_child: _BinaryTreeNode[T] | None
    _right_child: _BinaryTreeNode[T] | None

    def __init__(self, element: T, parent: _BinaryTreeNode[T] | None = None) -> None:
        self._element = element
        self._parent = parent
        self._left_child = None
        self._right_child = None

    @property
    def element(self) -> T:
        return self._element


class _Position(Position[T]):
    __slots__ = ('_container', '_node')
    _container: LinkedBinaryTree[T]
    _node: _BinaryTreeNode[T]

    def __init__(self, container: LinkedBinaryTree[T], node: _BinaryTreeNode[T]) -> None:
        """
        Constructor should not be invoked by user.
        LinkedBinaryTree class is responsible for making position instances.
        """
        self._container = container
        self._node = node

    def __eq__(self, other: object) -> bool:
        """Return True if other Position represents the same location of the same container."""
        if not isinstance(other, _Position):
            return False
        # return other._node is self._node
        return (self._container is other._container) and (self._node is other._node)

    @property
    def element(self) -> T:
        """Return the element stored at this position"""
        return self._node.element


class LinkedBinaryTree(_MutableBinaryTreeABC[T]):
    # class LinkedBinaryTree(_BinaryTreeABC[T]):
    _root: _BinaryTreeNode[T] | None
    _size: int

    # --------------------- public API ---------------------------
    def __init__(self) -> None:
        self._root = None
        self._size = 0

    # Accessors
    def __len__(self) -> int:
        """Return the total number of elements in the tree."""
        return self._size

    def root(self) -> Position[T] | None:
        """Return Position representing the tree's root (or None if empty)."""
        return self._make_position(self._root)

    def is_root(self, p: Position[T]) -> bool:
        """Return True if Position p represents the root of this tree."""
        node = self._validate_position(p)
        return node is self._root

    def parent(self, p: Position[T]) -> Position[T] | None:
        """Return Position representing p's parent (or None if p is root)."""
        node = self._validate_position(p)
        return self._make_position(node._parent)

    def left(self, p: Position[T]) -> Position[T] | None:
        """
        Return a Position representing p's left child.
        Returns None if Position p does not have a left child.
        """
        node = self._validate_position(p)
        return self._make_position(node._left_child)

    def right(self, p: Position[T]) -> Position[T] | None:
        """
        Return a Position representing p's right child.
        Returns None if Position p does not have a right child.
        """
        node = self._validate_position(p)
        return self._make_position(node._right_child)

    def num_children(self, p: Position[T]) -> int:
        """Return the number of children that Position p has."""
        node = self._validate_position(p)
        cnt = 0
        if node._left_child is not None:
            cnt += 1
        if node._right_child is not None:
            cnt += 1
        return cnt

    # Mutators
    def add_root(self, e: T) -> Position[T]:
        """
        Inserts e at the root of an empty tree.

        Return the Position of the root.
        Raise ValueError if tree is not empty.
        """
        if self._root is not None:
            raise ValueError('Root exists')
        self._size = 1
        self._root = _BinaryTreeNode(e)
        return self._make_position(self._root)

    def add_left(self, p: Position[T], e: T) -> Position[T]:
        """
        Creates a left child for Position p, storing element e.

        Return the position of the left child.
        Raise ValueError if position p is invalid or p already has a left child
        """
        node = self._validate_position(p)
        if node._left_child is not None:
            raise ValueError('Left child exists')
        self._size += 1
        node._left_child = _BinaryTreeNode(e, node)
        return self._make_position(node._left_child)

    def add_right(self, p: Position[T], e: T) -> Position[T]:
        """
        Creates a right child for Position p, storing element e.

        Return the position of the right child.
        Raise ValueError is positon p is invalud or p already has a right child.
        """
        node = self._validate_position(p)
        if node._right_child is not None:
            raise ValueError('Right child exists')
        self._size += 1
        node._right_child = _BinaryTreeNode(e, node)
        return self._make_position(node._right_child)

    def replace(self, p: Position[T], e: T) -> T:
        """
        Replace the element stored at Position p with e.

        Return the old element stored at Position p.
        Raise ValueError if Position p is invalid.
        """
        node = self._validate_position(p)
        old = node.element
        node._element = e
        return old

    def delete(self, p: Position[T]) -> T:
        """
        Delete Position p, and replace it with its child if any.
        Permits root deletion if root has 0 or 1 children.

        Return the element that had been stored in Position p.
        Raise ValueError if Position p has two children.
        """
        node = self._validate_position(p)
        if self.num_children(p) == 2:
            raise ValueError('Position has two children, cannot delete')

        child = node._left_child if node._left_child else node._right_child

        # handle root deletion
        if node is self._root:
            self._root = child  # child becomes root
            if child is not None:
                child._parent = None
        else:
            parent = node._parent
            assert parent is not None  # for type checker
            if node is parent._left_child:
                parent._left_child = child
            else:
                parent._right_child = child
            if child is not None:
                child._parent = parent

        self._size -= 1
        old = node.element
        self._deprecate_node(node)
        return old

    def attach(self, p: Position[T], t1: _MutableBinaryTreeABC[T], t2: _MutableBinaryTreeABC[T]) -> None:
        """
        Attach trees t1 and t2 as left and right subtrees of external Position p.

        Returns None
        Raise ValueError is Position p is invalid or is not a leaf of the tree.
        Raise TypeError if t1 and t2 are not trees of the same type as self.
        """

        # Runtime type safety check
        if type(t1) is not type(self) or type(t2) is not type(self):
            raise TypeError('Trees must be the same concrete type as this tree.')

        # SAFETY: Strengthen attach() safety.
        # Currently attach() does not invalidate donor tree positions.
        # Existing Position instances from t1/t2 can still mutate nodes
        # now owned by this tree. Introduce ownership or versioning to
        # prevent cross-tree mutation.

        node = self._validate_position(p)
        if not self.is_leaf(p):
            raise ValueError('position p must be a leaf')

        # Defensive check: all three trees must be of same type.
        if not type(self) is type(t1) is type(t2):
            raise TypeError('Tree types must match')

        self._size += len(t1) + len(t2)

        # attach t1 as left subtree of node
        if not t1.is_empty():
            assert t1._root is not None  # for type checkers
            t1._root._parent = node
            node._left_child = t1._root
            t1._root = None  # set t1 instance to empty
            t1._size = 0

        if not t2.is_empty():
            assert t2._root is not None  # for type checkers
            t2._root._parent = node
            node._right_child = t2._root
            t2._root = None  # set t2 instance to empty
            t2._size = 0
        return

    # ------------------- internal helpers ------------------------
    def _validate_position(self, p: Position[T]) -> _BinaryTreeNode[T]:
        """Return associated node, if Position p is valid."""
        if not isinstance(p, _Position):
            raise TypeError('p must be a proper Position type (expected a Position created by this tree).')
        if p._container is not self:
            raise ValueError('p does not belong to this container.')
        if p._node._parent is p._node:  # convention for deprecated nodes
            raise ValueError('p is no longer valid')
        return p._node

    # overloads for static type checking (no runtime effect!)
    @overload
    def _make_position(self, node: None) -> None: ...
    @overload
    def _make_position(self, node: _BinaryTreeNode[T]) -> _Position[T]: ...

    def _make_position(self, node: _BinaryTreeNode[T] | None) -> _Position[T] | None:
        """Return Position instance for given node (or None if no node)."""
        return _Position(self, node) if node is not None else None

    def _deprecate_node(self, node: _BinaryTreeNode[T]) -> None:
        """Deprecate a node using convention that parent points to node itself."""
        node._parent = node
