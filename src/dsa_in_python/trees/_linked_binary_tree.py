"""
This module provides a concrete implementation of a Binary Tree ADT
using linked nodes as the underlying storage structure.
"""

from __future__ import annotations

from typing import Generic, TypeVar, cast

from dsa_in_python.trees._base import _BinaryTreeABC, _PositionABC

T = TypeVar('T', bound=object)


class _BinaryTreeNode(Generic[T]):
    __slots__ = ('_element', '_parent', '_left_child', '_right_child')
    _element: T
    _parent: _BinaryTreeNode[T] | None
    _left_child: _BinaryTreeNode[T] | None
    _right_child: _BinaryTreeNode[T] | None

    def __init__(self, element: T, parent: _BinaryTreeNode[T] | None = None) -> None:
        self._element = element
        self._parent = parent

    @property
    def element(self) -> T:
        return self._element

    @property
    def parent(self) -> _BinaryTreeNode[T] | None:
        return self._parent

    @property
    def left_child(self) -> _BinaryTreeNode[T] | None:
        return self._left_child

    @property
    def right_child(self) -> _BinaryTreeNode[T] | None:
        return self._right_child


class _Position(_PositionABC[T]):
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
        """Return True if other Position represents the same location."""
        return type(other) is type(self) and cast(_Position[T], other)._node is self._node

    @property
    def element(self) -> T:
        """Return the element stored at this position"""
        return self._node.element

    @property
    def node(self) -> _BinaryTreeNode[T]:
        return self._node

    @property
    def container(self) -> LinkedBinaryTree[T]:
        return self._container


class LinkedBinaryTree(_BinaryTreeABC[T, _Position[T]], Generic[T]):
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

    def root(self) -> _Position[T] | None:
        """Return Position representing the tree's root (or None if empty)."""
        return self._make_position(self._root)

    def parent(self, p: _Position[T]) -> _Position[T] | None:
        """Return Position representing p's parent (or None if p is root)."""
        node = self._validate_position(p)
        return self._make_position(node.parent)

    def left(self, p: _Position[T]) -> _Position[T] | None:
        """
        Return a Position representing p's left child.
        Returns None if Position p does not have a left child.
        """
        node = self._validate_position(p)
        return self._make_position(node.left_child)

    def right(self, p: _Position[T]) -> _Position[T] | None:
        """
        Return a Position representing p's right child.
        Returns None if Position p does not have a right child.
        """
        node = self._validate_position(p)
        return self._make_position(node.right_child)

    def num_children(self, p: _Position[T]) -> int:
        """Return the number of children that Position p has."""
        node = self._validate_position(p)
        cnt = 0
        if node.left_child is not None:
            cnt += 1
        if node.right_child is not None:
            cnt += 1
        return cnt

    # Mutators
    def add_root(self, e: T) -> _Position[T]:
        """
        Inserts e at the root of an empty tree.

        Return the Position of the root.
        Raise ValueError if tree is not empty.
        """
        if self._root is not None:
            raise ValueError('Root exists')
        self._size = 1
        self._root = _BinaryTreeNode(e)
        return cast(_Position[T], self._make_position(self._root))

    def add_left(self, p: _Position[T], e: T) -> _Position[T]:
        """
        Creates a left child for Position p, storing element e.

        Return the position of the left child.
        Raise ValueError if position p is invalid or p already has a left child
        """
        node = self._validate_position(p)
        if node.left_child is not None:
            raise ValueError('Left child exists')
        self._size += 1
        node._left_child = _BinaryTreeNode(e, node)
        return cast(_Position[T], self._make_position(node.left_child))

    def add_right(self, p: _Position[T], e: T) -> _Position[T]:
        """
        Creates a right child for Position p, storing element e.

        Return the position of the right child.
        Raise ValueError is positon p is invalud or p already has a right child.
        """
        node = self._validate_position(p)
        if node.right_child is not None:
            raise ValueError('Right child exists')
        self._size += 1
        node._right_child = _BinaryTreeNode(e, node)
        return cast(_Position[T], self._make_position(node.right_child))

    def replace(self, p: _Position[T], e: T) -> T:
        """
        Replace the element stored at Position p with e.

        Return the old element stored at Position p.
        Raise ValueError if Position p is invalid.
        """
        node = self._validate_position(p)
        if node is None:
            raise ValueError('Position is no longer valid.')
        old = node.element
        node._element = e
        return old

    def delete(self, p: _Position[T]) -> T:
        """
        Delete Position p, and replace it with its child if any.
        Permits root deletion if root has 0 or 1 children.

        Return the element that had been stored in Position p.
        Raise ValueError if Position p has two children.
        """
        node = self._validate_position(p)
        if self.num_children(p) == 2:
            raise ValueError('Position has two children, cannot delete')

        child = node.left_child if node.left_child else node.right_child

        # handle root deletion
        if node is self._root:
            self._root = child  # child becomes root
        else:
            parent = node.parent
            assert parent is not None  # for type checker
            if node is parent.left_child:
                parent._left_child = child
            else:
                parent._right_child = child

        self._size -= 1
        self._deprecate_node(node)
        return node.element

    def attach(self, p: _Position[T], t1: LinkedBinaryTree[T], t2: LinkedBinaryTree[T]) -> None:
        """
        Attach trees t1 and t2 as left and right subtrees of external Position p.

        Returns None
        Raise ValueError is Position p is invalid or is not a leaf of the tree.
        Raise TypeError if t1 and t2 are not trees of the same type as self.
        """

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
    def _validate_position(self, p: _Position[T]) -> _BinaryTreeNode[T]:
        """Return associated node, if Position p is valid."""
        if not isinstance(p, _Position):
            raise TypeError('p must be a proper _Position type.')
        if p.container is not self:
            raise ValueError('p does not belong to this container.')
        if p.node.parent is p.node:  # convention for deprecated nodes
            raise ValueError('p is no longer valid')

        return p.node

    def _make_position(self, node: _BinaryTreeNode[T] | None) -> _Position[T] | None:
        """Return Position instance for given node (or None if no node)."""
        return _Position(self, node) if node is not None else None

    def _deprecate_node(self, node: _BinaryTreeNode[T]) -> None:
        """Deprecate a node using convention that parent points to node itself."""
        node._parent = node
