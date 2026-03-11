"""
This module defines a number of abstract base classes for Tree based data structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, Protocol, TypeVar

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class Position(Protocol[T_co]):
    """Opaque, read-only handle to a location in a tree.

    A Position represents the location of a single element within a specific tree
    container. Positions are created and managed by the tree implementation; users
    should not construct them directly.

    The only guaranteed operation is reading the stored element. A Position is
    intended to be passed back to the same tree instance that produced it.
    """

    @property
    def element(self) -> T_co: ...
    def __eq__(self, other: object) -> bool: ...


class _TreeABC(ABC, Generic[T]):
    """An abstract base class representing a generic tree structure."""

    # -------- abstract methods --------------
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the tree."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def root(self) -> Position[T] | None:
        """Return Position representing the tree's root (or None if empty)."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def parent(self, p: Position[T]) -> Position[T] | None:
        """Return Position representing p's parent (or None if p is root)."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def num_children(self, p: Position[T]) -> int:
        """Return the number of children that Position p has."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def children(self, p: Position[T]) -> Iterator[Position[T]]:
        """Generate an iteration of Positions representing Position p's children."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def is_valid_position(self, p: Position[T]) -> bool:
        """Return True if p is a live position belonging to this tree."""
        raise NotImplementedError('concrete subclass must implement.')

    # ------- concrete methods ------------------
    def is_root(self, p: Position[T]) -> bool:
        """Return True if Position p represents the root of the tree."""
        return self.root() == p

    def is_leaf(self, p: Position[T]) -> bool:
        """Return True if Position p does not have any children."""
        return self.num_children(p) == 0

    def is_empty(self) -> bool:
        """Return True if the tree is empty."""
        return len(self) == 0


class _BinaryTreeABC(_TreeABC[T], ABC):
    """
    Abstract base class representing a binary tree structure. Inherits from _TreeABC.
    Provides template methods for binary tree specific accessors.
    """

    # ------- additional abstract methods -----------
    @abstractmethod
    def left(self, p: Position[T]) -> Position[T] | None:
        """
        Return a Position representing p's left child.
        Returns None if Position p does not have a left child.
        """
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def right(self, p: Position[T]) -> Position[T] | None:
        """
        Return a Position representing p's right child.
        Returns None if Position p does not have a right child.
        """
        raise NotImplementedError('concrete subclass must implement.')

    # ------- concrete methods ----------------------
    def sibling(self, p: Position[T]) -> Position[T] | None:
        """Return a Position representing p's sibling (or None if no sibling)."""
        parent = self.parent(p)
        if parent is None:
            return None

        # p is either a left child or right child of it's parent
        if p == self.left(parent):
            return self.right(parent)
        else:
            return self.left(parent)

    def children(self, p: Position[T]) -> Iterator[Position[T]]:
        """
        Generate an iteration over Positions representing p's children.
        Overrides abstract method in TreeABC
        """
        left = self.left(p)
        right = self.right(p)

        if left is not None:
            yield left
        if right is not None:
            yield right
