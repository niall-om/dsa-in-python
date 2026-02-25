"""
This module defines a number of abstract base classes for Tree based data structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

T = TypeVar('T')


class _PositionABC(ABC, Generic[T]):
    """An abstraction representing the location of a single element in the Tree."""

    @property
    @abstractmethod
    def element(self) -> T:
        """Return the element stored at this position"""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Return True if other Position represents the same location."""
        raise NotImplementedError('concrete subclass must implement.')

    def __ne__(self, other: object) -> bool:
        """Return True if other Position does not represent the same location."""
        return not (self == other)


P = TypeVar('P', bound=_PositionABC[Any])


# Tree ABC
class _TreeABC(ABC, Generic[T, P]):
    """An abstract base class representing a generic tree structure."""

    # -------- abstract methods --------------
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements in the tree."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def root(self) -> P | None:
        """Return Position representing the tree's root (or None if empty)."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def parent(self, p: P) -> P | None:
        """Return Position representing p's parent (or None if p is root)."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def num_children(self, p: P) -> int:
        """Return the number of children that Position p has."""
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def children(self, p: P) -> Iterator[P]:
        """Generate an iteration of Positions representing Position p's children."""
        raise NotImplementedError('concrete subclass must implement.')

    # ------- concrete methods ------------------
    def is_root(self, p: P) -> bool:
        """Return True if Position p represents the root of the tree."""
        return self.root() == p

    def is_leaf(self, p: P) -> bool:
        """Return True if Position p does not have any children."""
        return self.num_children(p) == 0

    def is_empty(self) -> bool:
        """Return True if the tree is empty."""
        return len(self) == 0


# Binary Tree ABC
class _BinaryTreeABC(_TreeABC[T, P]):
    """
    Abstract base class representing a binary tree structure. Inherits from _TreeABC.
    Provides template methods for binary tree specific accessors.
    """

    # ------- additional abstract methods -----------
    @abstractmethod
    def left(self, p: P) -> P | None:
        """
        Return a Position representing p's left child.
        Returns None if Position p does not have a left child.
        """
        raise NotImplementedError('concrete subclass must implement.')

    @abstractmethod
    def right(self, p: P) -> P | None:
        """
        Return a Position representing p's right child.
        Returns None if Position p does not have a right child.
        """
        raise NotImplementedError('concrete subclass must implement.')

    # ------- concrete methods ----------------------
    def sibling(self, p: P) -> P | None:
        """Return a Position representing p's sibling (or None if no sibling)."""
        parent = self.parent(p)
        if parent is None:
            return None

        # p is either a left child or right child of it's parent
        if p == self.left(parent):
            return self.right(parent)
        else:
            return self.left(parent)

    def children(self, p: P) -> Iterator[P]:
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
