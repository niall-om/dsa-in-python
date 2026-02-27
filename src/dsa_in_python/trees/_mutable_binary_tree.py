"""
This module defines an abstract base class for mutable Binary Trees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from dsa_in_python.trees._base import Position, _BinaryTreeABC

T = TypeVar('T')


class _MutableBinaryTreeABC(_BinaryTreeABC[T], ABC):
    @abstractmethod
    def add_root(self, e: T) -> Position[T]:
        raise NotImplementedError('concrete subclasses must implement')

    @abstractmethod
    def add_left(self, p: Position[T], e: T) -> Position[T]:
        raise NotImplementedError('concrete subclasses must implement')

    @abstractmethod
    def add_right(self, p: Position[T], e: T) -> Position[T]:
        raise NotImplementedError('concrete subclasses must implement')

    @abstractmethod
    def replace(self, p: Position[T], e: T) -> T:
        raise NotImplementedError('concrete subclasses must implement')

    @abstractmethod
    def delete(self, p: Position[T]) -> T:
        raise NotImplementedError('concrete subclasses must implement')

    @abstractmethod
    def attach(self, p: Position[T], t1: _MutableBinaryTreeABC[T], t2: _MutableBinaryTreeABC[T]) -> None:
        raise NotImplementedError('concrete subclasses must implement')
