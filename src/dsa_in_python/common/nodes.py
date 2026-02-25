"""
This module provides implementations of Linked List Node classes
for convenient use in other modules.

Node: represents a node in a single linked list
DNode: represents a node in a doubly linked list
"""

from __future__ import annotations

from typing import Generic, TypeVar

V = TypeVar('V')


class Node(Generic[V]):
    __slots__ = ('_value', '_next')
    _value: V
    _next: Node[V] | None

    def __init__(self, value: V, next_node: Node[V] | None = None) -> None:
        self._value = value
        self._next = next_node
        return

    @property
    def value(self) -> V:
        return self._value


class DNode(Generic[V]):
    __slots__ = ('_value', '_prev', '_next')
    _value: V
    _prev: Node[V] | None
    _next: Node[V] | None

    def __init__(self, value: V, prev_node: Node[V] | None = None, next_node: Node[V] | None = None) -> None:
        self._value = value
        self._prev = prev_node
        self._next = next_node
        return

    @property
    def value(self) -> V:
        return self._value
