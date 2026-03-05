"""
This module provides varying implementations of the heapsort algorithm.

_heapsort_arr:
    Sort an array in-place using heapsort (max-heap).
    Uses Floyd's classic bottom-up algorithm to first heapify the array.
    O(n*logn) time complexity.
    O(1) auxilliary memory.

_heapsort_arr_keyed:
    Sort an array in-place using heapsort (max-heap).
    Supports a key function for customized comparisons.
    This implementation uses a decorate-sort-undecorate strategy to avoid
    repeated evaluation of ``key`` (at the cost of O(n) additional memory).
    O(n*logn) time complexity.
    O(n) auxilliary memory for decoration pattern.


_heapsort_linked_tree:
    (to be added)
    Implement heap sort algorithm using a linked binary tree data structure

"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from dsa_in_python.common.types import SupportsLessThan

T = TypeVar('T')
K = TypeVar('K', bound=SupportsLessThan)


def _heapsort_arr(arr: list[K]) -> None:
    """Sort ``arr`` in-place using heapsort (max-heap): O(n*logn) time complexity."""
    n = len(arr)
    if n < 2:
        return

    def _downheap(i: int, end: int) -> None:
        """Restore max-heap order for subtree rooted at i."""
        while True:
            left = 2 * i + 1
            if left > end:
                return

            right = left + 1
            largest = left
            if right <= end and arr[right] > arr[left]:
                largest = right

            if arr[i] < arr[largest]:
                arr[i], arr[largest] = arr[largest], arr[i]
                i = largest
            else:
                return

    # Heapify (Floyd bottom-up): build max-heap in O(n)
    for i in range(n // 2 - 1, -1, -1):  # starting from last internal node
        _downheap(i, n - 1)

    # Sortdown: repeatedly move max to end
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        _downheap(0, end - 1)


def heapsort(arr: list[T], key: Callable[[T], K] | None = None, reverse: bool = False) -> None:
    """
    Sort ``arr`` in-place using heapsort.

    This implementation uses a decorate-sort-undecorate strategy to avoid
    repeated evaluation of ``key`` (at the cost of O(n) additional memory).

    Time complexity: O(n log n)
    Extra space: O(n) due to decoration
    """
    n = len(arr)
    if n < 2:
        return

    if key is None:
        key = cast(Callable[[T], K], lambda x: x)

    decorated: list[tuple[K, T]] = [(key(item), item) for item in arr]

    def _downheap(i: int, end: int) -> None:
        while True:
            left = 2 * i + 1
            if left > end:
                return
            right = left + 1

            large = left
            if right <= end and decorated[right][0] > decorated[left][0]:
                large = right

            if decorated[i][0] < decorated[large][0]:
                decorated[i], decorated[large] = decorated[large], decorated[i]
                i = large
            else:
                return

    # heapify
    for i in range(n // 2 - 1, -1, -1):
        _downheap(i, n - 1)

    # sortdown
    for end in range(n - 1, 0, -1):
        decorated[0], decorated[end] = decorated[end], decorated[0]
        _downheap(0, end - 1)

    # undecorate
    for i in range(n):
        arr[i] = decorated[i][1]

    if reverse:
        arr.reverse()
