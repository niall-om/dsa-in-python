"""
Merge algorithm implementations.

This module contains internal implementations of the Merge sorting
algorithm, covering multiple design variants and trade-offs.

The implementations here may differ along dimensions such as:
- data structure (e.g. Python lists, linked lists)
- mutability (in-place vs out-of-place)
- recursive vs iterative approaches

These implementations are considered internal to the
``dsa_in_python.sorting`` subpackage and are not part of the public API.
Public-facing access to Quicksort functionality is provided via
``dsa_in_python.sorting``.

Design notes:
- All implementations aim for clarity and correctness first.
- Performance optimizations are included only where they preserve
  readability and algorithmic intent.
- Space complexity depends on the specific variant.

This module is intended for educational and comparative purposes.
"""

from __future__ import annotations

from typing import TypeVar, cast

from dsa_in_python.common.types import SupportsLessThan

T = TypeVar('T', bound=SupportsLessThan)


def _mergesort_arr_recursive(arr: list[T]) -> None:
    if not arr or len(arr) == 1:
        return None

    n = len(arr)
    merge_buffer: list[T | None] = [None] * n

    def _merge(arr: list[T], l_idx: int, mid: int, r_idx: int) -> None:
        read_l = l_idx
        read_r = mid + 1
        write = l_idx

        while read_l <= mid and read_r <= r_idx:
            if arr[read_l] < arr[read_r]:
                merge_buffer[write] = arr[read_l]
                read_l += 1
            else:
                merge_buffer[write] = arr[read_r]
                read_r += 1
            write += 1

        while read_l <= mid:
            merge_buffer[write] = arr[read_l]
            read_l += 1
            write += 1

        while read_r <= r_idx:
            merge_buffer[write] = arr[read_r]
            read_r += 1
            write += 1

        # copy merge buff to arr
        arr[l_idx : r_idx + 1] = cast(list[T], merge_buffer[l_idx : r_idx + 1])

        return

    def _merge_sort(arr: list[T], l_idx: int, r_idx: int) -> None:
        # base case: subarray size <= 1
        if r_idx <= l_idx:
            return

        # partition in two (deterministic split)
        mid = (l_idx + r_idx) // 2

        # sort left and right subarrays recursively
        _merge_sort(arr, l_idx, mid)
        _merge_sort(arr, mid + 1, r_idx)

        # merge the sorted subarrays
        _merge(arr, l_idx, mid, r_idx)

        return

    return _merge_sort(arr, 0, n - 1)


def _mergesort_arr_iterative(arr: list[T]) -> None:
    if not arr or len(arr) == 1:
        return

    n = len(arr)
    merge_buffer: list[T | None] = [None] * n

    def _merge(arr: list[T], l_idx: int, mid: int, r_idx: int) -> None:
        read_l = l_idx
        read_r = mid + 1
        write = l_idx

        while read_l <= mid and read_r <= r_idx:
            if arr[read_l] < arr[read_r]:
                merge_buffer[write] = arr[read_l]
                read_l += 1
            else:
                merge_buffer[write] = arr[read_r]
                read_r += 1
            write += 1

        while read_l <= mid:
            merge_buffer[write] = arr[read_l]
            read_l += 1
            write += 1

        while read_r <= r_idx:
            merge_buffer[write] = arr[read_r]
            read_r += 1
            write += 1

        arr[l_idx : r_idx + 1] = cast(list[T], merge_buffer[l_idx : r_idx + 1])

        return

    # initialise segement size
    seg_size = 1
    while seg_size < n:
        # merge (and sort) each sequential pair of subarrays of size seg_size
        for l_idx in range(0, n, seg_size * 2):
            mid = l_idx + seg_size - 1

            # check if we actually have a right subarray to merge
            if mid < n - 1:
                r_idx = min(mid + seg_size, n - 1)  # prevent out of bound indexing
                _merge(arr, l_idx, mid, r_idx)

        seg_size *= 2
    return


if __name__ == '__main__':
    import random

    test_arrs: list[list[int]] = [[random.randint(-100, 100) for _ in range(20)] for _ in range(10)]

    for i, arr in enumerate(test_arrs):
        print('=' * 100)
        print(f'Test Case {i + 1}')
        print(f'Test Array: {arr}')
        expected = sorted(arr)
        _mergesort_arr_iterative(arr)
        print(f'Sorted Array: {arr}')
        print(f'Test Result: {arr == expected}')
