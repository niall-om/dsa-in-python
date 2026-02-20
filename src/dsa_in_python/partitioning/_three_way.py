"""
3-Way partitioning algorithm implementations.

This module contains internal implementations of the 3-way partitioning
algorithm, covering multiple design variants and trade-offs.

The implementations here may differ along dimensions such as:
- data structure (e.g. Python lists, linked lists)
- pivot selection strategy (e.g. first element, random, median-of-three)

These implementations are considered internal to the
``dsa_in_python.partitioning`` subpackage and are not part of the public API.
Public-facing access to Lomuto functionality is provided via
``dsa_in_python.partitioning``.

Design notes:
- All implementations aim for clarity and correctness first.
- Performance optimizations are included only where they preserve
  readability and algorithmic intent.

This module is intended for educational and comparative purposes.
"""

from __future__ import annotations

import random
from typing import TypeVar

from dsa_in_python.common.types import SupportsLessThan

T = TypeVar('T', bound=SupportsLessThan)


def _partition_3way_array(arr: list[T], p_idx: int, l_idx: int = 0, r_idx: int | None = None) -> tuple[int, int]:
    if not arr:
        raise ValueError('arr must be non-empty')

    n = len(arr)
    if r_idx is None:
        r_idx = n - 1

    # Bounds / Consistency check
    if not (0 <= l_idx < n):
        raise IndexError(f'l_idx ({l_idx}) is out of bounds for array length {n}')
    if not (0 <= r_idx < n):
        raise IndexError(f'r_idx ({r_idx}) is out of bounds for array length {n}')
    if l_idx > r_idx:
        raise IndexError(f'l_idx ({l_idx}) must be <= r_idx ({r_idx})')
    if not (l_idx <= p_idx <= r_idx):
        raise IndexError(f'p_idx ({p_idx}) must be between l_idx ({l_idx}) and r_idx ({r_idx}) inclusive')

    # If caller passes a degenerate segment, just return the left boundary deterministically
    # (Better: have the caller avoid calling partition in the first place.)
    if l_idx == r_idx:
        return l_idx, r_idx

    # Partition
    pivot = arr[p_idx]
    i = j = l_idx
    k = r_idx

    while j <= k:
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1
        elif not (arr[j] < pivot or arr[j] > pivot):  # arr[j] == pivot
            j += 1

        else:  # arr[j] > pivot
            arr[k], arr[j] = arr[j], arr[k]
            k -= 1

    return i, j - 1


def _partition_3way_array_random(arr: list[T], l_idx: int = 0, r_idx: int | None = None) -> tuple[int, int]:
    if not arr:
        raise ValueError('arr must be non-empty')

    n = len(arr)

    if r_idx is None:
        r_idx = n - 1

    # Bounds / Consistency checks
    if not (0 <= l_idx < n):
        raise IndexError(f'l_idx ({l_idx}) is out of bounds for array length {n}')
    if not (0 <= r_idx < n):
        raise IndexError(f'r_idx ({r_idx}) is out of bounds for array length {n}')
    if l_idx > r_idx:
        raise IndexError(f'l_idx ({l_idx}) must be <= r_idx ({r_idx})')

    # If caller passes a degenerate segment, just return the left boundary deterministically
    # (Better: have the caller avoid calling partition in the first place.)
    if l_idx == r_idx:
        return l_idx, r_idx

    # Select random pivot
    p_idx = random.randint(l_idx, r_idx)
    pivot = arr[p_idx]

    # Partition
    i = j = l_idx
    k = r_idx

    while j <= k:
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1

        elif not (arr[j] < pivot or arr[j] > pivot):  # arr[j] == pivot
            j += 1

        else:
            arr[k], arr[j] = arr[j], arr[k]  # arr[j] > pivot
            k -= 1
    return i, j - 1


if __name__ == '__main__':
    test_arr = [random.randint(0, 10) for _ in range(20)]
    print(test_arr)
    l_bound, u_bound = _partition_3way_array_random(test_arr)
    print(l_bound, u_bound, test_arr[l_bound], test_arr[u_bound])
    print(test_arr)
