"""
Lomuto partitioning algorithm implementations.

This module contains internal implementations of the Lomuto partitioning
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

# comparable interface
from __future__ import annotations

import random


def _partition_lomuto_array(arr: list[int], p_idx: int, l_idx: int = 0, r_idx: int | None = None) -> int:
    """
    Partition a list (or subarray) in-place using the Lomuto partition scheme.

    This function rearranges elements in the inclusive range [l_idx, r_idx] around the pivot
    value initially located at `p_idx`. After partitioning:

    - The pivot value is moved to its final position `q` (the returned index).
    - All elements in arr[l_idx:q] are <= pivot.
    - All elements in arr[q+1:r_idx+1] are  > pivot.

    Notes:
    - This is a low-level primitive. For degenerate segments (size 0/1), callers should
      normally avoid calling partition at all.
    - Index validation is performed; invalid indices raise IndexError.

    Args:
        arr: List of integers to partition (modified in place).
        p_idx: Pivot index (must lie within [l_idx, r_idx]).
        l_idx: Left boundary of the subarray (inclusive). Defaults to 0.
        r_idx: Right boundary of the subarray (inclusive). Defaults to len(arr) - 1.

    Returns:
        The final index of the pivot after partitioning.

    Raises:
        ValueError: If `arr` is empty.
        IndexError: If any index is out of bounds or inconsistent.
    """

    if not arr:
        raise ValueError('arr must be non-empty')

    n = len(arr)
    if r_idx is None:
        r_idx = n - 1

    # Bounds / consistency checks
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
        return l_idx

    # Move pivot to the end (r_idx)
    arr[p_idx], arr[r_idx] = arr[r_idx], arr[p_idx]
    pivot = arr[r_idx]

    # Partition
    i = l_idx
    for j in range(l_idx, r_idx):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # Move pivot into its final place
    arr[i], arr[r_idx] = arr[r_idx], arr[i]
    return i


def _partition_lomuto_array_random(arr: list[int], l_idx: int = 0, r_idx: int | None = None) -> int:
    """
    Partition a list (or subarray) in-place using the Lomuto partition scheme and randomized pivot selection

    This function rearranges elements in the inclusive range [l_idx, r_idx] around
    the randomly selected pivot value. After partitioning:

    - The pivot value is moved to its final position `q` (the returned index).
    - All elements in arr[l_idx:q] are <= pivot.
    - All elements in arr[q+1:r_idx+1] are  > pivot.

    Notes:
    - This is a low-level primitive. For degenerate segments (size 0/1), callers should
      normally avoid calling partition at all.
    - Index validation is performed; invalid indices raise IndexError.

    Args:
        arr: List of integers to partition (modified in place).
        l_idx: Left boundary of the subarray (inclusive). Defaults to 0.
        r_idx: Right boundary of the subarray (inclusive). Defaults to len(arr) - 1.

    Returns:
        The final index of the pivot after partitioning.

    Raises:
        ValueError: If `arr` is empty.
        IndexError: If any index is out of bounds or inconsistent.
    """
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
        return l_idx

    # select random pivot
    p_idx = random.randint(l_idx, r_idx)

    # Move pivot to the end (r_idx)
    arr[p_idx], arr[r_idx] = arr[r_idx], arr[p_idx]
    pivot = arr[r_idx]

    # Partition
    i = l_idx
    for j in range(l_idx, r_idx):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # Move pivot into its final place
    arr[i], arr[r_idx] = arr[r_idx], arr[i]
    return i
