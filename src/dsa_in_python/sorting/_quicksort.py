"""
Quicksort algorithm implementations.

This module contains internal implementations of the Quicksort sorting
algorithm, covering multiple design variants and trade-offs.

The implementations here may differ along dimensions such as:
- data structure (e.g. Python lists, linked lists)
- mutability (in-place vs out-of-place)
- partitioning strategy (e.g. Lomuto, Hoare)
- pivot selection strategy (e.g. first element, random, median-of-three)

These implementations are considered internal to the
``dsa_in_python.sorting`` subpackage and are not part of the public API.
Public-facing access to Quicksort functionality is provided via
``dsa_in_python.sorting``.

Design notes:
- All implementations aim for clarity and correctness first.
- Performance optimizations are included only where they preserve
  readability and algorithmic intent.
- Time complexity: average O(n log n), worst-case O(n²).
- Space complexity depends on the specific variant.

This module is intended for educational and comparative purposes.
"""

from __future__ import annotations

import random
from typing import TypeVar, cast

from dsa_in_python.common.nodes import Node
from dsa_in_python.common.types import SupportsLessThan
from dsa_in_python.partitioning._lomuto import _partition_lomuto_array_random
from dsa_in_python.partitioning._three_way import _partition_3way_array_random

T = TypeVar('T', bound=SupportsLessThan)


# -------------------- Array based implementations ----------------------------
def _quicksort_arr_random_lomuto_recursive(arr: list[T]) -> None:
    # input validation
    if not arr:
        return

    def _quicksort(arr: list[T], l_idx: int, r_idx: int) -> None:
        if r_idx < l_idx:
            return

        # partition subarray l_idx : r_idx on randomly selected pivot
        # Lomuto partitioning scheme
        p_idx = _partition_lomuto_array_random(arr, l_idx, r_idx)

        # recurse down left and right partitions
        _quicksort(arr, l_idx, p_idx - 1)
        _quicksort(arr, p_idx + 1, r_idx)

        return

    return _quicksort(arr, l_idx=0, r_idx=len(arr) - 1)


def _quicksort_arr_random_3way_recursive(arr: list[T]) -> None:
    if not arr:
        return

    def _quicksort(arr: list[T], l_idx: int, r_idx: int) -> None:
        if r_idx < l_idx:
            return

        # partition subarray l_idx : r_idx on randomly selected pivot
        # 3-way partitioning scheme
        l_p_idx, r_p_idx = _partition_3way_array_random(arr, l_idx, r_idx)

        # Recurse down left and right partitions, middle partition ignored
        _quicksort(arr, l_idx, l_p_idx - 1)
        _quicksort(arr, r_p_idx + 1, r_idx)
        return

    return _quicksort(arr, 0, len(arr) - 1)


# Iterative QuickSort Implementations (Stack based)
def _quicksort_arr_random_lomuto_iterative(arr: list[T]) -> None:
    if not arr:
        return

    # initialize partition stack
    partition_stack: list[tuple[int, int]] = [(0, len(arr) - 1)]

    while len(partition_stack) > 0:
        # pop next partition to process
        l_idx, r_idx = partition_stack.pop()

        # partition & compute sub-partition sizes
        p_idx = _partition_lomuto_array_random(arr, l_idx, r_idx)
        lp_size = p_idx - l_idx
        rp_size = r_idx - p_idx

        # push subpartitions onto stack only if p_size > 1
        # push larger partition first to minimize stack size (O(log*n))
        # in a tie push left partition first
        if rp_size > lp_size:
            if rp_size > 1:
                partition_stack.append((p_idx + 1, r_idx))
            if lp_size > 1:
                partition_stack.append((l_idx, p_idx - 1))

        else:
            if lp_size > 1:
                partition_stack.append((l_idx, p_idx - 1))
            if rp_size > 1:
                partition_stack.append((p_idx + 1, r_idx))

    return


def _quicksort_arr_random_3way_iterative(arr: list[T]) -> None:
    if not arr:
        return

    # initialize partition stack
    partition_stack: list[tuple[int, int]] = [(0, len(arr) - 1)]

    while len(partition_stack) > 0:
        # pop next partition to process
        l_idx, r_idx = partition_stack.pop()

        # partition & compute sub-partition sizes
        l_p_idx, r_p_idx = _partition_3way_array_random(arr, l_idx, r_idx)
        lp_size = l_p_idx - l_idx
        rp_size = r_idx - r_p_idx

        # push subpartitions onto stack only if p_size > 1
        # push larger partition first to minimize stack size (O(log*n))
        # in a tie push left partition first
        if rp_size > lp_size:
            if rp_size > 1:
                partition_stack.append((r_p_idx + 1, r_idx))
            if lp_size > 1:
                partition_stack.append((l_idx, l_p_idx - 1))
        else:
            if lp_size > 1:
                partition_stack.append((l_idx, l_p_idx - 1))
            if rp_size > 1:
                partition_stack.append((r_p_idx + 1, r_idx))

    return


# -------------------- Linked List based implementations ----------------------------
def _quicksort_ll_random_3way_recursive(head: Node[T]) -> Node[T] | None:
    if not head or head._next is None:
        return head

    def _select_pivot(head: Node[T]) -> T:
        """
        Selects a Node from a list at random, with probability 1/ # Nodes;
        Returns the value referenced by node.
        Notes:
            - Implements reservoir sampling with sample size k = 1
            - O(n) time complexity
        """
        pivot: Node[T] = head
        cnt: int = 1
        curr_node: Node[T] | None = head._next

        while curr_node is not None:
            cnt += 1
            if random.randint(1, cnt) == 1:
                pivot = curr_node
            curr_node = curr_node._next
        return pivot.value

    def _append_node(node: Node[T], head: Node[T] | None, tail: Node[T] | None) -> tuple[Node[T], Node[T]]:
        """
        Helper: attaches node to the end of a list defined by head, tail
        Returns: head and new tail
        """
        node._next = None
        if head is None or tail is None:
            return node, node
        tail._next = node
        return head, node

    def _quick_sort(head: Node[T] | None) -> tuple[Node[T] | None, Node[T] | None]:
        """
        Recursive quicksort, using random pivot selection and 3-way partition
        """
        if head is None or head._next is None:  # base case, single node list
            return head, head

        # select pivot
        pivot: T = _select_pivot(head)

        # 3-way partition setup
        L_head = L_tail = None
        E_head = E_tail = None
        G_head = G_tail = None

        # 3-way partition
        curr_node: Node[T] | None = head
        while curr_node is not None:
            next_node = curr_node._next
            curr_value: T = curr_node.value

            if curr_value < pivot:  # add to L list
                L_head, L_tail = _append_node(curr_node, L_head, L_tail)

            elif not (curr_value < pivot or pivot < curr_value):  # add to E list
                E_head, E_tail = _append_node(curr_node, E_head, E_tail)
            else:
                # add to G list
                G_head, G_tail = _append_node(curr_node, G_head, G_tail)
            curr_node = next_node

        # recursive sort on partitions
        L_head, L_tail = _quick_sort(L_head)
        G_head, G_tail = _quick_sort(G_head)

        # recombine and return new_head
        new_head: Node[T] = cast(Node[T], E_head)
        new_tail: Node[T] = cast(Node[T], E_tail)
        if L_head is not None and L_tail is not None:
            L_tail._next = new_head
            new_head = L_head

        if G_head is not None and G_tail is not None:
            new_tail._next = G_head
            new_tail = G_tail

        return new_head, new_tail

    new_head, _ = _quick_sort(head)

    return new_head
