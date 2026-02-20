from typing import Protocol, Self


class SupportsLessThan(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...
