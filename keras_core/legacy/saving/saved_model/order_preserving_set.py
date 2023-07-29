"""A set based on dict so that it preserves key insertion order.

Python Dicts are order-preserving since 3.6
(https://mail.python.org/pipermail/python-dev/2017-December/151283.html),
but sets are not. This class implements a set on top of a dict so that we get
deterministic iteration order across runs.
"""

import collections.abc


class OrderPreservingSet(collections.abc.MutableSet):
    """A set based on dict so that it preserves key insertion order."""

    def __init__(self, iterable=None):
        self._dict = {item: None for item in (iterable or [])}

    # abstract from collections.MutableSet
    def __len__(self):
        return len(self._dict)

    # abstract from collections.MutableSet
    def __contains__(self, value):
        return value in self._dict

    # override from collections.MutableSet
    def __iter__(self):
        return iter(self._dict)

    # abstract from collections.MutableSet
    def add(self, item):
        self._dict[item] = None

    # abstract from collections.MutableSet
    def discard(self, value):
        del self._dict[value]

    # override from collections.MutableSet
    def clear(self):
        self._dict = {}

    # override from collections.Set
    def __eq__(self, other):
        if not isinstance(other, OrderPreservingSet):
            return NotImplemented
        return self._dict.keys() == other._dict.keys()

    # override from collections.Set
    def __le__(self, other):
        if not isinstance(other, OrderPreservingSet):
            return NotImplemented
        return self._dict.keys() <= other._dict.keys()

    # override from collections.Set
    def __ge__(self, other):
        if not isinstance(other, OrderPreservingSet):
            return NotImplemented
        return self._dict.keys() >= other._dict.keys()

    # override from collections.Set
    def __and__(self, other):
        # collections.Set defaults to the ordering in other, we want to use self
        return self._from_iterable(value for value in self if value in other)

    # override from collections.Set
    def __or__(self, other):
        # ensure that other is ordered before performing __or__
        if not isinstance(other, OrderPreservingSet):
            raise TypeError(
                "cannot union an 'OrderPreservingSet' with an "
                "unordered iterable."
            )
        result = self._from_iterable(value for value in self)
        for value in other:
            result._dict[value] = None
        return result

    def union(self, other):
        return self | other