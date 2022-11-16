from dataclasses import is_dataclass, fields, FrozenInstanceError
from typing import Any, Callable, Dict, Iterator, Tuple, Union


NestedDict = Union[Any, Dict[str, 'NestedDict']]


def map_inplace(
        fn: Callable,
        x: NestedDict,
        enter_dataclasses: bool = False
    ) -> NestedDict:
    """
    If x is an atomic object, returns fn(x).
    If x is a dict/dataclass, returns x, but after modification.
    """
    # Dict.
    if isinstance(x, dict):
        for k in x.keys():
            x[k] = map_inplace(fn, x[k], enter_dataclasses)
        return x

    # Dataclass.
    if enter_dataclasses and (is_dataclass(x) and not isinstance(x, type)):
        try:
            for k in fields(x):
                setattr(x, k, fn(getattr(x, k)))
            return x
        except FrozenInstanceError:
            return fn(x)

    # Regular item.
    return fn(x)


def items(
        x: NestedDict,
        prefix: str,
        sep: str = '/',
        enter_dataclasses: bool = False
    ) -> Iterator[Tuple[str, Any]]:
    """
    Example input:
        keys(
            {
                'a': {'x': 1, 'y': 2}
                'b': {'z': 3, 'w': 4}
                'c': 5
            },
            prefix='prefix',
            sep='/'
        )

    Example output:
        ('prefix/a/x', 1)
        ('prefix/a/y', 2)
        ('prefix/b/z', 3)
        ('prefix/b/w', 4)
        ('prefix/c', 5)
    """
    # Dict.
    if isinstance(x, dict):
        for k, v in x.items():
            yield from items(v, f'{prefix}{sep}{k}', sep, enter_dataclasses)

    # Dataclass.
    if enter_dataclasses and (is_dataclass(x) and not isinstance(x, type)):
        for k in fields(x):
            v = getattr(x, k)
            yield from items(v, f'{prefix}{sep}{k}', sep, enter_dataclasses)

    # Item.
    yield (prefix, x)