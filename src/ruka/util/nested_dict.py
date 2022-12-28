from dataclasses import is_dataclass, fields, FrozenInstanceError
from typing import Any, Callable, Dict, Iterator, Tuple, Union, Optional


NestedDict = Union[Any, Dict[str, 'NestedDict']]


def map_inplace(
        fn: Callable,
        x: NestedDict,
        enter_dataclasses: bool = False,
        enter_lists: bool = False,
        apply_on_nodes: bool = False,
    ) -> NestedDict:
    """
    Performs function on a nested dict recursively.
    If x is an atomic object, returns fn(x).
    If x is a dict/dataclass/list, returns x, but after modification.
    Additionaly may call fn(x) on dicts.
    Args:
        fn (Callable): function to call
        x (NestedDict): object to modify
        enter_dataclasses (bool, optional): if True will walk into dataclasses.
                                            Will treat it as atomic objects otherwise.
                                            Defaults to False.
        enter_lists (bool, optional): Walk into lists or treat is as atomic objects. Defaults to False.
        apply_on_nodes (bool, optional): Apply function on nodes. Defaults to False.

    Returns:
        NestedDict: modified object.
    """
    # Dict.
    if isinstance(x, dict):
        for k in x.keys():
            x[k] = map_inplace(fn, x[k], enter_dataclasses, enter_lists, apply_on_nodes)
        return fn(x) if apply_on_nodes else x

    # Dataclass.
    if enter_dataclasses and (is_dataclass(x) and not isinstance(x, type)):
        try:
            for k in fields(x):
                v = map_inplace(fn, getattr(x, k.name), enter_dataclasses, enter_lists, apply_on_nodes)
                setattr(x, k.name, v)
            return fn(x) if apply_on_nodes else x
        except FrozenInstanceError:
            return fn(x)

    # List.
    if enter_lists and isinstance(x, list):
        for ind in range(len(x)):
            x[ind] = map_inplace(fn, x[ind], enter_dataclasses, enter_lists, apply_on_nodes)
        return fn(x) if apply_on_nodes else x

    # Regular item.
    return fn(x)


def items(
        x: NestedDict,
        prefix: Optional[str] = None,
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
            if prefix:
                yield from items(v, f'{prefix}{sep}{k}', sep, enter_dataclasses)
            else:
                yield from items(v, f'{k}', sep, enter_dataclasses)
        return

    # Dataclass.
    if enter_dataclasses and (is_dataclass(x) and not isinstance(x, type)):
        for k in fields(x):
            v = getattr(x, k.name)
            if prefix:
                yield from items(v, f'{prefix}{sep}{k.name}', sep, enter_dataclasses)
            else:
                yield from items(v, f'{k.name}', sep, enter_dataclasses)
        return

    # Item.
    yield (prefix, x)

def flatten(x: NestedDict, sep: str = '/'):
    return {k: v for k, v in items(x, sep=sep)}

def unflatten(x: NestedDict, sep: str = '/'):
    out = dict()
    for key, value in x.items():
        parts = key.split(sep)
        d = out
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return out
