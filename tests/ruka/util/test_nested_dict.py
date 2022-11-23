from dataclasses import dataclass, is_dataclass
from typing import Any

import ruka.util.nested_dict as nd


@dataclass
class D:
    foo: Any
    bar: Any


def test_map_inplace():
    inc = lambda x: x + 1 if isinstance(x, int) else x

    x = 1
    y = nd.map_inplace(inc, x)
    assert y == 2

    x = [1]
    y = nd.map_inplace(inc, x)
    assert y is x
    assert y == [1]

    x = {'a': 1, 'b': 2}
    y = nd.map_inplace(inc, x)
    assert y is x
    assert y == {'a': 2, 'b': 3}

    x = D(1, 2)
    y = nd.map_inplace(inc, x)
    assert y is x
    assert y == D(1, 2)

    x = D(1, 2)
    y = nd.map_inplace(inc, x, enter_dataclasses=True)
    assert y is x
    assert y == D(2, 3)

    x = D({'fizz': 1, 'buzz': [2]}, 3)
    y = nd.map_inplace(inc, x, enter_dataclasses=True)
    assert y is x and y.foo is x.foo and y.foo['buzz'] is x.foo['buzz']
    assert y == D({'fizz': 2, 'buzz': [2]}, 4)


def test_map_inplace_enter_lists():
    inc = lambda x: x + 1 if isinstance(x, int) else x

    x = D(1, [2, 3])
    y = nd.map_inplace(inc, x, enter_dataclasses=True, enter_lists=True)
    assert y is x
    assert y == D(2, [3, 4])

    x = [D(1, [2, 3]), 4]
    y = nd.map_inplace(inc, x, enter_dataclasses=True, enter_lists=True)
    assert y is x
    assert y == [D(2, [3, 4]), 5]


def test_map_inplace_apply_on_nodes():
    def modify(x):
        if isinstance(x, int):
            return x + 1
        if isinstance(x, list):
            x.append('add')
            return x
        if isinstance(x, dict):
            x['add'] = 1
            return x
        if is_dataclass(x):
            x.foo = 10
            return x

    x = [D(1, [2, {'val': 3}]), 4]
    y = nd.map_inplace(modify, x, enter_dataclasses=True, enter_lists=True, apply_on_nodes=True)
    assert y is x
    assert y == [D(10, [3, {'val': 4, 'add': 1}, 'add']), 5, 'add']


def test_items():
    x = D({'fizz': 1, 'buzz': [2]}, 3)
    y = list(nd.items(x, prefix='root', sep='/'))
    assert y == [
        ('root', x)
    ]

    x = D({'fizz': 1, 'buzz': [2]}, 3)
    y = list(nd.items(x, prefix='root', sep='/', enter_dataclasses=True))
    assert y == [
        ('root/foo/fizz', 1),
        ('root/foo/buzz', [2]),
        ('root/bar', 3)
    ]