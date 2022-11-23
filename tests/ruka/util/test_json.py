from dataclasses import dataclass
from ruka.util.json import xdumps, xloads
from ruka.util.migrating import Migrating


@dataclass
class Foo(Migrating):
    x: int
    y: int


@dataclass
class Bar(Migrating):
    f: Foo


def test_xdumps():
    value = [1, 2, {'a': 3, 'b': Foo(4, 5)}]
    assert value == xloads(xdumps(value))

    value = Bar(Foo(1, 2))
    assert value == xloads(xdumps(value))