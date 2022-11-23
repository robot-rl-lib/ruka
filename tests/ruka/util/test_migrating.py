import pytest

from dataclasses import dataclass
from pickle import dumps, loads
from ruka.util.migrating import Migrating, Add, Remove, Rename
from typing import ClassVar



@dataclass
class Foo(Migrating):
    field1: int
    field2: int


FooV1 = Foo


@dataclass
class Foo(Migrating):
    field1: int
    field2: int
    field3: int = 3

    CHANGES: ClassVar = {
        2: [Add('field3')]
    }


FooV2 = Foo


@dataclass
class Foo(Migrating):
    field1: int
    field2: int
    field3: int

    CHANGES: ClassVar = {
        2: [Add('field3')]
    }


FooV2Invalid = Foo


@dataclass
class Foo(Migrating):
    field2: int
    field3: int = 3

    CHANGES: ClassVar = {
        2: [Add('field3')],
        3: [Remove('field1')]
    }


FooV3 = Foo


@dataclass
class Foo(Migrating):
    field2: int

    CHANGES: ClassVar = {
        2: [Add('field3')],
        3: [Remove('field1')],
        4: [Remove('field3')]
    }


FooV4 = Foo


@dataclass
class Foo(Migrating):
    field2: int
    field3: int = '3'

    CHANGES: ClassVar = {
        2: [Add('field3')],
        3: [Remove('field1')],
        4: [Remove('field3')],
        5: [Add('field3')]
    }


FooV5 = Foo


@dataclass
class Foo(Migrating):
    field2: int
    field3x: int

    CHANGES: ClassVar = {
        2: [Add('field3')],
        3: [Remove('field1')],
        4: [Remove('field3')],
        5: [Add('field3')],
        6: [Rename('field3', 'field3x')]
    }


FooV6Invalid = Foo


@dataclass
class Foo(Migrating):
    field2: int
    field3x: int = '3'

    CHANGES: ClassVar = {
        2: [Add('field3')],
        3: [Remove('field1')],
        4: [Remove('field3')],
        5: [Add('field3')],
        6: [Rename('field3', 'field3x')]
    }


FooV6 = Foo


Foo = None


def test_add():
    global Foo
    Foo = FooV1
    data_v1 = dumps(FooV1(1, 2))
    Foo = FooV2
    data_v2 = dumps(FooV2(1, 2, -3))
    assert loads(data_v1) == FooV2(1, 2, 3)
    assert loads(data_v2) == FooV2(1, 2, -3)


def test_add_nodefault():
    with pytest.raises(Exception):
        FooV2Invalid(1, 2, 3)


def test_remove():
    global Foo
    Foo = FooV1
    data_v1 = dumps(FooV1(1, 2))
    Foo = FooV2
    data_v2 = dumps(FooV2(1, 2, -3))
    Foo = FooV3
    data_v3 = dumps(FooV3(2, -3))
    assert loads(data_v1) == FooV3(2, 3)
    assert loads(data_v2) == FooV3(2, -3)
    assert loads(data_v3) == FooV3(2, -3)


def test_add_remove():
    global Foo
    Foo = FooV1
    data_v1 = dumps(FooV1(1, 2))
    Foo = FooV2
    data_v2 = dumps(FooV2(1, 2, -3))
    Foo = FooV3
    data_v3 = dumps(FooV3(2, -3))
    Foo = FooV4
    data_v4 = dumps(FooV4(2))
    assert loads(data_v1) == FooV4(2)
    assert loads(data_v2) == FooV4(2)
    assert loads(data_v3) == FooV4(2)
    assert loads(data_v4) == FooV4(2)


def test_remove_add():
    global Foo
    Foo = FooV1
    data_v1 = dumps(FooV1(1, 2))
    Foo = FooV2
    data_v2 = dumps(FooV2(1, 2, -3))
    Foo = FooV3
    data_v3 = dumps(FooV3(2, -3))
    Foo = FooV4
    data_v4 = dumps(FooV4(2))
    Foo = FooV5
    data_v5 = dumps(FooV5(2, '-3'))
    assert loads(data_v1) == FooV5(2, '3')
    assert loads(data_v2) == FooV5(2, '3')
    assert loads(data_v3) == FooV5(2, '3')
    assert loads(data_v4) == FooV5(2, '3')
    assert loads(data_v5) == FooV5(2, '-3')


def test_rename_nodefault():
    with pytest.raises(Exception):
        FooV6Invalid(2, '-3')


def test_rename():
    global Foo
    Foo = FooV1
    data_v1 = dumps(FooV1(1, 2))
    Foo = FooV2
    data_v2 = dumps(FooV2(1, 2, -3))
    Foo = FooV3
    data_v3 = dumps(FooV3(2, -3))
    Foo = FooV4
    data_v4 = dumps(FooV4(2))
    Foo = FooV5
    data_v5 = dumps(FooV5(2, '-3'))
    Foo = FooV6
    data_v6 = dumps(FooV6(2, '-3'))
    assert loads(data_v1) == FooV6(2, '3')
    assert loads(data_v2) == FooV6(2, '3')
    assert loads(data_v3) == FooV6(2, '3')
    assert loads(data_v4) == FooV6(2, '3')
    assert loads(data_v5) == FooV6(2, '-3')
    assert loads(data_v6) == FooV6(2, '-3')



