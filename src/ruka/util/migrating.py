from dataclasses import dataclass, is_dataclass, fields, MISSING
from ruka.util.json import XJSONSerializable, State
from typing import Dict, List, Optional


class Migrating(XJSONSerializable):
    """
    USAGE
        Migrating provides mechanism of backward-compatibility of serialized
        data. Specifically, newer version of code will be able to load a state,
        saved by the older version of code.

        Lets take this dataclass:

            @dataclass
            class Foo(Migrating):
                x: int
                y: int

            >> Foo(1, 2).__getstate__()
            {'x': 1, 'y': 2, 'class': {'name': Foo, 'module': 'foo', 'version': 1}}

        And create version 2 by adding a field:

            @dataclass
            class Foo(Migrating):
                x: int
                y: int
                z: int = 0

                CHANGES: typing.ClassVar = {
                    2: [Add('z')]
                }

            >>> Foo(1, 2).__getstate__()
            {'x': 1, 'y': 2, 'z': 0, 'class': {'name': Foo, 'module': 'foo', 'version': 2}}

        We've created version 2 of the class. When you add a new field, you
        must always specify either default= or default_factory=.

        Next, let's create version 3 by removing one field and by renaming
        another:

            @dataclass
            class Foo(Migrating):
                y1: int
                z: int = 0

                CHANGES: typing.ClassVar = {
                    2: [Add('z')],
                    3: [Remove('x'), Rename('y', 'y1')]
                }

            >>> Foo(2, 3).__getstate__()
            {'y1': 2, 'z': 3, 'class': {'name': Foo, 'module': 'foo', 'version': 3}}

    NOTES
        - Currently, there are three types of changes available: Add, Remove,
          and Rename. More might be available later.

        - Warning: only dataclass fields are saved or loaded in __getstate__ /
          __setstate__:

              @dataclass
              class Bar(Migrating):
                  x: int

              >>> b = Bar(1)
              >>> b.__getstate__()
              >>> {'x': 1, 'class': {'name': 'Bar', 'module': 'bar', 'version': 1}}
              >>> b.y = 2
              >>> b.__getstate__()
              >>> {'x': 1, 'class': {'name': 'Bar', 'module': 'bar', 'version': 1}}  # <--- No y!

        - Migrating is XJSONSerializable. This means that if all its fields are
          [X]JSONSerializable, the dataclass itself will also be serializable
          through ruka.json.x{dump,load}*
    """

    def __new__(cls, *args, **kwargs):
        validate_class(cls)
        return super().__new__(cls)

    def xjson_getstate(self) -> State:
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state['class'] = self.xjson_getclass()
        return state

    def xjson_setstate(self, state: State):
        validate_state(state, type(self))
        migrate_state_to_current_version(type(self), state)
        for k, v in state.items():
            if k == 'class':
                continue
            setattr(self, k, v)

    @classmethod
    def xjson_getclass(cls):
        j = super().xjson_getclass()
        j['version'] = get_current_version(cls)
        return j


class Change:
    pass


@dataclass
class Add(Change):
    name: str


@dataclass
class Remove(Change):
    name: str


@dataclass
class Rename(Change):
    name_old: str
    name_new: str


# --------------------------------------------------------------------- Util --


def validate_class(cls):
    """
    Check that class is a well-formed Migrating.
    """
    # Is Migrating.
    assert issubclass(cls, Migrating)

    # Class is a dataclass.
    assert is_dataclass(cls)

    # Versions are consecutive.
    changes = _get_changes(cls)
    if changes:
        current_version = get_current_version(cls)
        assert list(changes.keys()) == list(range(2, current_version + 1))

    # Have required defaults.
    should_have_defaults = set()

    for version in sorted(changes.keys()):
        for change in changes[version]:
            if isinstance(change, Add):
                should_have_defaults.add(change.name)
            elif isinstance(change, Remove):
                should_have_defaults.discard(change.name)
            elif isinstance(change, Rename):
                if change.name_old in should_have_defaults:
                    should_have_defaults.remove(change.name_old)
                    should_have_defaults.add(change.name_new)
            else:
                assert 0

    field_map = {field.name: field for field in fields(cls)}
    for key in should_have_defaults:
        field = field_map[key]
        if field.default is MISSING and field.default_factory is MISSING:
            assert 0, f'field {key} requires default= or default_factory='


def validate_state(state: State, cls: Optional[type] = None):
    """
    Check that state is well-formed.
    If cls is not None, check that state corresponds to cls.
    """
    # Check version.
    version_from = state['class']['version']
    if version_from <= 0:
        raise RuntimeError(f"invalid version: {state!r}")

    # Check that state corresponds to class.
    if cls is not None:
        if (
            state['class']['name'] != cls.__qualname__ or
            state['class']['module'] != cls.__module__
        ):
            raise RuntimeError(f'invalid state for {cls}: {state!r}')

    # Chech state version against class version.
    if cls is not None:
        current_version = get_current_version(cls)
        if not (0 < version_from <= current_version):
            raise RuntimeError(
                f"invalid version: {state!r}; expected 0..{current_version}")


def get_current_version(cls: type) -> int:
    assert issubclass(cls, Migrating)
    changes = _get_changes(cls)
    if not changes:
        return 1
    return max(changes.keys())


def migrate_state_to_current_version(cls: type, state: State):
    assert issubclass(cls, Migrating)
    class _Stub: pass

    version_from = state['class']['version']
    current_version = get_current_version(cls)

    # Sequentially migrate to a higher version.
    changes = _get_changes(cls)
    while version_from < current_version:
        for change in changes[version_from + 1]:
            if isinstance(change, Add):
                state[change.name] = _Stub
            elif isinstance(change, Remove):
                if change.name not in state:
                    raise RuntimeError(
                        f'invalid state: {state!r}, '
                        f'missing field {change.name}')
                state.pop(change.name)
            elif isinstance(change, Rename):
                if change.name_old not in state:
                    raise RuntimeError(
                        f'invalid state: {state!r}, '
                        f'missing field {change.name}')
                state[change.name_new] = state.pop(change.name_old)
            else:
                assert 0
        version_from += 1

    # Fill in defaults.
    field_map = {field.name: field for field in fields(cls)}
    for key in state.keys():
        if key == 'class':
            continue
        field = field_map[key]
        if state[key] is _Stub:
            if field.default is not MISSING:
                state[key] = field.default
            elif field.default_factory is not MISSING:
                state[key] = field.default_factory()
            else:
                assert 0

    # Fix version.
    state['class']['version'] = current_version

    # Check for extra fields.
    expected_keys = set(field_map.keys())
    real_keys = set(state.keys()) - {'class'}
    missing_keys = expected_keys - real_keys
    extra_keys = real_keys - expected_keys
    if missing_keys:
        raise RuntimeError(
            f'invalid state: {state!r}; '
            f'following keys are missing: {missing_keys}')
    if extra_keys:
        raise RuntimeError(
            f'invalid state: {state!r}; '
            f'following keys shouldn\'t be here: {extra_keys}')


def _get_changes(cls) -> Dict[str, List['Change']]:
    return getattr(cls, 'CHANGES', {})
