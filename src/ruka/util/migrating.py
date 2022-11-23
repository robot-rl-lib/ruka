from dataclasses import dataclass, is_dataclass, fields, MISSING
from typing import Any, Dict, List, Optional


State = Dict[str, Any]


class Migrating:
    """
    This class has two purposes.

    FIRST, it provides __getstate__ / __setstate__ to dataclasses. It is not
    being done so that you could pickle them, but rather so that you could
    inspect that state further (e.g. if you're serializing a dataclass into
    JSON).

    - State is a Dict[str, Any] which contains field names as keys and field
      values as values.
    - State contains class name and version.

    An example:

        @dataclass
        class Foo(Migrating):
            x: int
            y: int

        >>> Foo(1, 2).__getstate__()
        {'x': 1, 'y': 2, 'class': {'name': 'Foo', 'module': 'foo', 'version': 1}}

    SECOND, Migrating provides mechanism of backward-compatibility of
    serialized data. Specifically, newer version of code will be able to load
    a state, saved by the older version of code.

    Lets take the dataclass from the example above and modify it. First, we
    will create version 2 by adding a field:

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

    We've created version 2 of the class. When you add a new field, you should
    always specify either default= or default_factory=.

    Next, let's create version 3 by removing one field and by renaming another:

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

    Currently, there are three types of changes available: Add, Remove, and
    Rename.

    Note that only dataclass fields are saved or loaded in __getstate__ /
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
    """

    def __new__(cls, *args, **kwargs):
        validate_class(cls)
        return object.__new__(cls)

    def __getstate__(self) -> State:
        state = {}
        for field in fields(self):
            state[field.name] = getattr(self, field.name)
        state['class'] = {
            'name': type(self).__qualname__,
            'module': type(self).__module__,
            'version': get_current_version(type(self))
            }
        return state

    def __setstate__(self, state: State):
        validate_state(state, type(self))
        migrate_state_to_current_version(type(self), state)
        for k, v in state.items():
            if k == 'class':
                continue
            setattr(self, k, v)


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

    # Class is not nested in some funky place.
    assert cls.__qualname__
    assert cls.__module__ and not cls.__module__.startswith('__')
    assert '.' not in cls.__qualname__

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
    # Check schema.
    if (
        not isinstance(state, dict) or
        'class' not in state or
        not isinstance(state['class'], dict) or
        set(state['class'].keys()) != {'version', 'name', 'module'} or
        not isinstance(state['class']['version'], int) or
        not isinstance(state['class']['name'], str) or
        not isinstance(state['class']['module'], str)
        ):
        raise RuntimeError(f'invalid state for {cls}: {state!r}')

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


def load_object(state: State) -> Migrating:
    """
    Load an object from state. Like pickle.loads()
    """
    validate_state(state)
    module = __import__(state['class']['module'])
    cls = getattr(module, state['class']['name'])
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


def _get_changes(cls) -> Dict[str, List['Change']]:
    return getattr(cls, 'CHANGES', {})
