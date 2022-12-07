import importlib
import json
import tempfile

from ruka_os import distributed_fs_v2 as dfs
from typing import Any, Dict, List, Union


# ------------------------------------------------------------------- Typing --


# A typing hint for objects that can be *losslessly* serialized and
# deserialized through json.dump / json.load
#
# Note that e.g. tuples can be serialized, but they turn into lists when
# you deserialize them back. This is not lossless, so tuples are not included
# into this hint.


JSONScalar = Union[str, float, int, bool, type(None)]
JSONArray = List['JSONSerializable']
JSONMapping = Dict[str, 'JSONSerializable']
JSONSerializable = Union[JSONScalar, JSONArray, JSONMapping]


# --------------------------------------------------- Extended serialization --


# Extended JSON serialization functions:
#
#     xdumps / xloads  --  to/from string
#     xdumpj / xloadj  --  to/from object, serializable with json.dump / json.load
#     xdumpr / xloadr  --  to/from DFS
#
# In addition to regular json.dumps / json.loads, all these functions support
# saving and loading objects, derived from XJSONSerializable.
#
# Note that these functions have a limitation: they cannot process dicts that
# have 'class' key, since that key is used by Migrating to store the object
# info.


def xdumps(obj: Any) -> str:
    """
    xdumps = e[x]tended [dump] to [s]tring
    """
    return json.dumps(xdumpj(obj))


def xloads(s: str) -> Any:
    """
    xloads = e[x]tended [load] from [s]tring
    """
    return xloadj(json.loads(s))


def xdumpj(obj: Any) -> JSONSerializable:
    """
    xdumpj = e[x]tended [dump] to [J]SONSerializable
    """
    # JSONScalar.
    if type(obj) in [str, float, int, bool, type(None)]:
        return obj

    # JSONArray.
    if type(obj) is list:
        return [xdumpj(i) for i in obj]

    # JSONMapping.
    if type(obj) is dict:
        if not all(type(key) is str for key in obj.keys()):
            raise ValueError(f'not serializable losslessly to JSON: {obj!r}')
        if 'class' in obj:
            raise ValueError(f"cannot serialize dict that has a 'class' key: {obj!r}")
        return {key: xdumpj(value) for key, value in obj.items()}

    # XJSONSerializable.
    if isinstance(obj, XJSONSerializable):
        j = obj.xjson_getstate()
        if 'class' not in j:
            raise RuntimeError(
                f'invalid implementation of xjson_getstate() in {type(obj)}: '
                f'no "class" key in resulting dict: {j}')
        j = {key: xdumpj(value) for key, value in j.items()}
        return j

    # Unsupported type.
    raise ValueError(f'not serializable losslessly to JSON: {obj!r}')


def xloadj(j: JSONSerializable) -> Any:
    """
    xdumpj = e[x]tended [load] from [J]SONSerializable
    """
    # JSONScalar.
    if isinstance(j, (str, float, int, bool, type(None))):
        return j

    # JSONArray.
    if isinstance(j, list):
        return [xloadj(i) for i in j]

    # JSONMapping.
    if isinstance(j, dict):
        # - Check keys.
        if not all(type(key) is str for key in j.keys()):
            raise ValueError(f'invalid input: {j!r}')

        # - Load keys.
        j = {key: xloadj(value) for key, value in j.items()}

        # - XJSONSerializable.
        if 'class' in j:
            cls = XJSONSerializable.xjson_loadclass(j['class'])
            obj = cls.__new__(cls)
            obj.xjson_setstate(j)
            return obj

        return j

    # Unsupported type.
    raise ValueError(f'invalid input: {j!r}')


def xdumpr(obj: Any, remote_path: str):
    """
    xdumpr = e[x]tended [dump] to [r]emote DFS
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = f'{tmpdir}/file'
        with open(local_path, 'wt') as f:
            json.dump(xdumpj(obj), f)
        dfs.upload(local_path, remote_path)


def xloadr(remote_path: str) -> Any:
    """
    xloadr = e[x]tended [load] from [r]emote DFS
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = f'{tmpdir}/file'
        dfs.download(remote_path, local_path)
        with open(local_path, 'rt') as f:
            return xloadj(json.load(f))


# ------------------------------------------------ Custom JSON serializables --


State = Dict[str, Any]


class XJSONSerializable:
    """
    To make your class serializable with x{dump,load} functions:

    1. Inherit from XJSONSerializable;
    2. Override xjson_setstate() and xjson_getstate().

    This creates serialization hooks, that are parallel to __getstate__ and
    __setstate__. This way, you can support two serialization formats for the
    same object: human-readable and binary.
    """

    def xjson_getstate(self) -> State:
        """
        Must return dict.

        All keys must be strings.

        MUST contain 'class' key, that contains the result of xjson_getclass().

        As with __getstate__, the result will be recursively serialized by
        xdump().
        """
        raise NotImplementedError()

    def xjson_setstate(self, state: State):
        """
        Note that 'state' MUST have an extra 'class' field, which must contain
        the result of xjson_getclass().
        """
        raise NotImplementedError()

    @classmethod
    def xjson_getclass(cls) -> JSONSerializable:
        """
        Descendants can override this method and add fields to this dict.
        """
        return {
            'module': cls.__module__,
            'name': cls.__qualname__
        }

    @staticmethod
    def xjson_loadclass(j: JSONSerializable) -> type:
        module = importlib.import_module(j['module'])
        return getattr(module, j['name'])

    def __init_subclass__(cls):
        # Check that class is not nested in some funky place.
        assert cls.__qualname__
        assert '.' not in cls.__qualname__
        assert (
            cls.__module__ and
            not cls.__module__.startswith('__') and
            not cls.__module__.startswith('.')
        )

    def __getstate__(self) -> State:
        """
        XJSONSerializable are automatically picklable using the same state.
        You can override this if you want.
        """
        return self.xjson_getstate()

    def __setstate__(self, state: State):
        """
        XJSONSerializable are automatically picklable using the same state.
        You can override this if you want.
        """
        self.xjson_setstate(state)
