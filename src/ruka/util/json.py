import json
import tempfile

from ruka.util.migrating import Migrating, load_object
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
# saving and loading dataclasses, derived from ruka.util.migrating.Migrating.
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

    # Migrating.
    if isinstance(obj, Migrating):
        return {key: xdumpj(value) for key, value in obj.__getstate__().items()}

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

        # - Migrating.
        if 'class' in j:
            j = j.copy()
            for key in j:
                if key == 'class':
                    continue
                j[key] = xloadj(j[key])
            return load_object(j)

        # - dict.
        else:
            return {key: xloadj(value) for key, value in j.items()}

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
