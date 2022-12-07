import json
import pickle

from dataclasses import is_dataclass, fields
from ruka.util.json import xdumpj, xloadj, XJSONSerializable
from ruka.util.migrating import Migrating, State, validate_state
from ruka_os import distributed_fs_v2 as dfs


class SavedByRemotePath(Migrating):
    """
    Use this class if you wish to save something by DFS path rather than by
    value.

    - When saved through serialization mechanisms, such as pickle or
      ruka.util.json.xdump, only the remote path is saved.
    - When saved through .save() method, the "real" object contents are saved.

    The real object contents can be saved through pickling (default),
    ruka.util.json.xdump (see SavedByRemotePathToJSON), or in a completely
    custom way.

    Usage:

    1. Derive from SavedByRemotePath.

            @dataclass
            class BPL(SavedByRemotePath):
                '''
                A [b]ig [p]ersistent [l]ist.
                '''

                remote_path: str
                l: List[float]

       SavedByRemotePath is derived from Migratable, so you automatically get
       that functionality too.

    2. Create object.

            my_list = BPL(
                remote_path='//data/my_list.bpl.pickle',
                l=list(range(999999))
            )

       At this point, you can ignore the functionality of SavedByRemotePath
       completely, and use my_list as a regular dataclass.

    3. Save object to DFS.

            my_list.save()

    4. Load object from DFS.

            my_list_copy = BPL.load('//data/my_list.bpl.pickle')

    5. By default, save() and load() use pickling to save/load real contents.
       This can be overridden by inheriting from SavedByRemotePathToJSON
       instead of SavedByRemotePath:

            @dataclass
            class BPL(SavedByRemotePathToJSON):
                ...  # same as above


            my_list = BPL([1, 2, 3], '//data/my_list.bpl.json')
            my_list.save()
            my_list_copy = BPL.load('//data/my_list.bpl.json')
            my_list_copy = xloadr('//data/my_list.bpl.json')

        Note that when you derive from SavedByRemotePathToJSON, you can use
        xloadr() too.

        Also note, that you can edit '//data/my_list.bpl.json' by hand.

    6. You can also override load() and save() methods completely to not
       pickle, not JSON, but a custom format:

        @dataclass
        class JPEGImage(SavedByRemotePath):
            remote_path: str
            img: np.ndarray  # [H, W, 3]

            def save(self):
                dfs.upload_bytes(img2jpg(self.img), self.remote_path)

            @classmethod
            def load(cls, remote_path: str) -> JPEGImage:
                return JPEGImage(
                    remote_path=remote_path,
                    img=jpg2img(dfs.download_bytes(remote_path))
                )

       This image will be then saved by reference (i.e. by remote_path)
       instead of by value.
    """
    def __new__(cls, *args, **kwargs):
        assert is_dataclass(cls)
        assert 'remote_path' in {f.name for f in fields(cls)}
        return Migrating.__new__(cls)

    def xjson_getstate(self) -> State:
        if not isinstance(self.remote_path, str):
            raise RuntimeError(
                f'Cannot serialize {type(self)}, remote_path is {self.remote_path}')
        return {'remote_path': self.remote_path, 'class': self.xjson_getclass()}

    def xjson_setstate(self, state: State):
        validate_state(state, type(self))
        obj = self.load(state['remote_path'])
        assert type(obj) is type(self)
        for f in fields(self):
            setattr(self, f.name, getattr(obj, f.name))

    def save(self):
        if not isinstance(self.remote_path, str):
            raise RuntimeError(
                f'Cannot serialize {type(self)}, '
                f'remote_path is {self.remote_path}')
        state = Migrating.xjson_getstate(self)
        dfs.upload_bytes(pickle.dumps(state), self.remote_path)

    @classmethod
    def load(cls, remote_path: str) -> 'SavedByRemotePath':
        state = pickle.loads(dfs.download_bytes(remote_path))
        obj = cls.__new__(cls)
        Migrating.xjson_setstate(obj, state)
        return obj


class SavedByRemotePathToJSON(SavedByRemotePath):
    def save(self):
        if not isinstance(self.remote_path, str):
            raise RuntimeError(
                f'Cannot serialize {type(self)}, '
                f'remote_path is {self.remote_path}')
        state = Migrating.xjson_getstate(self)
        state['class'] = self.xjson_getclass()
        state = {k: xdumpj(v) for k, v in state.items()}
        dfs.upload_str(json.dumps(state), self.remote_path)

    @classmethod
    def load(cls, remote_path: str) -> 'SavedByRemotePathToJSON':
        state = json.loads(dfs.download_str(remote_path))
        state = {k: xloadj(v) for k, v in state.items()}
        cls = XJSONSerializable.xjson_loadclass(state['class'])
        obj = cls.__new__(cls)
        Migrating.xjson_setstate(obj, state)
        return obj