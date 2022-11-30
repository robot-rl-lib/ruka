import json
import pickle

from dataclasses import is_dataclass, fields
from ruka.util.json import xdumpj, xloadj
from ruka.util.migrating import Migrating, State, validate_state, load_object
from ruka_os import distributed_fs_v2 as dfs


class SavedByRemotePath(Migrating):
    """
    Use this class if you wish to save something by DFS path rather than by
    value.

    - When saved through serialization mechanisms, such as pickle or
      ruka.util.json.xdump, only the remote path is saved.
    - When saved through .save() method, the "real" object contents are saved.

    The real object contents can be saved through pickling (default),
    ruka.util.json.xdump (see JSONSavedByRemotePath), or in a completely custom
    way.

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
       This can be overridden by inheriting from JSONSavedByRemotePath
       instead of SavedByRemotePath:

            @dataclass
            class BPL(JSONSavedByRemotePath):
                ...  # same as above


            my_list = BPL([1, 2, 3], '//data/my_list.bpl.json')
            my_list.save()
            my_list_copy = BPL.load('//data/my_list.bpl.json')
            my_list_copy = xloadr('//data/my_list.bpl.json')

        Note that when you derive from JSONSavedByRemotePath, you can use
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

    def __getstate__(self) -> State:
        if not isinstance(self.remote_path, str):
            raise RuntimeError(
                f'Cannot serialize {type(self)}, remote_path is {self.remote_path}')
        state = Migrating.__getstate__(self)
        return {
            'remote_path': self.remote_path,
            'class': state['class']
        }

    def __setstate__(self, state: State):
        validate_state(state, type(self))
        if not isinstance(state.get('remote_path'), str):
            raise RuntimeError(f'invalid/no remote_path in state: {state!r}')
        obj = self.load(state['remote_path'])
        assert type(obj) is type(self)
        for f in fields(self):
            setattr(self, f.name, getattr(obj, f.name))

    def save(self):
        if not isinstance(self.remote_path, str):
            raise RuntimeError(
                f'Cannot serialize {type(self)}, remote_path is {self.remote_path}')
        state = Migrating.__getstate__(self)
        dfs.upload_bytes(pickle.dumps(state), self.remote_path)

    @classmethod
    def load(cls, remote_path: str) -> 'SavedByRemotePath':
        state = pickle.loads(dfs.download_bytes(remote_path))
        obj = cls.__new__(cls)
        Migrating.__setstate__(obj, state)
        return obj


class JSONSavedByRemotePath(SavedByRemotePath):
    def save(self):
        if not isinstance(self.remote_path, str):
            raise RuntimeError(
                f'Cannot serialize {type(self)}, remote_path is {self.remote_path}')
        state = Migrating.__getstate__(self)
        for k in state:
            if k == 'class':
                continue
            state[k] = xdumpj(state[k])
        dfs.upload_str(json.dumps(state), self.remote_path)

    @classmethod
    def load(cls, remote_path: str) -> 'JSONSavedByRemotePath':
        state = json.loads(dfs.download_str(remote_path))
        for k in state:
            if k == 'class':
                continue
            state[k] = xloadj(state[k])
        obj = cls.__new__(cls)
        Migrating.__setstate__(obj, state)
        return obj