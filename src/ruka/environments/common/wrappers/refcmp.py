import copy
import io
import enum
import functools
import gym
import numpy as np
import os
import pickle
import pybullet
import random
import sys

from dataclasses import dataclass
from pybullet_utils import bullet_client
from ruka.util import tensorboard as tb
from ruka.util.distributed_fs import upload_maybe
from ruka.util.random import seed_everything
from typing import Any, Dict, Tuple



@dataclass
class RefCmpConfig:
    env: gym.Env
    ref: gym.Env
    name: str = ''
    atol: float = 1e-4
    infinity: float = 1e4
    tensorboard: bool = True
    stderr: bool = True
    exception: bool = False
    breakpoint: bool = False
    pickle: bool = True
    pickle_timeout_in_steps: int = 5000


@dataclass
class _RefCmpCall:
    seed: int  # int32
    method_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class RefCmpWrapper:
    def __init__(self, config):
        assert not config.name.startswith('/')
        assert not config.name.endswith('/')
        assert '..' not in config.name
        
        self.__config = config
        self.__rng = random.Random()
        self.__step = 0
        self.__diff_sum = 0

        # Pickling.
        self.__calls_since_reset : List[_RefCmpCall] = []
        self.__pickled_env_before_reset = None
        self.__pickled_ref_before_reset = None
        self.__last_pickled_step = None
        
    def step(self, *args, **kwargs):
        return self.__call('step', args, kwargs)

    def reset(self, *args, **kwargs):
        return self.__call('reset', args, kwargs)

    def __getattr__(self, attr):
        return getattr(self.__config.env, attr)
        
    def __call(self, method_name, args, kwargs):
        seed = self.__rng.getrandbits(32)

        # Call environment.
        seed_everything(seed)
        env_method = getattr(self.__config.env, method_name)
        env_result = env_method(*args, **kwargs)

        # Call reference.
        seed_everything(seed)
        ref_method = getattr(self.__config.ref, method_name)
        ref_result = ref_method(*args, **kwargs)

        # Track.
        if self.__config.pickle:
            if method_name == 'reset':
                self.__calls_since_reset.clear()
                self.__pickled_env_before_reset = dumps(self.__config.env)
                self.__pickled_ref_before_reset = dumps(self.__config.ref)
            self.__calls_since_reset.append(_RefCmpCall(
                seed=seed, 
                method_name=method_name, 
                args=copy.deepcopy(args), 
                kwargs=copy.deepcopy(kwargs)
            ))

        # Trigger.
        diff = self.__diff(env_result, ref_result)
        if diff > self.__config.atol:
            self.__trigger(env_result, ref_result, diff)
        self.__diff_sum += diff

        # Tensorboard.
        if self.__config.tensorboard:
            infix = self.__config.name
            if infix:
                infix += '/'
            tb.scalar(f'refcmp/{infix}diff', diff, self.__step)
            tb.scalar(f'refcmp/{infix}diff_sum', self.__diff_sum, self.__step)

        # Step.
        self.__step += 1

        # Return.
        return ref_result

    def __diff(self, x, y) -> float:
        """
        Return config.infinity if structure is different, not just tensor values.
        """
        # Check types.
        if type(x) != type(y):
            return self.__config.infinity

        # Tuples and lists.
        if isinstance(x, (tuple, list)):
            if len(x) != len(y):
                return self.__config.infinity
            return max(self.__diff(i, j) for i, j in zip(x, y))

        # Dicts.
        elif isinstance(x, dict):
            return self.__diff(tuple(x.items()), tuple(y.items()))

        # Arrays.
        elif isinstance(x, np.ndarray):
            if x.shape != y.shape or x.dtype != y.dtype:
                return self.__config.infinity
            return np.max(np.abs(x - y))

        # Other types.
        else:
            if x == y:
                return 0
            else:
                return self.__config.infinity

    def __trigger(self, env_result, ref_result, diff):
        # Compose message.
        msg = (
            f'REFCMP: environment {self.__config.env} acts '
            f'differently from reference {self.__config.ref}: '
            f'{env_result} != {ref_result} with max(|a - b|) = {diff} '
            f'on step {self.__step}.'
        )

        # Stderr.
        if self.__config.stderr:
            print(msg, file=sys.stderr)

        # Exception.
        if self.__config.exception:
            raise RuntimeError(msg)

        # Breakpoint.
        if self.__config.breakpoint:
            breakpoint()

        # Pickle.
        last_pickled = self.__last_pickled_step
        pickle_timeout = self.__config.pickle_timeout_in_steps
        if last_pickled is None or self.__step - last_pickled > pickle_timeout:
            self.__last_pickled_step = self.__step
            self.__pickle()

    def __pickle(self):
        # Code body.
        body = ''
        for i, call in enumerate(self.__calls_since_reset):
            # - Seed.
            body += f'    seed_everything({call.seed})\n'

            # - Args.
            args = [repr(a) for a in call.args]
            args += [f'{k}={repr(v)}' for k, v in call.kwargs.items()]
            args = ', '.join(args)

            # - Return.
            ret = ''
            if i + 1 == len(self.__calls_since_reset):
                ret = 'return '

            # - Compile.
            body += f'    {ret}env.{call.method_name}({args})\n'

        # Code.
        code = (
            'from numpy import *\n'
            'from ruka.environments.common.wrappers.refcmp import loads\n'
            'from ruka.util.random import seed_everything\n'
            '\n'
            'def load_env():\n'
           f'    with open("step_{self.__step}_env.pickle", "rb") as f:\n'
           f'        return loads(f.read())\n'
            '\n'
            '\n'
            'def load_ref():\n'
           f'    with open("step_{self.__step}_ref.pickle", "rb") as f:\n'
           f'        return loads(f.read())\n'
            '\n'
            '\n'
            'def test(env):\n'
           f'{body}'
            '\n'
            '\n'
            'assert test(load_env()) == test(load_ref())\n'
        )

        # Create dir.
        dir = 'refcmp'
        if self.__config.name:
            dir = dir + '/' + self.__config.name
        os.makedirs(dir, exist_ok=True)
        print(f'REFCMP: pickling to {dir}/ on step {self.__step}', file=sys.stderr)

        # Pickle.
        env_path = f'{dir}/step_{self.__step}_env.pickle'
        ref_path = f'{dir}/step_{self.__step}_ref.pickle'
        py_path = f'{dir}/step_{self.__step}.py'

        with open(env_path, 'wb') as f:
            f.write(self.__pickled_env_before_reset)
        with open(ref_path, 'wb') as f:
            f.write(self.__pickled_ref_before_reset)
        with open(py_path, 'wt') as f:
            f.write(code)

        # Upload.
        upload_maybe(env_path, wait=True)
        upload_maybe(ref_path, wait=True)
        upload_maybe(py_path, wait=True)


def dumps(obj: Any) -> bytes:
    f = io.BytesIO()
    p = _Pickler(f)
    p.dump(obj)
    return f.getvalue()


def loads(data: bytes) -> Any:
    f = io.BytesIO(data)
    p = _Unpickler(f)
    return p.load()


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if obj is np.random:
            return ('module', 'np.random')
        if isinstance(obj, bullet_client.BulletClient):
            return ('BulletClient', id(obj))
        return None


class _Unpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__memo = {}  # pid => object

    def persistent_load(self, pid):
        if pid in self.__memo:
            return self.__memo[pid]

        if pid == ('module', 'np.random'):
            return np.random
        if pid[0] == 'BulletClient':
            obj = bullet_client.BulletClient(pybullet.DIRECT)
            self.__memo[pid] = obj
            return obj

        return super().persistent_load(pid)