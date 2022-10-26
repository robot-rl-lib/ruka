from typing import List

from .env import Transition, Path


class PathCollector:
    def collect_paths(self, n: int) -> List[Path]:
        """
        Does env.reset() before each path.
        """
        raise NotImplementedError()

    def collect_transtitions(self, n: int) -> List[Transition]:
        """
        Does not do env.reset() when unnecessary.
        """
        raise NotImplementedError()