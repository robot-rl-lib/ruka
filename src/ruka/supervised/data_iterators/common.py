from typing import Callable, Iterator, List, Dict
from ruka.types import Dictator
from ruka.environments.common.env import Observation
from typing import Any
from ruka.util.collate import default_collate_fn
import collections

def batchify(
    it: Iterator[Dict[str, Any]],
    batch_size: int,
    collate_fn: Callable[[List[Any]], Iterator[Dictator]] = default_collate_fn
    ) -> Iterator[Dictator]:

    while True:
        batch_raw = collections.defaultdict(list)
        for _ in range(batch_size):
            datasample = next(it)
            for k, v in datasample.items():
                batch_raw[k + '_batch'].append(v)
        yield collate_fn(batch_raw)
