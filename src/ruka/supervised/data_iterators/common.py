import time
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


class Batchify():
    """ Iterator over batchs which support restart with __iter__ """

    def __init__(self, 
                make_iter: Callable,
                batch_size: int,
                collate_fn: Callable[[List[Any]], Iterator[Dictator]] = default_collate_fn,
                infinity: bool=True,
                partly_batch: bool=False,):
        """
        Args:
            make_iter: function for create itrator over batch items
            batch_size: batc size
            collate_fn: function for collate batch
            infinity: if True continue with restart item iterator (call make_iter again) 
            partly_batch: if True can return batch less than batch_size when item iter finished            
        """
        self._make_iter = make_iter
        self._it = None
        self.batch_size = batch_size
        self._collate_fn = collate_fn
        self._infinity = infinity
        self._partly_batch = partly_batch
        self._stop_iter = False


    def __iter__(self):
        self._stop_iter = False
        self._it = self._make_iter()
        return self

    def __next__(self):
        if self._stop_iter and not self._infinity:
            raise StopIteration()
        elif self._stop_iter and self._infinity:
            self._it = self._make_iter()
            self._stop_iter = False

        batch_raw = collections.defaultdict(list)
        for i in range(self.batch_size):
            try:
                datasample = next(self._it)
            except StopIteration:
                self._stop_iter = True
                break

            for k, v in datasample.items():
                batch_raw[k + '_batch'].append(v)            
            
        if i + 1 == self.batch_size or (i > 0 and self._partly_batch):
            return  self._collate_fn(batch_raw)
        else:
            if not self._infinity:
                raise StopIteration()
            else:
                return self.__next__()


