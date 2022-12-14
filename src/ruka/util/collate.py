import numpy as np
from typing import List, TypeVar

def default_collate_fn(dict_of_sequences):
    """ Collates lists of lists of Dictators or lists of lists of arraya
        # TODO: make more general
    """
    out = dict()
    for k, sequence_list in dict_of_sequences.items():
        out[k] = _collate_sequence_list(sequence_list)
    return out

def _collate_sequence_list(sequence_list):
    if len(sequence_list[0]) == 0:
        return
    if not isinstance(sequence_list[0][0], dict):
        return (np.array(sequence_list))
    out = dict()
    for key in sequence_list[0][0].keys():
        val = [[x[key] for x in sequence] for sequence in sequence_list]
        out[key] = (np.array(val)) # BS, T, *DIMS 
    return out

T = TypeVar("T")
U = TypeVar("U")

def default_collate_fn2(x: List[T]) -> U:         
    assert isinstance(x, list)
    assert x

    # T = NDArray.
    if isinstance(x[0], np.ndarray):
        assert all(isinstance(i, np.ndarray) for i in x)
        return np.stack(x, axis=0)

    # T = Dict.
    if isinstance(x[0], dict):
        assert all(isinstance(i, dict) for i in x)
        assert all(set(i.keys()) == set(x[0].keys()) for i in x)
        return {
            k: default_collate_fn2([i[k] for i in x])
            for k in x[0].keys()
        }

    # T = List.
    if isinstance(x[0], list):
        assert all(isinstance(i, list) for i in x)
        assert all(len(x[0]) == len(i) for i in x)
        return [
            default_collate_fn2([i[ii] for i in x])
            for ii in range(len(x[0]))
        ]
    
    # T = Tuple.
    if isinstance(x[0], tuple):
        assert all(isinstance(i, tuple) for i in x)
        assert all(len(x[0]) == len(i) for i in x)
        return tuple(
            default_collate_fn2([i[ii] for i in x])
            for ii in range(len(x[0]))
        )

    # T = np.int64 | np.int32 | ...
    if isinstance(x[0], (np.int64, np.int32)):
        assert all(isinstance(i, type(x[0])) for i in x)
        return np.array(x, dtype=type(x[0]))

    assert 0, type(x[0])
