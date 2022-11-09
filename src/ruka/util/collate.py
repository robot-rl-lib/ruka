import numpy as np

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
