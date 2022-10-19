
def smart_shape(object):
    """ Custom object shape for debug """
    if object is None:
        return None
    elif isinstance(object, (int, float, str)):
        return f'{type(object)}: {object}'
    elif isinstance(object, (list, tuple)):
        return [f'len: {len(object)}'] + [smart_shape(i) for i in object]
    elif hasattr(object, 'shape'):
        return object.shape
    elif isinstance(object, dict):
        return {k: smart_shape(v) for k,v in object.items()}
    else:
        return f'Unknow type: {type(object)}'