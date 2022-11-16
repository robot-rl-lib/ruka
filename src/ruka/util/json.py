from typing import Dict, List, Union


JSONScalar = Union[str, float, int, bool, type(None)]
JSONArray = List['JSONSerializable']
JSONMapping = Dict[str, 'JSONSerializable']

JSONSerializable = Union[JSONScalar, JSONArray, JSONMapping]