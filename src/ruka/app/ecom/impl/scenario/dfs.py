import tempfile
import numpy as np
import yaml

from typing import Any, List, Type
from ruka.app.ecom.ecom import SKU, Basket
from ruka.app.ecom.picker import HomeRobot, Pick, Place
from ruka.app.ecom.scenario import Scenario
from ruka.app.ecom.world import ResetBasket
from ruka.util.compression import jpg2img
from ruka.util.nested_dict import map_inplace
from ruka_os.distributed_fs_v2 import download


class DFSScenarioRegistry:
    """
    Get scenario stored on DFS
    """

    def get(self, uri: str) -> Scenario:
        """
        Args:
            uri (str): DFS URI of the scenario

        Returns:
            Scenario
        """
        dfs_loader = DFSObjectLoader(
            [SKU, Scenario, ResetBasket, Basket, HomeRobot, Pick, Place]
        )
        return dfs_loader.load(uri)


class DFSObjectLoader:
    """
    Loads dataclasses and simple types stored on DFS.
    For example

    yaml file at dfs://app/ecom/scenarios/lemon_pick_10_single_item_bakset.yaml:
    type: Scenario
    args:
        name: lemon_pick_10_single_item_bakset
        commands:
        - type: ResetBasket
            args:
            basket:
                type: Basket
                args:
                items:
                    - dfs://app/ecom/sku/lemon/info.yaml
        - type: HomeRobot
        - type: Pick
            args:
            item: dfs://app/ecom/sku/lemon/info.yaml

    will be loaded as
    @dataclass
    Scenario:
        uri: dfs://app/ecom/scenarios/lemon_pick_10_single_item_bakset.yaml
        name: lemon_pick_10_single_item_bakset
        commands: [...]

    yaml file at dfs://app/ecom/sku/lemon/info.yaml:
    type: SKU
    args:
        name: lemon
        reference_img: dfs://app/ecom/sku/lemon/lemon.jpeg

    will be loaded as
    @dataclass
    SKU:
        url: dfs://app/ecom/sku/lemon/info.yaml
        name: lemon
        reference_img: np.ndarray
    """

    def __init__(self, types: List[Type]):
        """
        Args:
            types (List[Type]): list of supported types
        """
        self._types = {t.__name__: t for t in types}
        self._cache = {}

    def load(self, uri: str) -> Any:
        """
        Args:
            uri (str): DFS URI of the object

        Raises:
            NotImplementedError: if DFS URI not found or object type is not supported

        Returns:
            Any: loaded object
        """
        if uri in self._cache:
            return self._cache[uri]

        if uri.endswith(".yaml"):
            obj = self._load_yaml(self._load_uri_content(uri))
            obj = self._init_obj(obj)
            obj.uri = uri
        elif uri.endswith(".jpeg"):
            obj = self._load_jpeg(self._load_uri_content(uri))
        else:
            raise NotImplementedError("file type is not supported")

        self._cache[uri] = obj
        return obj

    def _init_obj(self, obj):
        def _init(raw_obj):
            if isinstance(raw_obj, dict) and "type" in raw_obj:
                type_name = raw_obj["type"]
                if type_name not in self._types:
                    raise NotImplementedError(f"type {type_name} is not supported")

                if "args" in raw_obj:
                    obj = self._types[type_name](**raw_obj["args"])
                else:
                    obj = self._types[type_name]()
                return obj

            elif isinstance(raw_obj, str) and raw_obj.startswith("dfs://"):
                return self.load(raw_obj)

            return raw_obj

        return map_inplace(_init, obj, enter_lists=True, apply_on_nodes=True)

    def _load_yaml(self, content: str) -> Any:
        return yaml.safe_load(content)

    def _load_jpeg(self, content) -> np.ndarray:
        return jpg2img(np.fromstring(content, np.uint8))

    def _load_uri_content(self, uri: str):
        with tempfile.NamedTemporaryFile() as tmp_f:
            try:
                download(uri[6:], tmp_f.name)
            except ValueError as ex:
                raise KeyError(f"Uri {uri} is not found. Underlying exception: {ex}")

            with open(tmp_f.name, "rb") as f:
                return f.read()


_dfs_scenario_registry = DFSScenarioRegistry()


def get_dfs_scenario_registry() -> DFSScenarioRegistry:
    return _dfs_scenario_registry
