import numpy as np

from typing import List
from ruka.app.ecom.ecom import SKU, Basket, Command
from ruka.app.ecom.picker import Pick, Place
from ruka.app.ecom.scenario import Scenario
from ruka.app.ecom.world import ResetBasket


class DummyScenarioRegistry:
    def get(self, name: str) -> Scenario:
        if name not in _scenarios:
            raise KeyError(f"Scenario {name} is unknown")
        return _scenarios[name]


def get_dummy_scenario_registry() -> DummyScenarioRegistry:
    return _dummy_scenario_registry


_lemon = SKU(
    name="lemon",
    reference_img=np.ones((3, 32, 32)),
)

_lemon_pick_10_single_item_bakset = Scenario(
    name="lemon_pick_10_single_item_bakset",
    commands=[ResetBasket(Basket([_lemon]))] + ([Pick(_lemon), Place()]) * 10,
)

_lemon_pick_1_single_item_bakset = Scenario(
    name="lemon_pick_1_single_item_bakset",
    commands=[ResetBasket(Basket([_lemon]))] + ([Pick(_lemon), Place()]) * 1,
)

_scenarios = {
    _lemon_pick_10_single_item_bakset.name: _lemon_pick_10_single_item_bakset,
    _lemon_pick_1_single_item_bakset.name: _lemon_pick_1_single_item_bakset,
}

_dummy_scenario_registry = DummyScenarioRegistry()
