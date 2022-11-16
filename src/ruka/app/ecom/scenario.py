from dataclasses import dataclass
from typing import List
from ruka.app.ecom.ecom import Command


@dataclass
class Scenario:
    name: str
    commands: List[Command]

    def __str__(self) -> str:
        return f"{self.name} scenario"


class ScenarioRegistry:
    def get(self, name: str) -> Scenario:
        raise NotImplementedError()
