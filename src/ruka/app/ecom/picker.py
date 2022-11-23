from dataclasses import dataclass
from ruka.app.ecom.ecom import SKU, Command
from ruka.environments.common.env import Episode


class PickerCommand(Command):
    pass


@dataclass
class Pick(PickerCommand):
    item: SKU


@dataclass
class Place(PickerCommand):
    pass


@dataclass
class HomeRobot(PickerCommand):
    pass


class EcomPicker:
    def handle_pick(self, cmd: Pick) -> Episode:
        raise NotImplementedError()

    def handle_place(self, cmd: Place) -> Episode:
        raise NotImplementedError()

    def handle_home(self, cmd: HomeRobot) -> Episode:
        raise NotImplementedError()
