from dataclasses import dataclass
from ruka.app.ecom.ecom import Basket, Command


class WorldCommand(Command):
    pass


@dataclass
class ResetBasket(WorldCommand):
    basket: Basket


class World:
    def handle_reset_basket(self, cmd: ResetBasket):
        raise NotImplementedError()
