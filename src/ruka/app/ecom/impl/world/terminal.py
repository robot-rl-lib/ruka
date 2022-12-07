from ruka.app.ecom.world import ResetBasket, World


class TerminalWorld(World):
    def handle_reset_basket(self, cmd: ResetBasket):
        print("Please clear the basket and place following items:")
        for item in cmd.basket.items:
            print(f"  - {item.name}")
        while _ := input("enter yes or y when done ->") not in ["y", "yes"]:
           pass

class AlwaysReadyWorld(World):
    def handle_reset_basket(self, cmd: ResetBasket):
        print("Basket should be ready:")
        for item in cmd.basket.items:
            print(f"  - {item.name}")
