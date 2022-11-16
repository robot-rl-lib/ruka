from typing import Tuple
from ruka.app.ecom.ecom import Command
from ruka.app.ecom.picker import EcomPicker, Pick, PickerCommand, Place
from ruka.app.ecom.scenario import Scenario
from ruka.app.ecom.world import ResetBasket, World, WorldCommand
from ruka.environments.common.env import Episode


def handle_picker_command(picker: EcomPicker, cmd: PickerCommand) -> Episode:
    if isinstance(cmd, Pick):
        return picker.handle_pick(cmd)
    elif isinstance(cmd, Place):
        return picker.handle_place(cmd)
    else:
        raise NotImplementedError(f"picker command is not supported: {cmd}")


def handle_world_command(world: World, cmd: WorldCommand) -> Episode:
    if isinstance(cmd, ResetBasket):
        return world.handle_reset_basket(cmd)
    else:
        raise NotImplementedError(f"world command is not supported: {cmd}")


def perform_scenario(
    scenario: Scenario,
    world: World,
    picker: EcomPicker,
) -> Tuple[Command, Episode]:
    # TODO: how to handle pick and place failures?
    for cmd in scenario.commands:
        if isinstance(cmd, WorldCommand):
            handle_world_command(world, cmd)
        elif isinstance(cmd, PickerCommand):
            yield cmd, handle_picker_command(picker, cmd)
