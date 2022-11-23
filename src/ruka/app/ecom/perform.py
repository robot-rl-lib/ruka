from typing import Iterator
from ruka.app.ecom.picker import EcomPicker, HomeRobot, Pick, PickerCommand, Place
from ruka.app.ecom.scenario import Scenario
from ruka.app.ecom.world import ResetBasket, World, WorldCommand
from ruka.environments.common.env import Episode


def handle_world_command(world: World, cmd: WorldCommand) -> Episode:
    if isinstance(cmd, ResetBasket):
        return world.handle_reset_basket(cmd)
    else:
        raise NotImplementedError(f"world command is not supported: {cmd}")


def fill_meta(ep: Episode):
    last_info = ep.infos[-1]
    ep.meta['pick_time'] = last_info['timestamp_finish_step'] - ep.infos[0]['timestamp_start_step']
    if 'is_time_limit' in last_info:
        ep.meta['is_time_limit'] = last_info['is_time_limit']
    ep.meta['is_pick_successful'] = last_info['is_success']
    ep.meta['give_up'] = not last_info['is_success']


def perform_scenario(
    scenario: Scenario,
    world: World,
    picker: EcomPicker,
) -> Iterator[Episode]:
    """
    Performs data collection scenario.
    Yields:
        Iterator[Episode]: episodes during data collection.
    """

    # TODO: refactor scenarios into:
    # Scenario:
    #   basket: Basket
    #   items: [PickAndPlace]
    # PickAndPlace:
    #   item: SKU
    #   dst: ?
    # and rewrite this function into explicit logic

    for cmd in scenario.commands:
        if isinstance(cmd, WorldCommand):
            handle_world_command(world, cmd)
        elif isinstance(cmd, PickerCommand):

            # here we can handle picker failures
            if isinstance(cmd, Pick):
                episode = picker.handle_pick(cmd)
                fill_meta(episode)

                home_ep = picker.handle_home(cmd)
                episode.meta['home_time'] = home_ep.meta['episode_time']
                episode.meta['episode_time'] = episode.meta['pick_time'] + episode.meta['home_time']
                yield episode

            elif isinstance(cmd, Place):
                raise NotImplementedError("Implement place logic")
            elif isinstance(cmd, HomeRobot):
                print(f"HomeRobot command is deprecated")
            else:
                raise NotImplementedError(f"picker command is not supported: {cmd}")
