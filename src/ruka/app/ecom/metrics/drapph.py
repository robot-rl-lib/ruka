import random

from typing import List


def sample_from_list(alist: List[float]):
    return alist[random.randint(0, len(alist) - 1)]


def calc_drapph(
    num_cobots: int,
    num_operators: int,
    num_attempts: int,
    success_rate: float,
    time_success_sec: List[float],
    time_failure_sec: List[float],
    time_to_fix_sec: List[float],
) -> float:
    """
    Calculate DRAPPH metric which is
    disengagement rate - adjusted picks per hour.
    Metric is calculated using monte carlo simulation.
    Now only one cobot and one operator is supported.

    Args:
        num_cobots (int): number of cobots in simulation.
        num_operators (int): number of operators in simulation.
        num_attempts (int): number of simulated attempts.
        success_rate (float): task success rate.
        time_success_sec (List[float]): list of succesful attempts time samples.
        time_failure_sec (List[float]): list of failure attempts time samples.
        time_to_fix_sec (List[float]): list of cobot repair time samples.

    Raises:
        NotImplementedError: raised if num_cobots != 1
        NotImplementedError: raised if num_operators != 1

    Returns:
        float: metric value
    """

    if num_cobots != 1:
        raise NotImplementedError("Supported only one cobot")
    if num_operators != 1:
        raise NotImplementedError("Supported only one operator")

    num_succesful_picks = 0
    time_elapsed = 0

    for _ in range(num_attempts):
        is_success = random.uniform(0, 1) < success_rate
        num_succesful_picks += 1 if is_success else 0
        time_elapsed += sample_from_list(
            time_success_sec if is_success else time_failure_sec
        )
        time_elapsed += 0 if is_success else sample_from_list(time_to_fix_sec)

    return num_succesful_picks * 3600 / time_elapsed
