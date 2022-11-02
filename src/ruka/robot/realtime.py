import time

from dataclasses import dataclass
from typing import Optional


@dataclass
class WatchdogParams:
    """
    See description of Watchdog.step()
    """

    dt: float
    grace_period: float   # 0 to disable
    max_fail_time: float  # 0 to disable
    max_fail_rate: float
    window_in_steps: int  # 0 to disable

    def __post_init__(self):
        assert 0 < self.dt
        assert 0 <= self.grace_period
        assert 0 <= self.max_fail_rate <= 1
        assert 0 <= self.max_fail_time
        assert 0 <= self.window_in_steps
        if self.window_in_steps > 0:
            assert self.grace_period <= self.max_fail_time


class Watchdog:
    """
    https://en.wikipedia.org/wiki/Watchdog_timer

    Unlike classic, hardware watchdog timer, requires the next step() call to
    raise an exception if a deadline was failed.

                                  grace
                  dt              period
      |<---------------------->|<-------->|
      {         step() is allowed         }{     step() is not allowed       }
      ^                        ^
       \                        \                                              .
        \--- deadline of         \--- deadline of
             previous                 current
             step()                   step()

    There is also a possibility to fail not after a single missed deadline, but
    after a certain percentage of missed deadlines among the last N. And when
    this feature is on, also a possibility to still fail after a single missed
    deadline, but if it's missed by too much.
    """

    def __init__(self, params: WatchdogParams):
        self._params = params
        self._deadline = None
        self._failed = None
        self._failed_index = None
        self._failed_count = None
        self._failed_threshold = \
            int(params.max_fail_rate * params.window_in_steps)
        self.reset()

    def reset(self):
        self._deadline = None
        self._failed = [0] * self._params.window_in_steps
        self._failed_index = 0
        self._failed_count = 0

    def step(self):
        """
        The first step() call starts the watchdog.

        N'th step() call is expected to be called no later than t0 + (N-1) * dt
        seconds of absolute time, where t0 is the time of the first step() call.

        Raise DeadlineError if either of the following conditions hold:

        A) A single deadline is failed by more than max_fail_time;
        B) Among window_in_steps last step() calls, more than
          max_fail_rate * window_in_steps calls occurred after their respective
          deadline;
        C) If window_in_steps == 0, then a single failed deadline causes an
          exception.

        A deadline is not considered failed if it is failed by more than
        grace_period.
        """
        now = time.time()

        # Start.
        if self._deadline is None:
            self._deadline = now + self._params.dt
            return

        # Determine whether the deadline was failed.
        failed_by = now - self._deadline
        failed = failed_by > self._params.grace_period

        # Set next deadline.
        self._deadline += self._params.dt

        # A. Single deadline failed by too much.
        if self._params.max_fail_time > 0:
            if failed_by > self._params.max_fail_time:
                raise DeadlineError(
                    f'single fail by more than {self._params.max_fail_time} s')

        # B. Too many fails in a window.
        if self._params.window_in_steps > 0:
            # - Update count.
            self._failed_count -= self._failed[self._failed_index]
            self._failed[self._failed_index] = int(failed)
            self._failed_count += self._failed[self._failed_index]

            # - Increase index.
            self._failed_index = \
                (self._failed_index + 1) % self._params.window_in_steps

            # - Raise error.
            if self._failed_count > self._failed_threshold:
                raise DeadlineError(
                    f'more than {self._failed_threshold} fails among past '
                    f'{self._params.window_in_steps} steps'
                )

        # C. window_in_steps == 0.
        if self._params.window_in_steps == 0:
            if failed:
                raise DeadlineError('single fail is enough')


    def get_next_deadline(self) -> Optional[float]:
        """
        Return None if not running.
        """
        return self._deadline

    def get_time_till_next_deadline(self) -> Optional[float]:
        """
        Return None if not running.
        Return negative number if deadline is failed.
        """
        if self._deadline is None:
            return None
        return self._deadline - time.time()

    def get_time_to_sleep(self) -> float:
        """
        Return time till next deadline if deadline is not failed.
        Return 0 if the next deadline is failed.
        Return 0 if not running.
        """
        if self._deadline is None:
            return 0
        return max(0, self.get_time_till_next_deadline())


class DeadlineError(Exception):
    pass