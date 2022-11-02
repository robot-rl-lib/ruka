import pytest
import time

from ruka.robot.realtime import Watchdog, WatchdogParams, DeadlineError


def test_watchdog_param_checks():
    WatchdogParams(
        dt=0.01,
        grace_period=0,
        max_fail_time=0,
        max_fail_rate=0,
        window_in_steps=0
    )

    WatchdogParams(
        dt=0.01,
        grace_period=0.001,
        max_fail_time=0,
        max_fail_rate=0,
        window_in_steps=0
    )

    with pytest.raises(Exception):
        WatchdogParams(
            dt=0,
            grace_period=0,
            max_fail_time=0,
            max_fail_rate=0,
            window_in_steps=0
        )

    with pytest.raises(Exception):
        WatchdogParams(
            dt=0.01,
            grace_period=-1,
            max_fail_time=0,
            max_fail_rate=0,
            window_in_steps=0
        )

    with pytest.raises(Exception):
        WatchdogParams(
            dt=0.01,
            grace_period=0,
            max_fail_time=-1,
            max_fail_rate=0,
            window_in_steps=0
        )

    with pytest.raises(Exception):
        WatchdogParams(
            dt=0.01,
            grace_period=0,
            max_fail_time=0,
            max_fail_rate=-0.1,
            window_in_steps=0
        )

    with pytest.raises(Exception):
        WatchdogParams(
            dt=0.01,
            grace_period=0,
            max_fail_time=0,
            max_fail_rate=1.1,
            window_in_steps=0
        )


    with pytest.raises(Exception):
        WatchdogParams(
            dt=0.01,
            grace_period=0,
            max_fail_time=0,
            max_fail_rate=0,
            window_in_steps=-1
        )

    with pytest.raises(Exception):
        WatchdogParams(
            dt=0.01,
            grace_period=0.1,
            max_fail_time=0.01,
            max_fail_rate=0,
            window_in_steps=-1
        )


def test_watchdog_vanilla(monkeypatch):
    params = WatchdogParams(
        dt=0.1,
        grace_period=0,
        max_fail_time=0,
        max_fail_rate=0,
        window_in_steps=0
    )

    monkeypatch.setattr(time, 'time', lambda: 42.)
    watchdog = Watchdog(params)

    monkeypatch.setattr(time, 'time', lambda: 42.1)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.19)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.3)
    watchdog.step()

    with pytest.raises(DeadlineError):
        monkeypatch.setattr(time, 'time', lambda: 42.41)
        watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.6)
    watchdog.reset()

    monkeypatch.setattr(time, 'time', lambda: 42.7)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.81)
    watchdog.reset()
    watchdog.step()


def test_watchdog_grace_period(monkeypatch):
    params = WatchdogParams(
        dt=0.1,
        grace_period=0.01,
        max_fail_time=0,
        max_fail_rate=0,
        window_in_steps=0
    )

    monkeypatch.setattr(time, 'time', lambda: 42.)
    watchdog = Watchdog(params)

    monkeypatch.setattr(time, 'time', lambda: 42.1)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.19)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.3)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.41)
    watchdog.step()


    with pytest.raises(DeadlineError):
        monkeypatch.setattr(time, 'time', lambda: 42.511)
        watchdog.step()


def test_watchdog_max_fail_rate(monkeypatch):
    params = WatchdogParams(
        dt=0.1,
        grace_period=0,
        max_fail_time=0,
        max_fail_rate=0.5,
        window_in_steps=4
    )

    monkeypatch.setattr(time, 'time', lambda: 42.)
    watchdog = Watchdog(params)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.1)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.2)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.3)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.41)
    watchdog.step()  # fail 1

    monkeypatch.setattr(time, 'time', lambda: 42.51)
    watchdog.step()  # fail 2

    with pytest.raises(DeadlineError):
        monkeypatch.setattr(time, 'time', lambda: 42.61)
        watchdog.step()  # fail 3


def test_watchdog_max_fail_time(monkeypatch):
    params = WatchdogParams(
        dt=0.1,
        grace_period=0,
        max_fail_time=0.02,
        max_fail_rate=0.5,
        window_in_steps=4
    )

    monkeypatch.setattr(time, 'time', lambda: 42.)
    watchdog = Watchdog(params)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.1)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.2)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.3)
    watchdog.step()

    monkeypatch.setattr(time, 'time', lambda: 42.42)
    watchdog.step()

    with pytest.raises(DeadlineError):
        monkeypatch.setattr(time, 'time', lambda: 42.521)
        watchdog.step()
