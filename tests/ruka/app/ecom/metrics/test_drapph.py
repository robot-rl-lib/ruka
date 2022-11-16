import math

from ruka.app.ecom.metrics.drapph import calc_drapph


def test_drapph():

    dr_apph = calc_drapph(
        num_cobots=1,
        num_operators=1,
        num_attempts=10000,
        success_rate=1,
        time_success_sec=[1],
        time_failure_sec=[1],
        time_to_fix_sec=[1],
    )

    assert math.isclose(dr_apph, 3600)

    dr_apph = calc_drapph(
        num_cobots=1,
        num_operators=1,
        num_attempts=10000,
        success_rate=0,
        time_success_sec=[1],
        time_failure_sec=[1],
        time_to_fix_sec=[1],
    )

    assert math.isclose(dr_apph, 0)

    dr_apph = calc_drapph(
        num_cobots=1,
        num_operators=1,
        num_attempts=10000,
        success_rate=0.95,
        time_success_sec=[5],
        time_failure_sec=[10],
        time_to_fix_sec=[120],
    )

    assert 250 < dr_apph < 400
