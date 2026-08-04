from ml.models.ensemble import (
    HEALTH_CRITICAL,
    HEALTH_FAILURE_RISK,
    HEALTH_NORMAL,
    HEALTH_STABLE,
    HEALTH_WARNING,
    ensemble_score,
    get_health_state,
    is_anomaly,
)


def test_is_anomaly_ae_only():
    assert is_anomaly(ae_error=10.0, ae_threshold=5.0, if_flag=False) is True


def test_is_anomaly_if_only():
    assert is_anomaly(ae_error=1.0, ae_threshold=5.0, if_flag=True) is True


def test_is_anomaly_neither():
    assert is_anomaly(ae_error=1.0, ae_threshold=5.0, if_flag=False) is False


def test_ensemble_score_at_threshold_is_one():
    score = ensemble_score(ae_error=5.0, ae_threshold=5.0, if_score=2.0, if_threshold=2.0)
    assert abs(score - 1.0) < 1e-9


def test_ensemble_score_weighting():
    # AE alone at 2x threshold, IF at its own threshold
    score = ensemble_score(ae_error=10.0, ae_threshold=5.0, if_score=2.0, if_threshold=2.0)
    assert abs(score - (0.7 * 2.0 + 0.3 * 1.0)) < 1e-9


def test_health_state_normal_below_threshold():
    assert get_health_state(score=0.5) == HEALTH_NORMAL


def test_health_state_stable_sustained_normal():
    recent = [0.3] * 50
    assert get_health_state(score=0.3, recent_scores=recent) == HEALTH_STABLE


def test_health_state_warning():
    assert get_health_state(score=1.2) == HEALTH_WARNING


def test_health_state_critical_single_spike_not_failure_risk():
    # A single spike >= 3.0 without persistence should stay Critical, not escalate.
    assert get_health_state(score=4.0, recent_scores=[0.5, 0.5, 0.5]) == HEALTH_CRITICAL


def test_health_state_failure_risk_requires_persistence():
    recent = [3.5, 3.5, 3.5, 3.5]
    assert get_health_state(score=3.5, recent_scores=recent) == HEALTH_FAILURE_RISK


def test_health_state_escalates_on_near_term_forecast():
    state = get_health_state(score=0.5, predicted_steps_to_anomaly=3, near_term_horizon=10)
    assert state == HEALTH_WARNING  # escalated one level from Normal


def test_health_state_no_escalation_on_far_forecast():
    state = get_health_state(score=0.5, predicted_steps_to_anomaly=50, near_term_horizon=10)
    assert state == HEALTH_NORMAL


def test_health_state_anomaly_flag_floors_normal_to_warning():
    # A firing detector should never be silently reported as "normal", even
    # if the blended score itself stayed below 1.0 (see ensemble.py docstring).
    state = get_health_state(score=0.3, anomaly_flag=True)
    assert state == HEALTH_WARNING


def test_health_state_anomaly_flag_does_not_downgrade_higher_state():
    state = get_health_state(score=4.0, anomaly_flag=True, recent_scores=[0.5, 0.5, 0.5])
    assert state == HEALTH_CRITICAL
