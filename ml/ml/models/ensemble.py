"""Single source of truth for score combination and health-state classification.

Imported by both the ml/ training+evaluation pipeline and the backend's
inference service -- the combination/health-state logic must never be
duplicated or drift between offline evaluation and live serving.
"""

AE_WEIGHT = 0.7
IF_WEIGHT = 0.3

HEALTH_STABLE = "stable"
HEALTH_NORMAL = "normal"
HEALTH_WARNING = "warning"
HEALTH_CRITICAL = "critical"
HEALTH_FAILURE_RISK = "failure_risk"

_ESCALATION_ORDER = [
    HEALTH_STABLE,
    HEALTH_NORMAL,
    HEALTH_WARNING,
    HEALTH_CRITICAL,
    HEALTH_FAILURE_RISK,
]


def is_anomaly(ae_error: float, ae_threshold: float, if_flag: bool) -> bool:
    """Binary decision: OR rule -- either detector firing is enough to flag."""
    return bool(ae_error > ae_threshold or if_flag)


def ensemble_score(
    ae_error: float, ae_threshold: float, if_score: float, if_threshold: float
) -> float:
    """Continuous dashboard score: weighted blend of each detector, each
    normalized by its own threshold so that 1.0 == "at that detector's own
    anomaly threshold". This is the score `get_health_state` expects.
    """
    ae_norm = ae_error / ae_threshold if ae_threshold > 1e-8 else 0.0
    if_norm = if_score / if_threshold if if_threshold > 1e-8 else 0.0
    return AE_WEIGHT * ae_norm + IF_WEIGHT * if_norm


def get_health_state(
    score: float,
    anomaly_flag: bool = False,
    recent_scores: list[float] | None = None,
    predicted_steps_to_anomaly: int | None = None,
    near_term_horizon: int = 10,
) -> str:
    """Derived health-state rule (not a separate model).

    `score` and `recent_scores` are threshold-relative units from
    `ensemble_score` (1.0 == at the blended anomaly threshold). The binary
    `is_anomaly` OR-rule and this weighted-blend score can disagree (one
    detector firing hard can outvote the blend) -- `anomaly_flag` is a floor
    so a firing detector is never silently reported as "normal".
    """
    recent_scores = recent_scores or []

    if score < 1.0:
        if len(recent_scores) >= 50 and all(s < 1.0 for s in recent_scores[-50:]):
            state = HEALTH_STABLE
        else:
            state = HEALTH_NORMAL
    elif score < 1.5:
        state = HEALTH_WARNING
    elif score < 3.0:
        state = HEALTH_CRITICAL
    else:
        # Persistence rule: only escalate to Failure Risk after >=5 consecutive
        # points at/above 3x threshold, to avoid single-spike false criticals.
        recent_high = recent_scores[-4:] + [score]
        if len(recent_high) >= 5 and all(s >= 3.0 for s in recent_high):
            state = HEALTH_FAILURE_RISK
        else:
            state = HEALTH_CRITICAL

    if anomaly_flag and _ESCALATION_ORDER.index(state) < _ESCALATION_ORDER.index(HEALTH_WARNING):
        state = HEALTH_WARNING

    if predicted_steps_to_anomaly is not None and predicted_steps_to_anomaly <= near_term_horizon:
        state = _escalate(state)

    return state


def _escalate(state: str) -> str:
    idx = _ESCALATION_ORDER.index(state)
    return _ESCALATION_ORDER[min(idx + 1, len(_ESCALATION_ORDER) - 1)]
