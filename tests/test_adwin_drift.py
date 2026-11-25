from src.active_learning import CausalQuantileThreshold, VerificationController
from src.drift import ADWINDriftManager
from src.simulator import DelayedLabelQueue


def test_adwin_error_stream_and_drift_action():
    """ADWIN receives 0-1 prediction errors and boosts discordance ratio on drift."""
    drift_mgr = ADWINDriftManager(
        delta=0.002,
        cooldown_duration=50,
        base_disc_ratio=0.30,
        boosted_disc_ratio=0.50,
    )
    ctrl = VerificationController(
        strategy="B4_Hybrid",
        window_size=500,
        budget_per_window=20,
        hybrid_top_ratio=0.70,
        seed=42,
    )

    for t in range(150):
        drift_mgr.update(y_pred=0, y_true=0, current_time=float(t))
    assert drift_mgr.get_current_disc_ratio() == 0.30
    assert not drift_mgr.is_in_cooldown()

    drift_fired = False
    for t in range(150, 450):
        if drift_mgr.update(y_pred=1, y_true=0, current_time=float(t)):
            drift_fired = True
            ctrl.set_discordance_ratio(drift_mgr.get_current_disc_ratio())
            break

    assert drift_fired is True
    assert drift_mgr.is_in_cooldown() is True
    assert drift_mgr.get_current_disc_ratio() == 0.50
    assert len(drift_mgr.drift_events) >= 1
    assert "Boost Discordance" in drift_mgr.drift_events[0]["action"]
    assert ctrl.budget_top + ctrl.budget_disc == ctrl.budget_per_window
    assert ctrl.budget_disc == 10


def test_causal_threshold_recalibrate_on_drift_no_leakage():
    """Drift recalibration uses only completed historical scores."""
    thresh = CausalQuantileThreshold(alert_rate=0.05, initial_threshold=0.50)
    for i in range(200):
        thresh.record_score(i / 200.0)
    t_before = thresh.get_threshold()
    assert t_before == 0.50  # current chunk must not alter current threshold

    thresh.end_chunk()
    t_after_chunk = thresh.get_threshold()
    assert t_after_chunk > 0.80

    # New chunk scores (0.01) must NOT leak into recalibrate_on_drift
    for _ in range(50):
        thresh.record_score(0.01)
    recal = thresh.recalibrate_on_drift(recent_tail=200)
    assert recal == t_after_chunk


def test_delayed_queue_carries_y_pred_at_arrive_for_adwin():
    """ADWIN receives y_pred_at_arrive only when label matures at t_label."""
    q = DelayedLabelQueue(
        confirm_window_sec=100.0,
        min_fraud_delay_sec=50.0,
        fast_track_delay_sec=10.0,
        unreported_fraud_rate=0.0,
        seed=42,
    )
    q.push(
        tx_id=1,
        t_arrive=0.0,
        x={"V1": 0.1},
        y_true=1,
        is_fast_track=True,
        p_hat_at_arrive=0.25,
        y_pred_at_arrive=0,
    )
    assert q.pop_ready(9.0) == []
    ready = q.pop_ready(10.0)
    assert len(ready) == 1
    assert ready[0].y_pred_at_arrive == 0
    assert ready[0].y_observed == 1
