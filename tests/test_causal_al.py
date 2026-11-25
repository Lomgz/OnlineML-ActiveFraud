import numpy as np
import pandas as pd

from src.active_learning import CausalQuantileThreshold, VerificationController
from src.metrics import compute_metrics
from src.models import BatchXGBoostWrapper, OnlineFraudModel, SemiSupervisedHST
from src.simulator import DelayedLabelQueue, prepare_temporal_split


def test_no_data_leakage_delayed_queue():
    q = DelayedLabelQueue(
        confirm_window_sec=1000.0,
        min_fraud_delay_sec=400.0,
        fast_track_delay_sec=50.0,
        unreported_fraud_rate=0.0,
        seed=42,
    )
    q.push(tx_id=1, t_arrive=100.0, x={"f1": 1.0}, y_true=0, is_fast_track=False)
    q.push(tx_id=2, t_arrive=100.0, x={"f1": 2.0}, y_true=1, is_fast_track=False)
    q.push(tx_id=3, t_arrive=100.0, x={"f1": 3.0}, y_true=1, is_fast_track=True)

    assert q.pop_ready(149.9) == []

    ready_150 = q.pop_ready(150.0)
    assert [e.tx_id for e in ready_150] == [3]
    assert all(e.t_label <= 150.0 for e in ready_150)

    assert q.pop_ready(399.0) == []

    ready_1100 = q.pop_ready(1100.0)
    assert sorted(e.tx_id for e in ready_1100) == [1, 2]
    assert all(e.t_label <= 1100.0 for e in ready_1100)
    assert q.pending_count() == 0


def test_budget_cap_all_strategies():
    rng = np.random.default_rng(7)
    window_size = 200
    budget = 10

    for strategy in VerificationController.VALID_STRATEGIES:
        ctrl = VerificationController(
            strategy=strategy,
            window_size=window_size,
            budget_per_window=budget,
            hybrid_top_ratio=0.70,
            seed=7,
        )
        for _ in range(3):
            window_queries = 0
            for _ in range(window_size):
                p_hat = float(rng.uniform(0.0, 1.0))
                hst_score = float(rng.uniform(0.0, 1.0))
                dec = ctrl.decide(p_hat=p_hat, hst_score=hst_score, decision_threshold=0.5)
                if dec.should_query:
                    window_queries += 1
            if strategy == "B0_NoAL":
                assert window_queries == 0
            else:
                assert window_queries <= budget
                if strategy == "B4_Hybrid":
                    assert ctrl.used_top_in_window <= ctrl.budget_top
                    assert ctrl.used_disc_in_window <= ctrl.budget_disc
            ctrl.end_window()


def test_semi_supervised_hst_isolation():
    hst = SemiSupervisedHST(min_warmup_normals=10, seed=42)
    rng = np.random.default_rng(123)

    for _ in range(25):
        x = {"v1": float(rng.normal(0, 1)), "v2": float(rng.normal(0, 1))}
        assert hst.learn_one(x, y=0) is True

    probe = {"v1": 5.0, "v2": -4.0}
    score_before = hst.score_one(probe)
    normals_before = hst.n_normal_seen

    for _ in range(15):
        assert hst.learn_one({"v1": 99.0, "v2": -99.0}, y=1) is False

    assert hst.n_normal_seen == normals_before
    assert hst.score_one(probe) == score_before


def test_causal_quantile_threshold():
    thresh_a = CausalQuantileThreshold(alert_rate=0.1, initial_threshold=0.5)
    thresh_b = CausalQuantileThreshold(alert_rate=0.1, initial_threshold=0.5)

    for s in np.linspace(0.0, 0.9, 100):
        thresh_a.record_score(float(s))
        thresh_b.record_score(float(s))
    t1_a = thresh_a.end_chunk()
    t1_b = thresh_b.end_chunk()
    assert t1_a == t1_b

    for _ in range(50):
        thresh_a.record_score(0.01)
        thresh_b.record_score(0.99)
        assert thresh_a.get_threshold() == t1_a
        assert thresh_b.get_threshold() == t1_b


def test_batch_parity_and_online_model_smoke():
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float) * 100.0,
            "V1": rng.normal(0, 1, n),
            "V2": rng.normal(0, 1, n),
            "Class": np.where(rng.random(n) < 0.15, 1, 0),
        }
    )

    matured, pending, test_df, feature_cols = prepare_temporal_split(
        df, warmup_ratio=0.5, confirm_window_sec=1000.0
    )
    assert "Time" not in feature_cols and "Class" not in feature_cols
    t_split = float(test_df.iloc[0]["Time"])
    assert (matured["Time"] + 1000.0 <= t_split).all()
    assert (pending["Time"] + 1000.0 > t_split).all()

    X_init = matured[feature_cols].to_numpy(dtype=np.float32)
    y_init = matured["Class"].to_numpy(dtype=np.int32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    static_clf = BatchXGBoostWrapper(feature_cols, mode="static", n_estimators=10, seed=42)
    static_clf.fit(X_init, y_init)
    preds_before = static_clf.predict_proba_batch(X_test)

    assert static_clf.update_if_allowed(X_test, test_df["Class"].to_numpy()) is False
    assert static_clf.fit_count == 1
    np.testing.assert_allclose(preds_before, static_clf.predict_proba_batch(X_test))

    periodic_clf = BatchXGBoostWrapper(feature_cols, mode="periodic", n_estimators=10, seed=42)
    periodic_clf.fit(X_init, y_init)
    assert periodic_clf.update_if_allowed(X_test, test_df["Class"].to_numpy()) is True
    assert periodic_clf.fit_count == 2

    oml = OnlineFraudModel(n_models=3, seed=42)
    x0 = {"V1": 0.5, "V2": -0.3}
    p0 = oml.predict_proba_one(x0)
    assert 0.0 <= p0 <= 1.0
    oml.learn_one(x0, 1)
    assert oml.n_updates == 1

    m = compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], k=2, latencies_ms=[1.0, 2.0])
    assert m["Pr@k"] == 1.0 and m["Recall"] == 1.0 and m["FP"] == 0.0