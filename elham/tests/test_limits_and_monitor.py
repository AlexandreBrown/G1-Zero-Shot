import numpy as np
import yaml

import g1_limits
import lowstate_monitor as lm
from conftest import DEFAULT_HW, REPO


def test_joint_table_is_hardware_order_with_conservative_gate():
    table = g1_limits.load_joint_table()
    assert len(table.names) == 29
    assert table.names[0] == "left_hip_pitch_joint" and table.names[12] == "waist_yaw_joint"
    np.testing.assert_allclose(table.default, DEFAULT_HW)
    rows = yaml.safe_load((REPO / "configs" / "g1_joint_limits.yaml").read_text())["joints"]
    for row in rows:
        assert row["tau_gate"] == min(row["tau_train"], row["tau_mjcf"])
        assert row["vel_gate"] == min(row["vel_train"], row["vel_urdf"])
    assert table.tau_gate[0] == 88.0 and table.tau_train[0] == 139.0


def test_deploy_constants_match_the_joint_table():
    constants = yaml.safe_load((REPO / "configs" / "sonic_deploy_constants.yaml").read_text())
    assert [r["name"] for r in constants["joints"]] == list(g1_limits.load_joint_table().names)
    assert sorted(constants["isaaclab_to_hw"]) == list(range(29))
    np.testing.assert_allclose([r["default"] for r in constants["joints"]], DEFAULT_HW)


def test_tilt():
    half = np.radians(30.0) / 2
    assert abs(float(g1_limits.tilt_deg(np.array([np.cos(half), 0.0, np.sin(half), 0.0]))) - 30.0) < 1e-6
    yaw_only = np.array([np.cos(0.5), 0.0, 0.0, np.sin(0.5)])
    assert float(g1_limits.tilt_deg(yaw_only)) < 1e-6


def test_forward_lean_sign_and_size():
    half = np.radians(7.0) / 2
    nose_down = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    assert abs(float(g1_limits.forward_lean_deg(nose_down)) - 7.0) < 1e-6
    assert abs(float(g1_limits.forward_lean_deg(nose_down * [1, 0, -1, 0])) + 7.0) < 1e-6
    roll_only = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    assert abs(float(g1_limits.forward_lean_deg(roll_only))) < 1e-6


def test_saturation_longest_run_and_share():
    thresholds = g1_limits.Thresholds()
    limit = np.full(29, 50.0)
    torque = np.zeros((100, 29))
    torque[10:13, 4] = 50.0
    torque[50, 7] = 49.5
    stats = g1_limits.saturation(torque, limit, thresholds)
    assert stats == {"longest_s": 0.06, "share": 0.04, "joint": 4}
    assert g1_limits.saturation(np.zeros((10, 29)), limit, thresholds)["joint"] == -1


def test_gate_verdict_judges_tilt_against_the_reference():
    thresholds = g1_limits.Thresholds()
    calm = {"longest_s": 0.02, "share": 0.001, "joint": 0}
    assert g1_limits.gate_verdict(False, 40.0, 38.0, 0.5, calm, 0.1, thresholds) == []
    assert g1_limits.gate_verdict(False, 40.0, 20.0, 0.5, calm, 0.1, thresholds) == [
        "tilt 40 deg vs the reference's 20 deg"
    ]
    assert g1_limits.gate_verdict(False, 56.0, 50.0, 0.5, calm, 0.1, thresholds) == ["tilt 56 deg"]
    long_run = {"longest_s": 0.14, "share": 0.03, "joint": 3}
    reasons = g1_limits.gate_verdict(True, 10.0, None, 1.2, long_run, -0.05, thresholds)
    assert reasons[0] == "fell" and len(reasons) == 5


def sample(**overrides) -> lm.Sample:
    values = dict(
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        q=DEFAULT_HW.copy(),
        dq=np.zeros(29),
        tau=np.zeros(29),
        temperature=np.full(29, 35.0),
        motor_error=np.zeros(29, dtype=int),
        mode_machine=5,
    )
    values.update(overrides)
    return lm.Sample(**values)


def test_stream_alerts():
    thresholds = g1_limits.Thresholds()
    stamps = np.arange(0, 1, 0.002)
    assert lm.stream_alerts(stamps, np.arange(len(stamps)), thresholds) == []
    gappy = np.concatenate((stamps[:100], stamps[100:] + 0.2))
    messages = [a.message for a in lm.stream_alerts(gappy, np.arange(len(gappy)), thresholds)]
    assert any("gap" in m for m in messages)
    slow = np.arange(0, 1, 0.01)
    assert any("rate" in a.message for a in lm.stream_alerts(slow, np.arange(len(slow)), thresholds))
    ticks = np.arange(len(stamps))
    ticks[10] = ticks[9]
    assert lm.stream_alerts(stamps, ticks, thresholds) == []
    ticks[11] = ticks[8]
    assert any("tick" in a.message for a in lm.stream_alerts(stamps, ticks, thresholds))
    assert lm.stream_alerts(np.array([]), np.array([]), thresholds)[0].level == "stop"


def test_state_alerts():
    table, thresholds = g1_limits.load_joint_table(), g1_limits.Thresholds()
    assert lm.state_alerts(sample(), table, thresholds) == []
    q = DEFAULT_HW.copy()
    q[5] = table.pos_hi[5] - 0.01
    assert lm.state_alerts(sample(q=q), table, thresholds)[0].level == "stop"
    hot = np.full(29, 35.0)
    hot[5] = 75.0
    assert [a.level for a in lm.state_alerts(sample(temperature=hot), table, thresholds)] == ["warn"]
    hot[5] = 90.0
    assert [a.level for a in lm.state_alerts(sample(temperature=hot), table, thresholds)] == ["stop"]
    errors = np.zeros(29, dtype=int)
    errors[3] = 7
    assert "left_knee_joint=7" in lm.state_alerts(sample(motor_error=errors), table, thresholds)[0].message


def test_preflight():
    table, thresholds = g1_limits.load_joint_table(), g1_limits.Thresholds()
    stamps = np.arange(0, 2, 0.002)
    assert lm.preflight(stamps, np.arange(len(stamps)), sample(), table, thresholds) == []
    bad_quat = sample(quat_wxyz=np.array([0.5, 0.0, 0.0, 0.0]))
    assert any("norm" in a.message for a in lm.preflight(stamps, np.arange(len(stamps)), bad_quat, table, thresholds))


def test_torso_imu_alerts():
    assert lm.torso_imu_alerts(1000, 3.0) == []
    assert lm.torso_imu_alerts(0, 3.0)[0].level == "stop"


def test_deploy_sim_effort_caps_hip_pitch_and_roll():
    table = g1_limits.load_joint_table()
    effort = g1_limits.deploy_sim_effort(table)
    assert effort[0] == 88.0 and effort[1] == 88.0 and effort[3] == 139.0 and effort[4] == 50.0
