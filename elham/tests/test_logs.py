from types import SimpleNamespace

import numpy as np
from conftest import make_trial

import analyze_logs
import build_dataset
import deploy_logs
import g1_limits
import session_recorder


def test_analyze_passes_a_quiet_trial(trial):
    summary = analyze_logs.summarise(trial, g1_limits.load_joint_table(), g1_limits.Thresholds())
    assert summary["rows"] == 100
    assert summary["playing_s"] == 1.0
    assert summary["max_tilt_deg"] == 0.0
    assert abs(summary["tracking_mean_abs_joint_err_rad"] - 0.02) < 1e-4
    assert summary["target_rows_minus_state_rows"] == 0
    assert summary["gate_pass"]


def test_analyze_flags_torque_above_the_wrist_limit(tmp_path):
    trial = make_trial(tmp_path, torque_scale=20.0)
    summary = analyze_logs.summarise(trial, g1_limits.load_joint_table(), g1_limits.Thresholds())
    assert summary["peak_torque_over_limit"]["joint"].endswith(("wrist_pitch_joint", "wrist_yaw_joint"))
    assert summary["peak_torque_over_limit"]["value"] == 4.0
    assert not summary["gate_pass"]


def test_analyze_flags_tilt(tmp_path):
    half = np.radians(50.0) / 2
    trial = make_trial(tmp_path, tilt_quat=(np.cos(half), np.sin(half), 0.0, 0.0))
    summary = analyze_logs.summarise(trial, g1_limits.load_joint_table(), g1_limits.Thresholds())
    assert abs(summary["max_tilt_deg"] - 50.0) < 0.1
    assert not summary["gate_pass"]


def test_analyze_reports_the_lean_held_on_frame_0(tmp_path):
    half = np.radians(5.0) / 2
    trial = make_trial(tmp_path, ticks=400, tilt_quat=(np.cos(half), 0.0, np.sin(half), 0.0))
    summary = analyze_logs.summarise(trial, g1_limits.load_joint_table(), g1_limits.Thresholds())
    assert abs(summary["held_lean_deg"] - 5.0) < 0.05
    assert summary["ref_frame0_lean_deg"] == 0.0


def test_command_lines_run_end_to_end(trial, tmp_path):
    import json
    import subprocess
    import sys

    from conftest import REPO

    out = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "analyze_logs.py"), str(trial), "--json", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())[0]["gate_pass"]
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_dataset.py"), str(trial)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    with np.load(trial / "episode.npz") as episode:
        assert episode["q"].shape == (100, 29)


def test_non_trial_folders_are_skipped(trial, tmp_path):
    import subprocess
    import sys

    from conftest import REPO

    empty = tmp_path / "launched_but_never_started"
    (empty / "state").mkdir(parents=True)
    (empty / "state" / "q.csv").write_text("")
    summary_file = tmp_path / "summary.json"
    summary_file.write_text("[]")
    for script in ("analyze_logs.py", "build_dataset.py"):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / script), str(trial), str(empty), str(summary_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "skipped" in result.stdout


def test_dataset_pairs_inputs_with_the_next_logged_action(tmp_path):
    trial = make_trial(tmp_path)
    arrays, _ = build_dataset.build_episode(trial)
    np.testing.assert_array_equal(arrays["action_out"][:-1], arrays["action"][1:])
    assert np.isnan(arrays["action_out"][-1]).all()


def test_sim_trials_use_the_simulator_torque_limits(tmp_path):
    trial = make_trial(tmp_path, torque_scale=0.0)
    tau = trial / "state" / "motor_torque.csv"
    rows = tau.read_text().splitlines()
    header, data = rows[0], [r.split(",") for r in rows[1:]]
    for k, row in enumerate(data):
        if 10 <= k < 20:
            row[5 + 1] = "87.5"  # hip roll at the simulator's 88 Nm clip for 0.2 s
    tau.write_text("\n".join([header] + [",".join(r) for r in data]) + "\n")
    (trial / "run_manifest.txt").write_text("mode: real\n")
    real = analyze_logs.summarise(trial, g1_limits.load_joint_table(), g1_limits.Thresholds())
    (trial / "run_manifest.txt").write_text("mode: sim  iface: lo\n")
    sim = analyze_logs.summarise(trial, g1_limits.load_joint_table(), g1_limits.Thresholds())
    assert real["gate_pass"] and real["torque_saturation"]["longest_s"] == 0.0
    assert not sim["gate_pass"] and sim["torque_saturation"]["longest_s"] == 0.2


def test_post_state_rows_align_by_entry_index(tmp_path):
    trial = make_trial(tmp_path, drop_post_row=60)
    arrays, meta = build_dataset.build_episode(trial)
    assert arrays["q"].shape == (100, 29)
    assert np.isnan(arrays["token"][60]).all()
    assert not np.isnan(arrays["token"][61]).any()
    assert arrays["motion_name"][60] == ""
    assert arrays["motion_name"][61] == "asap_squat_lv1"
    assert meta["rows"]["token_state"] == 99


def test_episode_holds_targets_reference_and_inputs(trial):
    arrays, meta = build_dataset.build_episode(trial)
    np.testing.assert_allclose(arrays["q_target"][0], g1_limits.load_joint_table().default, atol=1e-9)
    assert arrays["ref_q"].shape == (100, 29)
    assert arrays["policy_input"].shape == (100, 994)
    assert arrays["motor_temperature"].shape == (100, 29, 2)
    assert arrays["motion_playing"].dtype == bool
    assert "not_aligned" not in meta
    np.testing.assert_allclose(np.diff(arrays["t_monotonic"]), 0.02, atol=1e-9)


def test_joint_targets_follow_the_deploy_formula():
    action = np.zeros((1, 29))
    action[0, 1] = 1.0
    targets = build_dataset.joint_targets(action)
    isaaclab_to_hw, default, scale = build_dataset.deploy_constants()
    hw = int(np.flatnonzero(isaaclab_to_hw == 1)[0])
    assert hw == 6
    np.testing.assert_allclose(targets[0, hw], default[hw] + scale[hw])
    np.testing.assert_allclose(np.delete(targets[0], hw), np.delete(default, hw))


def test_unindexed_reader_drops_the_trailing_comma(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("1,2,3,\n4,5,6,\n")
    np.testing.assert_array_equal(deploy_logs.read_unindexed(path), [[1, 2, 3], [4, 5, 6]])


def fake_lowstate(tick: int) -> bytes:
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

    msg = unitree_hg_msg_dds__LowState_()
    msg.tick = tick
    msg.mode_machine = 5
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    for motor in msg.motor_state:
        motor.q, motor.tau_est, motor.vol, motor.temperature = 0.1, 1.0, 48.0, [30, 31]
    return msg.serialize()


def test_recorder_chunks_merge_and_episode_includes_them(trial):
    writer = session_recorder.ChunkWriter(trial / "hw", session_recorder.CONVERTERS)
    for tick in range(5):
        writer.add("lowstate", fake_lowstate(tick), 0.002 * tick, 1000 * tick)
    writer.flush()
    for tick in range(5, 8):
        writer.add("lowstate", fake_lowstate(tick), 0.002 * tick, 1000 * tick)
    writer.flush()
    merged = session_recorder.merge_chunks(trial / "hw")
    assert writer.chunks == 2 and writer.counts["lowstate"] == 8
    np.testing.assert_array_equal(merged["lowstate/tick"], np.arange(8))
    assert merged["lowstate/q"].shape == (8, 35)
    assert merged["lowstate/temperature"].shape == (8, 35, 2)
    np.testing.assert_array_equal(merged["lowstate/t_source_ns"], 1000 * np.arange(8))
    arrays, _ = build_dataset.build_episode(trial)
    assert arrays["hw_lowstate_q"].shape == (8, 35)


def test_recorder_notices_that_its_launcher_is_gone():
    import os
    import subprocess

    assert session_recorder.process_alive(os.getpid())
    child = subprocess.Popen(["true"])
    child.wait()
    assert not session_recorder.process_alive(child.pid)


def test_sources_report_one_writer_with_repeated_ticks():
    import lowstate_sources

    info = SimpleNamespace(publication_handle=7)
    samples = [(0.0, 1_000_000 * i, fake_lowstate(tick), info) for i, tick in enumerate([1, 2, 2, 3])]
    lines = lowstate_sources.describe(samples, 1.0)
    assert "from 1 writer(s)" in lines[0]
    assert "(1, 2)" in lines[1] and "(0, 1)" in lines[1]
