import numpy as np
import pytest
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)

import g1_cdr


def random_low_state(rng: np.random.Generator):
    msg = unitree_hg_msg_dds__LowState_()
    msg.version = [int(v) for v in rng.integers(0, 2**32, 2)]
    msg.mode_pr = int(rng.integers(0, 256))
    msg.mode_machine = int(rng.integers(0, 256))
    msg.tick = int(rng.integers(0, 2**32))
    imu = msg.imu_state
    imu.quaternion = rng.normal(size=4).astype(np.float32).tolist()
    imu.gyroscope = rng.normal(size=3).astype(np.float32).tolist()
    imu.accelerometer = rng.normal(size=3).astype(np.float32).tolist()
    imu.rpy = rng.normal(size=3).astype(np.float32).tolist()
    imu.temperature = int(rng.integers(-300, 300))
    for motor in msg.motor_state:
        motor.mode = int(rng.integers(0, 256))
        motor.q, motor.dq, motor.ddq, motor.tau_est, motor.vol = rng.normal(size=5).astype(np.float32).tolist()
        motor.temperature = [int(v) for v in rng.integers(-300, 300, 2)]
        motor.sensor = [int(v) for v in rng.integers(0, 2**32, 2)]
        motor.motorstate = int(rng.integers(0, 2**32))
        motor.reserve = [int(v) for v in rng.integers(0, 2**32, 4)]
    msg.wireless_remote = [int(v) for v in rng.integers(0, 256, 40)]
    msg.reserve = [int(v) for v in rng.integers(0, 2**32, 4)]
    msg.crc = int(rng.integers(0, 2**32))
    return msg


def random_low_cmd(rng: np.random.Generator):
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = int(rng.integers(0, 256))
    msg.mode_machine = int(rng.integers(0, 256))
    for motor in msg.motor_cmd:
        motor.mode = int(rng.integers(0, 256))
        motor.q, motor.dq, motor.tau, motor.kp, motor.kd = rng.normal(size=5).astype(np.float32).tolist()
        motor.reserve = int(rng.integers(0, 2**32))
    msg.crc = int(rng.integers(0, 2**32))
    return msg


def test_layout_sizes_match_the_official_serializer():
    assert g1_cdr.message_size(g1_cdr.LOW_STATE) == len(unitree_hg_msg_dds__LowState_().serialize()) == 2092
    assert g1_cdr.message_size(g1_cdr.LOW_CMD) == len(unitree_hg_msg_dds__LowCmd_().serialize()) == 1004
    assert g1_cdr.message_size(g1_cdr.IMU_STATE) == len(unitree_hg_msg_dds__IMUState_().serialize()) == 58


def test_low_state_fields_match_the_official_serializer():
    rng = np.random.default_rng(0)
    msgs = [random_low_state(rng) for _ in range(6)]
    out = g1_cdr.decode(g1_cdr.LOW_STATE, [m.serialize() for m in msgs])
    np.testing.assert_array_equal(out["tick"], [m.tick for m in msgs])
    np.testing.assert_array_equal(out["mode_machine"], [m.mode_machine for m in msgs])
    np.testing.assert_array_equal(out["version"], [m.version for m in msgs])
    np.testing.assert_array_equal(out["imu_state.quaternion"], np.float32([m.imu_state.quaternion for m in msgs]))
    np.testing.assert_array_equal(out["imu_state.temperature"], [m.imu_state.temperature for m in msgs])
    for field in ("q", "dq", "ddq", "tau_est", "vol"):
        expected = np.float32([[getattr(motor, field) for motor in m.motor_state] for m in msgs])
        np.testing.assert_array_equal(out[f"motor_state.{field}"], expected)
    for field in ("mode", "motorstate"):
        expected = [[getattr(motor, field) for motor in m.motor_state] for m in msgs]
        np.testing.assert_array_equal(out[f"motor_state.{field}"], expected)
    np.testing.assert_array_equal(
        out["motor_state.temperature"], [[motor.temperature for motor in m.motor_state] for m in msgs]
    )
    reserve = [[motor.reserve for motor in m.motor_state] for m in msgs]
    np.testing.assert_array_equal(out["motor_state.reserve"], reserve)
    np.testing.assert_array_equal(out["wireless_remote"], [m.wireless_remote for m in msgs])
    np.testing.assert_array_equal(out["crc"], [m.crc for m in msgs])


def test_low_cmd_fields_match_the_official_serializer():
    rng = np.random.default_rng(1)
    msgs = [random_low_cmd(rng) for _ in range(4)]
    out = g1_cdr.decode(g1_cdr.LOW_CMD, [m.serialize() for m in msgs])
    for field in ("q", "dq", "tau", "kp", "kd"):
        expected = np.float32([[getattr(motor, field) for motor in m.motor_cmd] for m in msgs])
        np.testing.assert_array_equal(out[f"motor_cmd.{field}"], expected)
    np.testing.assert_array_equal(out["motor_cmd.mode"], [[motor.mode for motor in m.motor_cmd] for m in msgs])
    np.testing.assert_array_equal(out["crc"], [m.crc for m in msgs])


def test_wrong_message_size_is_refused():
    with pytest.raises(ValueError, match="1004"):
        g1_cdr.decode(g1_cdr.LOW_STATE, [unitree_hg_msg_dds__LowCmd_().serialize()])
