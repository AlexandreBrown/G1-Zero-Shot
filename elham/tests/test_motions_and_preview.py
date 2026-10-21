import numpy as np
import yaml
from conftest import DEFAULT_HW, REPO

import export_motions
import rotations

CONSTANTS = yaml.safe_load((REPO / "configs" / "sonic_deploy_constants.yaml").read_text())
HW_TO_ISAACLAB = np.argsort(CONSTANTS["isaaclab_to_hw"])


def test_rotation_6d_identity_and_yaw():
    np.testing.assert_allclose(rotations.rotation_6d(np.array([1.0, 0, 0, 0])), [1, 0, 0, 1, 0, 0])
    half = np.pi / 4
    q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    np.testing.assert_allclose(rotations.rotation_6d(q), [0, -1, 1, 0, 0, 0], atol=1e-12)


def write_motion(folder, frames: int = 60):
    folder.mkdir(parents=True)
    hw_to_isaaclab = HW_TO_ISAACLAB
    q = np.tile(DEFAULT_HW[hw_to_isaaclab], (frames, 1))
    q[:, 0] += np.linspace(0, 0.3, frames)

    def save(name, data):
        header = ",".join(f"c{i}" for i in range(data.shape[1]))
        np.savetxt(folder / name, data, delimiter=",", header=header, comments="")

    save("joint_pos.csv", q)
    save("joint_vel.csv", np.full((frames, 29), 0.5))
    pos = np.zeros((frames, 42))
    pos[:, 0] = np.linspace(0, 1.0, frames)
    pos[:, 2] = 0.8
    save("body_pos.csv", pos)
    save("body_quat.csv", np.tile([1.0, 0, 0, 0] * 14, (frames, 1)))
    save("body_lin_vel.csv", np.zeros((frames, 42)))
    save("body_ang_vel.csv", np.zeros((frames, 42)))


def test_motion_stats(tmp_path):
    write_motion(tmp_path / "m")
    stats = export_motions.motion_stats(tmp_path / "m", export_motions.load_joint_table())
    assert stats["frames"] == 60 and stats["frames_consistent"]
    assert stats["duration_s"] == 1.2
    assert stats["max_travel_m"] == 1.0
    assert stats["frame0_max_dev_from_stand_rad"] == 0.0
    assert stats["last_max_dev_from_stand_rad"] == 0.3
    assert stats["max_joint_speed_rad_s"] == 0.5


def test_preview_gain_factor_syntax():
    import preview_mujoco

    factors = preview_mujoco.gain_factors("0,6=0.75;12-14=1.1")
    assert factors[0] == 0.75 and factors[6] == 0.75 and factors[1] == 1.0
    assert list(factors[12:15]) == [1.1, 1.1, 1.1] and factors[15] == 1.0
    assert (preview_mujoco.gain_factors("") == 1.0).all()
