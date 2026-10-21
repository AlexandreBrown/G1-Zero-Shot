"""sonic_run.sh against a fake SONIC tree whose 'deploy binary' records its arguments."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import REPO

SCRIPT = REPO / "scripts" / "sonic_run.sh"


@pytest.fixture
def fake_tree(tmp_path: Path) -> dict:
    sonic = tmp_path / "sonic"
    deploy = sonic / "gear_sonic_deploy"
    binary = deploy / "target" / "release" / "g1_deploy_onnx_ref"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_ARGS_OUT"\n')
    binary.chmod(0o755)
    (deploy / "scripts").mkdir()
    (deploy / "scripts" / "setup_env.sh").write_text("export FAKE_ENV=1\n")
    trt = tmp_path / "trt" / "include"
    trt.mkdir(parents=True)
    machine = os.uname().machine
    major, minor = ("10", "13") if machine == "x86_64" else ("10", "7")
    (trt / "NvInferVersion.h").write_text(
        f"#define TRT_MAJOR_ENTERPRISE 0\n#define NV_TENSORRT_MAJOR {major}\n#define NV_TENSORRT_MINOR {minor}\n"
    )
    policy = tmp_path / "policy"
    policy.mkdir()
    for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"):
        (policy / name).write_text(name)
    (policy / "manifest.json").write_text(json.dumps({"label": "test", "problems": []}))
    (policy / "parity.json").write_text(json.dumps({"pass": True}))
    motions = tmp_path / "motions" / "squat_lv1" / "asap_squat_lv1"
    motions.mkdir(parents=True)
    (motions / "joint_pos.csv").write_text("joint_0\n0.0\n")
    env = dict(os.environ, TensorRT_ROOT=str(tmp_path / "trt"), FAKE_ARGS_OUT=str(tmp_path / "args.txt"))
    env.pop("TMUX", None)
    return {"sonic": sonic, "policy": policy, "motions": motions.parent, "env": env, "tmp": tmp_path}


def run(tree: dict, *extra: str) -> subprocess.CompletedProcess:
    args = [
        "bash",
        str(SCRIPT),
        "--sonic-root",
        str(tree["sonic"]),
        "--policy",
        str(tree["policy"]),
        "--motions",
        str(tree["motions"]),
        "--session",
        "s1",
        "--log-root",
        str(tree["tmp"] / "logs"),
        *extra,
    ]
    return subprocess.run(args, env=tree["env"], capture_output=True, text=True, timeout=60)


def test_sim_run_passes_logging_flags_and_writes_a_manifest(fake_tree):
    result = run(fake_tree, "--sim", "--trial", "t1", "--no-recorder")
    assert result.returncode == 0, result.stderr
    args = (fake_tree["tmp"] / "args.txt").read_text().split("\n")
    assert args[0] == "lo"
    assert args[1] == str(fake_tree["policy"] / "model_decoder.onnx")
    for flag in (
        "--disable-crc-check",
        "--enable-csv-logs",
        "--target-motion-logfile",
        "--policy-input-logfile",
        "--record-input-file",
        "--encoder-file",
        "--obs-config",
    ):
        assert flag in args
    assert "--planner-file" not in args
    log_dir = fake_tree["tmp"] / "logs" / "s1" / "t1"
    manifest = (log_dir / "run_manifest.txt").read_text()
    assert "mode: sim" in manifest and "squat_lv1" in manifest and "model_decoder.onnx" in manifest
    assert (log_dir / "policy_manifest.json").exists()


def test_refuses_to_overwrite_a_trial(fake_tree):
    assert run(fake_tree, "--sim", "--trial", "t1", "--no-recorder").returncode == 0
    second = run(fake_tree, "--sim", "--trial", "t1", "--no-recorder")
    assert second.returncode != 0 and "exists" in second.stderr


def test_refuses_policy_without_manifest(fake_tree):
    (fake_tree["policy"] / "manifest.json").unlink()
    result = run(fake_tree, "--sim", "--trial", "t2", "--no-recorder")
    assert result.returncode != 0 and "manifest.json" in result.stderr


def test_real_needs_an_explicit_interface(fake_tree):
    result = run(fake_tree, "--real", "--trial", "t3", "--no-recorder")
    assert result.returncode != 0 and "--iface" in result.stderr
    result = run(fake_tree, "--real", "--iface", "lo", "--trial", "t3", "--no-recorder")
    assert result.returncode != 0


def test_wrong_tensorrt_version_is_refused(fake_tree):
    header = fake_tree["tmp"] / "trt" / "include" / "NvInferVersion.h"
    header.write_text("#define NV_TENSORRT_MAJOR 10\n#define NV_TENSORRT_MINOR 3\n")
    result = run(fake_tree, "--sim", "--trial", "t4", "--no-recorder")
    assert result.returncode != 0 and "TensorRT 10.3" in result.stderr


def test_tensorrt_version_through_macro_indirection(fake_tree):
    machine = os.uname().machine
    major, minor = ("10", "13") if machine == "x86_64" else ("10", "7")
    header = fake_tree["tmp"] / "trt" / "include" / "NvInferVersion.h"
    header.write_text(
        f"#define TRT_MAJOR_ENTERPRISE {major}\n#define TRT_MINOR_ENTERPRISE {minor}\n"
        "#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< major\n"
        "#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE //!< minor\n"
    )
    assert run(fake_tree, "--sim", "--trial", "t8", "--no-recorder").returncode == 0


def test_dry_run_prints_both_commands_and_writes_nothing(fake_tree):
    result = run(fake_tree, "--sim", "--trial", "t5", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "recorder:" in result.stdout and "session_recorder.py" in result.stdout
    assert "g1_deploy_onnx_ref" in result.stdout
    assert not (fake_tree["tmp"] / "logs").exists()


def test_unknown_option_is_an_error(fake_tree):
    result = run(fake_tree, "--sim", "--trial", "t6", "--enable-csv-logs")
    assert result.returncode != 0 and "unknown option" in result.stderr


def test_missing_recorder_python_is_an_error(fake_tree):
    result = run(fake_tree, "--sim", "--trial", "t7", "--recorder-python", "/nonexistent/python")
    assert result.returncode != 0 and "unitree_sdk2py" in result.stderr


def test_real_refuses_a_policy_whose_parity_check_did_not_pass(fake_tree):
    (fake_tree["policy"] / "parity.json").write_text(json.dumps({"pass": False}))
    result = run(fake_tree, "--real", "--iface", "lo", "--trial", "t9", "--no-recorder")
    assert result.returncode != 0 and "parity" in result.stdout + result.stderr


def test_real_requires_manager_input_and_tmux(fake_tree):
    result = run(
        fake_tree, "--real", "--iface", "eth-fake", "--input-type", "keyboard", "--trial", "t10", "--no-recorder"
    )
    assert result.returncode != 0 and "manager" in result.stderr
    result = run(fake_tree, "--real", "--iface", "eth-fake", "--trial", "t11", "--no-recorder")
    assert result.returncode != 0 and "tmux" in result.stderr
