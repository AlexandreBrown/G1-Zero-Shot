# SONIC on the G1 EDU U4 Orin

## Current stage: read-only hardware inventory

Do not run the cluster launchers on the robot. They select loopback and disable
CRC checks for MuJoCo. The cluster x86_64 executable, desktop TensorRT 10.13
libraries, and L40S engines are not the Orin deployment.

Upstream requires JetPack 6 (L4T R36) and TensorRT 10.7 on Orin:
https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html
Inspect installed versions before installing or upgrading anything. This kit
does not flash the robot, change packages, stop services, or start a controller.

## Transfer after switching to the robot network

LOCAL:

```bash
scp ~/Documents/g1_zero/G1-Zero-Shot/scripts/robot/probe_orin.sh unitree@192.168.123.164:/tmp/sonic_probe_orin.sh
ssh unitree@192.168.123.164
```

ROBOT (Orin):

```bash
bash /tmp/sonic_probe_orin.sh 2>&1 | tee ~/sonic-orin-preflight.txt
```

Return the report for dependency selection. No motor commands are sent. The
report does not establish hardware joint compatibility: confirm the 29-body-
joint configuration and installed hand type separately before activation.

## Offline source and models

LOCAL, before switching networks:

```bash
bash ~/Documents/g1_zero/G1-Zero-Shot/scripts/robot/prepare_bundle.sh
```

The archive is created under `artifacts/orin/` with a SHA-256 checksum and
upstream commit ID. It includes source, example references, ONNX weights, and
licenses, but is not a complete offline installer: ARM64 dependencies depend on
the preflight report. Never reuse the desktop dependency environment on Orin.

LOCAL, on the robot network, optional transfer of prepared files:

```bash
ssh unitree@192.168.123.164 'mkdir -p ~/g1_zero/incoming'
scp ~/Documents/g1_zero/G1-Zero-Shot/artifacts/orin/* unitree@192.168.123.164:g1_zero/incoming/
```

ROBOT (Orin), verify archive integrity without executing it:

```bash
cd ~/g1_zero/incoming
sha256sum -c SHA256SUMS
```

## Hardware activation is a separate stage

Reading the upstream code revealed actions before the `]` key: the constructor
calls Unitree MotionSwitcher ReleaseMode, and INIT sends a ramp toward default
standing angles. INIT also closes/opens Dex3 hands. Thus launching the upstream
executable is itself hardware activation; `Init Done` is not a no-motion gate.

Before a hardware launch, prepare and validate a standing-only input mode in
MuJoCo, block reference-motion playback and locomotion commands, verify the
29-joint mapping/hand configuration, explicitly choose the robot-facing DDS
interface, and retain CRC validation. Standing-only means no requested walking;
the balance policy can still step or move limbs. A gantry does not prevent joint
motion. Stop the MuJoCo processes before connecting a controller to hardware.

For the initial trial, use rated support, appropriate foot contact, clear space,
and a person at the robot's hardware stop. Keyboard O exits the policy; it is
not a guaranteed stand-down or an independent hardware emergency stop. Do not
leave policy startup/stop behavior to an SSH connection alone.
