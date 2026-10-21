# G1ZeroShot
Unitree G1 real world zero shot performance using frontier motion tracking models and VLMs.

## SONIC on Mila

The scripts in `scripts/` currently target the Mila layout
`~/Documents/g1_zero/{G1-Zero-Shot,GR00T-WholeBodyControl,deps}`.
Run both simulator and controller on the same allocated GPU node, in separate
interactive terminals:

```bash
bash ~/Documents/g1_zero/G1-Zero-Shot/scripts/run_sonic_sim.sh
```

```bash
bash ~/Documents/g1_zero/G1-Zero-Shot/scripts/run_sonic_controller.sh
```

Wait for `Init Done`; press `]` and Enter in the controller terminal, then `9`
in the simulator terminal to release suspension. Controller keys: `1` selects
slow walking, W starts forward movement, R halts movement, O exits. Press q in
the simulator terminal to exit.

To rebuild, run `bash scripts/build_sonic.sh` from this repository on the GPU
node. `scripts/sonic_env.sh` loads CUDA and configures installed dependencies;
it does not install them. TensorRT currently resides on Mila scratch and is
symlinked from `deps/`; restore it if scratch cleanup removes it.

The C++ dependency library path includes its own OpenSSL. After sourcing
`sonic_env.sh`, use `env -u LD_LIBRARY_PATH git ...` for Git operations so system
SSH loads its matching system OpenSSL. This does not change GitHub keys.

For laptop visualization, forward the allocated node's debug port (replace
`cn-l076` if the allocation changes):

```bash
ssh -N -L 127.0.0.1:5557:cn-l076:5557 mila
```

Then, from the laptop's upstream checkout:

```bash
cd ~/Documents/g1_zero/GR00T-WholeBodyControl/gear_sonic_deploy
../.venv_sim/bin/python visualize_motion.py --realtime_debug_url tcp://127.0.0.1:5557
```

The viewer requires `lxml` in addition to the simulator dependencies. The stream
is a debug visualization; it does not run physics locally.
