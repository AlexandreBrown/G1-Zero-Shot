#!/usr/bin/env bash
# Read-only probe of the G1's PC2: can it run pc2_camera_server.py with what is already installed?
# Usage (laptop, robot on Ethernet):  ssh unitree@192.168.123.164 'bash -s' < probe_pc2.sh
set -u
echo "== host"; hostname; uname -m; head -1 /etc/nv_tegra_release 2>/dev/null; lsb_release -ds 2>/dev/null
echo "== disk"; df -h / | tail -1
echo "== python"; command -v python3 && python3 --version
for mod in pyrealsense2 cv2 numpy zmq msgpack; do
  python3 -c "import $mod; print('$mod', getattr($mod, '__version__', 'ok'))" 2>/dev/null || echo "$mod MISSING"
done
echo "== realsense devices"; command -v rs-enumerate-devices >/dev/null && rs-enumerate-devices -s 2>&1 | head -5 || echo "rs-enumerate-devices not installed"
lsusb 2>/dev/null | grep -i -E "intel|realsense" || echo "no Intel USB device listed"
echo "== port 5555 in use?"; (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E ":5555\b" || echo "5555 free"
echo "== processes holding the camera"; pgrep -af -i "realsense|camera" | head -5 || true
