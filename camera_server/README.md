# G1 camera quick start

Connect the laptop and G1 to the robot router. **The camera server and viewer do not need internet.** Run laptop commands from the root of `G1-Zero-Shot`.

## First time

1. **Connect to the G1 once** to trust its SSH host key, then exit:

   ```bash
   ssh unitree@192.168.123.164
   exit
   ```

2. **Install the camera server on the G1.** SSH into the G1, clone [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) if needed, and from that repo run:

   ```bash
   bash install_scripts/install_camera_server.sh
   ```

   For our RealSense RGB camera, answer: **service: y**, **retry OAK: n**, **camera: usb**, **device ID: 4**, **wrist cameras: n**, **port: Enter**, **confirm: y**. The video number can change after a reboot; check `/dev/v4l/by-id/` if the camera doesn't work. Re-running the installer recreates `.venv_camera`. Type `exit` to return to the laptop.

   The initial clone and dependency installation normally need internet on the G1; see the optional sharing step below if needed. Once installed, the camera server works without it.

## Before GPT control

After starting the G1, copy the synchronized laptop's time to it:

```bash
python3 camera_server/sync_time.py
```

The G1 needs only its network connection to the laptop, not internet. The laptop must have a synchronized clock. To check without changing the clock, add `--check`. Other laptops need accurate clocks of their own. This step is not needed just to view the camera.

The scripts use `unitree@192.168.123.164` for SSH and sudo. For a different password, set `G1_SSH_PASSWORD`; add `--preserve-env=G1_SSH_PASSWORD` after `sudo` for scripts run with sudo.

The camera server should start automatically. Check it on the G1 with `systemctl status composed_camera_server.service`.

## View the camera on the laptop

With the repositories side by side and `.venv_sim` already installed:

```bash
cd ../GR00T-WholeBodyControl
.venv_sim/bin/python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.123.164 --camera-port 5555
```

Press **Q** to quit or **R** to record. A GPU is not required.

## Optional: give the G1 internet

Use this only when the G1 needs internet, such as for installing software while SSHed in. Connect the laptop to Wi-Fi, then run on the laptop:

```bash
sudo python3 camera_server/share_internet.py start
```

The laptop must stay connected. Other people can SSH into the G1 while it uses the shared connection. Stop sharing with:

```bash
sudo python3 camera_server/share_internet.py stop
```

For automatic clock synchronization when both machines have internet, run `sudo python3 camera_server/sync_time.py --configure-ntp` once. Its default servers include our school and public servers; add `--servers server1 server2` for a different network.
