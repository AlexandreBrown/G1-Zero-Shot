#!/usr/bin/env python3
"""Configure and check one Linux system's time service."""

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

CONFIG = Path('/etc/systemd/timesyncd.conf.d/80-g1-camera-time.conf')
MARKER = '# Managed by G1-Zero-Shot camera_server/sync_time.py\n'


def command(*args, check=True):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check)


def ntp_sample(servers):
    errors = []
    for host in servers:
        try:
            addresses = socket.getaddrinfo(host, 123, socket.AF_UNSPEC, socket.SOCK_DGRAM)
            for family, kind, proto, _, address in addresses:
                try:
                    with socket.socket(family, kind, proto) as sock:
                        sock.settimeout(2)
                        sock.connect(address)
                        request = bytearray(48)
                        request[0] = 0x23  # NTP v4 client
                        start = time.time()
                        seconds = int(start)
                        struct.pack_into('!II', request, 40, seconds + 2208988800,
                                         int((start - seconds) * 2**32))
                        sock.send(request)
                        reply = sock.recv(512)
                        end = time.time()
                    if len(reply) < 48 or reply[0] & 7 != 4 or reply[0] >> 6 == 3 or not 1 <= reply[1] <= 15:
                        raise ValueError('invalid or unsynchronized NTP response')
                    values = struct.unpack('!12I', reply[:48])
                    if reply[24:32] != request[40:48]:
                        raise ValueError('NTP response does not match request')
                    received = values[8] - 2208988800 + values[9] / 2**32
                    sent = values[10] - 2208988800 + values[11] / 2**32
                    offset = ((received - start) + (sent - end)) / 2
                    delay = end - start - (sent - received)
                    return {'server': host, 'offset_ms': round(offset * 1000, 1),
                            'round_trip_ms': round(max(0, delay) * 1000, 1)}
                except (OSError, ValueError) as exc:
                    errors.append(f'{host}: {exc}')
        except OSError as exc:
            errors.append(f'{host}: {exc}')
    raise RuntimeError('No time server responded. Check internet sharing and UDP 123. ' + '; '.join(errors))


def service_running():
    result = command('systemctl', 'is-active', 'systemd-timesyncd.service', check=False)
    return result.returncode == 0


def check(servers):
    if not service_running():
        raise RuntimeError('systemd-timesyncd is not running; this script supports systems using it.')
    status = command('timedatectl', 'show', '-p', 'NTPSynchronized', '--value').stdout.strip()
    return {'ntp': ntp_sample(servers), 'synchronized': status == 'yes'}


def install(servers):
    if os.geteuid() != 0:
        raise RuntimeError('Configuration requires sudo.')
    check(servers)  # Do not alter settings if the NTP service or network is unavailable.
    old = CONFIG.read_text() if CONFIG.exists() else None
    if old is not None and not old.startswith(MARKER):
        raise RuntimeError(f'{CONFIG} already exists and is not managed by this script.')
    data = MARKER + '[Time]\nNTP=' + ' '.join(servers) + '\n'
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=CONFIG.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
        temporary.chmod(0o644)
        temporary.replace(CONFIG)
        command('systemctl', 'restart', 'systemd-timesyncd.service')
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if old is None:
            CONFIG.unlink(missing_ok=True)
        else:
            CONFIG.write_text(old)
        command('systemctl', 'restart', 'systemd-timesyncd.service', check=False)
        raise

    deadline = time.monotonic() + 45
    result = None
    while time.monotonic() < deadline:
        result = check(servers)
        if result['synchronized'] and abs(result['ntp']['offset_ms']) < 200:
            return result
        time.sleep(2)
    raise RuntimeError(f'Time configuration saved, but synchronization has not converged: {result}. '
                       'Run --check later; do not use timestamp-sensitive control yet.')


def main():
    action, *servers = sys.argv[1:]
    if not servers:
        raise RuntimeError('No NTP servers provided.')
    result = check(servers) if action == 'check' else install(servers) if action == 'install' else None
    if result is None:
        raise RuntimeError('Invalid action.')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            message = exc.stderr.strip() or str(exc)
        else:
            message = str(exc)
        print(f'Time sync error: {message}', file=sys.stderr)
        sys.exit(1)
