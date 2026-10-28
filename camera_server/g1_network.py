#!/usr/bin/env python3
"""Remote half of share_internet.py. Invoked as root on the G1 over SSH."""

import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

STATE = Path('/run/g1-camera-internet-client.json')
TIME_CONFIG = Path('/etc/systemd/timesyncd.conf.d/80-g1-camera-time.conf')


def run(*args, check=True):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check).stdout.strip()


def resolv_values(kind, dev):
    output = run('resolvectl', kind, dev)
    return output.split(':', 1)[1].split()


def set_resolv(kind, dev, values):
    run('resolvectl', kind, dev, *(values or ['']))


def refresh_time_sync(label, wait=15):
    """Wake the configured time service after the network becomes usable."""
    if not TIME_CONFIG.exists():
        return None
    try:
        if run('systemctl', 'is-active', 'systemd-timesyncd.service', check=False) != 'active':
            print(f'{label}: time service is not active; run sync_time.py --configure-ntp.', flush=True)
            return False
        run('systemctl', 'restart', 'systemd-timesyncd.service')
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if run('timedatectl', 'show', '-p', 'NTPSynchronized', '--value') == 'yes':
                print(f'{label}: clock synchronized.', flush=True)
                return True
            time.sleep(1)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'{label}: could not refresh time synchronization: {exc}', flush=True)
        return False
    print(f'{label}: internet is available, but time synchronization is still pending. '
          'Check with sync_time.py --check before using timestamp-sensitive control.', flush=True)
    return False


def restore(state):
    dev, gateway, metric = state['dev'], state['gateway'], str(state['metric'])
    run('ip', '-4', 'route', 'del', 'default', 'via', gateway, 'dev', dev,
        'metric', metric, check=False)
    set_resolv('dns', dev, state['old_dns'])
    set_resolv('domain', dev, state['old_domains'])
    STATE.unlink(missing_ok=True)


def start(gateway, laptop_time, dns):
    ipaddress.IPv4Address(gateway)
    for server in dns:
        ipaddress.IPv4Address(server)
    if not dns:
        raise RuntimeError('No Wi-Fi DNS server was detected on the laptop.')
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state['gateway'] != gateway or state['dns'] != dns:
            restore(state)
            state = None
    else:
        state = None
    if state is None:
        route = json.loads(run('ip', '-j', '-4', 'route', 'get', gateway))[0]
        dev = route['dev']
        routes = json.loads(run('ip', '-j', '-4', 'route', 'show', 'default'))
        metrics = {r.get('metric', 0) for r in routes}
        metric = next((n for n in range(1, 100) if n not in metrics), None)
        if metric is None:
            raise RuntimeError('No available route metric for sharing.')
        state = dict(dev=dev, gateway=gateway, dns=dns, metric=metric,
                     old_dns=resolv_values('dns', dev),
                     old_domains=resolv_values('domain', dev))
        fd = os.open(str(STATE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f)
    try:
        run('ip', '-4', 'route', 'replace', 'default', 'via', gateway,
            'dev', state['dev'], 'metric', str(state['metric']))
        set_resolv('dns', state['dev'], dns)
        set_resolv('domain', state['dev'], ['~.'])
        if abs(time.time() - laptop_time) > 300:
            run('date', '-u', '-s', f'@{int(laptop_time)}')
            print('Corrected the G1 clock using the laptop clock.')
        # This checks routing, DNS and TLS, without depending on ping being allowed.
        request = urllib.request.Request('https://github.com/', method='HEAD')
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status >= 400:
                raise RuntimeError(f'GitHub returned HTTP {response.status}.')
    except Exception:
        restore(state)
        raise
    print('G1 internet verified: https://github.com')
    refresh_time_sync('G1')


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'start':
        start(sys.argv[2], float(sys.argv[3]), sys.argv[4:])
    elif len(sys.argv) == 2 and sys.argv[1] == 'stop':
        if STATE.exists():
            restore(json.loads(STATE.read_text()))
        print('G1 network settings restored.')
    else:
        raise RuntimeError('Invalid remote network command.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            message = exc.stderr.strip() or str(exc)
        else:
            message = str(exc)
        print(f'G1 network setup failed: {message}', file=sys.stderr)
        sys.exit(1)
