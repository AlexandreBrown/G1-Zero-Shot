#!/usr/bin/env python3
"""Set the G1 clock from this laptop without requiring internet on the G1."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
import tempfile
import time

import time_sync_agent

PUBLIC_TIME_SERVERS = ('ntp.ubuntu.com', 'time.cloudflare.com')


def laptop_synchronized():
    status = subprocess.run(['timedatectl', 'show', '-p', 'NTPSynchronized', '--value'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return status.stdout.strip() == 'yes'


def discover_time_servers():
    # Some networks block public NTP but provide it on their DNS servers.
    servers = []
    try:
        result = subprocess.run(['resolvectl', 'dns'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=False, timeout=5)
        for word in result.stdout.split():
            try:
                address = str(ipaddress.ip_address(word))
            except ValueError:
                continue
            if address not in servers:
                servers.append(address)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return servers[:8] + list(PUBLIC_TIME_SERVERS)


def ensure_laptop_time(servers=None):
    if laptop_synchronized():
        return
    if not time_sync_agent.service_running():
        raise RuntimeError('Laptop clock is not synchronized and systemd-timesyncd is not running. '
                           'Enable a time service and connect to a network with NTP, then retry.')
    candidates = servers or discover_time_servers()
    print('Laptop clock is not synchronized. Looking for a reachable time server...', flush=True)
    try:
        sample = time_sync_agent.ntp_sample(candidates)
    except RuntimeError as exc:
        raise RuntimeError('No laptop time server responded. Connect the laptop to a network '
                           'that allows NTP (UDP 123), then retry.') from exc
    selected = sample['server']
    configured = [selected] + [server for server in PUBLIC_TIME_SERVERS if server != selected]
    print(f'Found {selected}. Configuring laptop time; sudo may ask for your password...', flush=True)
    subprocess.run(['sudo', sys.executable, str(Path(__file__).with_name('time_sync_agent.py')),
                    'install', *configured], check=True)
    if not laptop_synchronized():
        raise RuntimeError('Laptop time service did not synchronize. Retry before setting the G1 clock.')
    print('Laptop time synchronized.', flush=True)


class RobotClock:
    def __init__(self, host, user, change):
        self.host, self.user, self.change = host, user, change
        self.buffer = b''
        self.temp = None
        self.process = None
        self.errors = None

    def __enter__(self):
        known_hosts = Path.home() / '.ssh' / 'known_hosts'
        if not known_hosts.is_file():
            raise RuntimeError('SSH host key not found; connect to the G1 with ssh once first.')
        password = os.environ.get('G1_SSH_PASSWORD', '123')
        if not password or '\n' in password:
            raise RuntimeError('G1_SSH_PASSWORD must be a nonempty single line.')
        source = Path(__file__).with_name('robot_clock_agent.py').read_text()
        args = (['sudo', '-S', '-p', ''] if self.change else [])
        args += ['python3', '-u', '-c', source, 'set' if self.change else 'check']
        self.temp = tempfile.TemporaryDirectory(prefix='g1-clock-ssh-')
        askpass = Path(self.temp.name) / 'askpass'
        askpass.write_text('#!/bin/sh\nprintf "%s\\n" "$G1_SSH_PASSWORD"\n')
        askpass.chmod(0o700)
        env = os.environ.copy()
        env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE='force',
                   DISPLAY='g1-clock:0', G1_SSH_PASSWORD=password)
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen([
                'ssh', '-F', '/dev/null', '-T', '-o', 'ConnectTimeout=8',
                '-o', 'NumberOfPasswordPrompts=1', '-o', 'StrictHostKeyChecking=yes',
                '-o', f'UserKnownHostsFile={known_hosts}',
                '-o', 'PreferredAuthentications=publickey,password,keyboard-interactive',
                f'{self.user}@{self.host}', ' '.join(shlex.quote(arg) for arg in args),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
               env=env, start_new_session=True, bufsize=0)
            if self.change:
                self.process.stdin.write((password + '\n').encode())
            if self.read().get('ready') is not True:
                raise RuntimeError('Unexpected G1 clock response.')
            return self
        except Exception:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + 15
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], max(0, remaining))[0]:
                raise RuntimeError('Timed out waiting for the G1 clock over SSH.')
            data = os.read(self.process.stdout.fileno(), 4096)
            if not data:
                self.errors.seek(0)
                detail = self.errors.read(4096).decode(errors='replace').strip()
                raise RuntimeError(f'G1 clock connection failed: {detail or "SSH exited"}')
            self.buffer += data
            if len(self.buffer) > 65536:
                raise RuntimeError('Unexpected G1 clock output.')
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def send(self, payload):
        self.process.stdin.write((json.dumps(payload) + '\n').encode())
        return self.read()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        if self.errors is not None:
            self.errors.close()
        if self.temp is not None:
            self.temp.cleanup()

    def __exit__(self, *args):
        self.close()


def sample(clock, count=6):
    samples = []
    for _ in range(count):
        mono0, wall0 = time.monotonic_ns(), time.time_ns()
        response = clock.send({'action': 'probe'})
        wall1, mono1 = time.time_ns(), time.monotonic_ns()
        rtt = mono1 - mono0
        if abs((wall1 - wall0) - rtt) > 50_000_000:
            raise RuntimeError('Laptop clock changed during measurement; try again.')
        value = response.get('unix_ns')
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError('Invalid G1 clock response.')
        samples.append((rtt, value - (wall0 + wall1) // 2))
    return min(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Measure difference without changing the G1 clock')
    parser.add_argument('--robot', default='192.168.123.164', help='G1 IPv4 address')
    parser.add_argument('--ssh-user', default='unitree', help='G1 SSH username')
    parser.add_argument('--configure-ntp', action='store_true',
                        help='Configure automatic NTP on both computers (requires sudo and G1 internet)')
    parser.add_argument('--servers', nargs='+', metavar='NTP_SERVER',
                        help='Time servers to try when configuring NTP')
    args = parser.parse_args(argv)
    ipaddress.IPv4Address(args.robot)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*\$?', args.ssh_user):
        raise RuntimeError('Invalid SSH username.')
    if args.configure_ntp:
        import _configure_ntp
        forwarded = ['--robot', args.robot, '--ssh-user', args.ssh_user]
        if args.check:
            forwarded.append('--check')
        if args.servers:
            forwarded.extend(['--servers', *args.servers])
        return _configure_ntp.main(forwarded)
    if args.servers and any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*', server) for server in args.servers):
        raise RuntimeError('Invalid NTP server name.')
    if os.geteuid() == 0:
        raise RuntimeError('Run sync_time.py without sudo; sudo is only needed with --configure-ntp.')
    if not args.check:
        ensure_laptop_time(args.servers)
    with RobotClock(args.robot, args.ssh_user, not args.check) as clock:
        rtt, offset = sample(clock)
        print(f'Before: G1 minus laptop {offset / 1e6:+.1f} ms '
              f'(round trip {rtt / 1e6:.1f} ms)', flush=True)
        if not args.check:
            if clock.send({'action': 'set', 'unix_ns': time.time_ns() + rtt // 2}).get('set') is not True:
                raise RuntimeError('G1 did not confirm its clock change.')
            rtt, offset = sample(clock)
            print(f'After: G1 minus laptop {offset / 1e6:+.1f} ms '
                  f'(round trip {rtt / 1e6:.1f} ms)', flush=True)
            if abs(offset) + rtt // 2 >= 500_000_000:
                raise RuntimeError('Could not verify clocks within 500 ms; check the connection.')
            print('G1 clock set from the laptop.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        print(f'Error: {detail}', file=sys.stderr)
        sys.exit(1)
