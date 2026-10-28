#!/usr/bin/env python3
"""Set the laptop and G1 to use the same NTP servers automatically."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
import sys
import tempfile

import time_sync_agent as agent

DEFAULT_SERVERS = ('10.120.31.31', '10.120.184.31', 'ntp.ubuntu.com', 'time.cloudflare.com')


def remote(action, robot, user, servers):
    current_user = os.environ.get('SUDO_USER') or pwd.getpwuid(os.getuid()).pw_name
    known_hosts = Path(pwd.getpwnam(current_user).pw_dir) / '.ssh' / 'known_hosts'
    if not known_hosts.is_file():
        raise RuntimeError('No trusted SSH host key found. SSH into the G1 once first.')
    password = os.environ.get('G1_SSH_PASSWORD', '123')
    if not password or '\n' in password:
        raise RuntimeError('G1_SSH_PASSWORD must be a nonempty single line.')
    source = Path(__file__).with_name('time_sync_agent.py').read_text()
    args = (['sudo', '-S', '-p', ''] if action == 'install' else [])
    args += ['python3', '-c', source, action, *servers]
    with tempfile.TemporaryDirectory(prefix='g1-time-ssh-') as directory:
        askpass = Path(directory) / 'askpass'
        askpass.write_text('#!/bin/sh\nprintf "%s\\n" "$G1_SSH_PASSWORD"\n')
        askpass.chmod(0o700)
        env = os.environ.copy()
        env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE='force',
                   DISPLAY='g1-time:0', G1_SSH_PASSWORD=password)
        result = subprocess.run([
            'ssh', '-F', '/dev/null', '-T', '-o', 'ConnectTimeout=8',
            '-o', 'NumberOfPasswordPrompts=1', '-o', 'StrictHostKeyChecking=yes',
            '-o', f'UserKnownHostsFile={known_hosts}',
            '-o', 'PreferredAuthentications=publickey,password,keyboard-interactive',
            f'{user}@{robot}', ' '.join(shlex.quote(arg) for arg in args),
        ], input=(password + '\n') if action == 'install' else '', text=True,
           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=85,
           start_new_session=True)
    if result.returncode:
        raise RuntimeError('G1: ' + (result.stderr.strip() or 'SSH failed.'))
    return json.loads(result.stdout.strip())


def report(label, result):
    sample = result['ntp']
    state = 'synchronized' if result['synchronized'] else 'not synchronized'
    print(f"{label}: {state}; {sample['server']} is {sample['offset_ms']:+.1f} ms "
          f"from its clock (network round trip {sample['round_trip_ms']:.1f} ms)", flush=True)


def difference(robot, laptop):
    a, b = robot['ntp'], laptop['ntp']
    if a['server'] != b['server']:
        print('The computers reached different time servers; their mutual clock difference was not verified.', flush=True)
        return None
    value = a['offset_ms'] - b['offset_ms']
    uncertainty = (a['round_trip_ms'] + b['round_trip_ms']) / 2
    print(f'Estimated laptop minus G1 clock: {value:+.1f} ms '
          f'(network uncertainty at least ±{uncertainty:.1f} ms)', flush=True)
    return abs(value) + uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check both clocks without changing them')
    parser.add_argument('--robot', default='192.168.123.164', help='G1 IPv4 address')
    parser.add_argument('--ssh-user', default='unitree', help='G1 SSH username')
    parser.add_argument('--servers', nargs='+', default=list(DEFAULT_SERVERS), metavar='NTP_SERVER',
                        help='NTP servers in priority order; defaults include the school and public servers')
    args = parser.parse_args()
    ipaddress.IPv4Address(args.robot)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*\$?', args.ssh_user):
        raise RuntimeError('Invalid SSH username.')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*', server) for server in args.servers):
        raise RuntimeError('Invalid NTP server name.')
    if not args.check and os.geteuid() != 0:
        raise RuntimeError('Run setup with sudo; --check does not require sudo.')
    print('Checking NTP access on both computers...', flush=True)
    robot_before = remote('check', args.robot, args.ssh_user, args.servers)
    laptop_before = agent.check(args.servers)
    report('G1', robot_before)
    report('Laptop', laptop_before)
    if args.check:
        difference(robot_before, laptop_before)
        return
    print('Configuring the G1...', flush=True)
    robot_after = remote('install', args.robot, args.ssh_user, args.servers)
    report('G1', robot_after)
    print('Configuring the laptop...', flush=True)
    try:
        laptop_after = agent.install(args.servers)
    except Exception:
        print('The G1 was configured, but laptop setup failed; run --check and retry.', file=sys.stderr)
        raise
    report('Laptop', laptop_after)
    # Compare samples from the same reference after both services converge.
    robot_now = remote('check', args.robot, args.ssh_user, args.servers)
    laptop_now = agent.check(args.servers)
    if not robot_now['synchronized'] or not laptop_now['synchronized']:
        raise RuntimeError('Both were configured, but one time service is no longer synchronized.')
    margin = difference(robot_now, laptop_now)
    if margin is None:
        raise RuntimeError('Both were configured, but their clock difference could not be verified.')
    if margin >= 500:
        raise RuntimeError('Clock difference is too close to the controller’s 500 ms limit; '
                           'wait and run --check before using timestamp-sensitive control.')
    print('Both computers are configured for automatic time synchronization.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
