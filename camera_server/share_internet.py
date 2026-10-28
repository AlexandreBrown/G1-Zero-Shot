#!/usr/bin/env python3
"""Share the laptop's Wi-Fi with devices on its existing robot LAN."""

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
import time

try:
    from .g1_network import refresh_time_sync
except ImportError:
    from g1_network import refresh_time_sync

STATE = Path('/run/g1-camera-internet-share.json')
CHAIN = 'G1_CAMERA_SHARE'


def run(*args, check=True, capture=False):
    return subprocess.run(args, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def ipt(*args, check=True):
    return run('iptables', '-w', *args, check=check, capture=True)


def route(address):
    return json.loads(run('ip', '-j', '-4', 'route', 'get', address, capture=True).stdout)[0]


def configuration(robot):
    lan_route, wan_route = route(robot), route('1.1.1.1')
    lan, wan = lan_route['dev'], wan_route['dev']
    if lan == wan:
        raise RuntimeError('The robot and internet use the same interface; connect both Ethernet and Wi-Fi first.')
    addresses = json.loads(run('ip', '-j', '-4', 'address', 'show', 'dev', lan, capture=True).stdout)
    robot_ip = ipaddress.ip_address(robot)
    matches = []
    for addr in addresses[0]['addr_info']:
        if addr.get('family') == 'inet':
            interface = ipaddress.ip_interface(f"{addr['local']}/{addr['prefixlen']}")
            if robot_ip in interface.network:
                matches.append(interface)
    if len(matches) != 1:
        raise RuntimeError('Could not identify one Ethernet subnet containing the robot.')
    wifi_path = Path('/sys/class/net') / wan / 'wireless'
    if not wifi_path.exists():
        raise RuntimeError(f'The internet route uses {wan}, which is not a Wi-Fi interface.')
    interface = matches[0]
    result = run('nmcli', '-g', 'IP4.DNS', 'device', 'show', wan, check=False, capture=True)
    dns = []
    for value in re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', result.stdout or ''):
        try:
            addr = ipaddress.IPv4Address(value)
            if not addr.is_loopback and not addr.is_unspecified:
                dns.append(str(addr))
        except ipaddress.AddressValueError:
            pass
    return dict(lan=lan, wan=wan, gateway=str(interface.ip), subnet=str(interface.network), dns=dns)


def remote(action, cfg):
    """Use the existing trusted SSH host key and authenticate without a prompt."""
    user = os.environ.get('SUDO_USER') or pwd.getpwuid(os.getuid()).pw_name
    home = Path(pwd.getpwnam(user).pw_dir)
    known_hosts = home / '.ssh' / 'known_hosts'
    if not known_hosts.is_file():
        raise RuntimeError('SSH host key is not saved; connect to the G1 with ssh once first.')
    password = os.environ.get('G1_SSH_PASSWORD', '123')
    if '\n' in password or not password:
        raise RuntimeError('G1_SSH_PASSWORD must be a nonempty single line.')
    program = Path(__file__).with_name('g1_network.py').read_text()
    args = ['sudo', '-S', '-p', '', 'python3', '-c', program, action]
    if action == 'start':
        args += [cfg['gateway'], str(int(time.time())), *cfg['dns']]
    remote_command = ' '.join(shlex.quote(arg) for arg in args)
    with tempfile.TemporaryDirectory(prefix='g1-ssh-') as directory:
        askpass = Path(directory) / 'askpass'
        askpass.write_text('#!/bin/sh\nprintf "%s\\n" "$G1_SSH_PASSWORD"\n')
        askpass.chmod(0o700)
        env = os.environ.copy()
        env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE='force',
                   DISPLAY='g1-share:0', G1_SSH_PASSWORD=password)
        result = subprocess.run([
            'ssh', '-F', '/dev/null', '-T', '-o', 'ConnectTimeout=8',
            '-o', 'NumberOfPasswordPrompts=1', '-o', 'StrictHostKeyChecking=yes',
            '-o', f'UserKnownHostsFile={known_hosts}',
            '-o', 'PreferredAuthentications=publickey,password,keyboard-interactive',
            f"{cfg.get('ssh_user', 'unitree')}@{cfg['robot']}", remote_command,
        ], input=password + '\n', text=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE, env=env, timeout=50, start_new_session=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'SSH or G1 setup failed.')
    print(result.stdout.strip())


def active_matches(cfg):
    saved = json.loads(STATE.read_text())
    for key in ('lan', 'wan', 'gateway', 'subnet', 'dns'):
        if saved.get(key) != cfg.get(key):
            raise RuntimeError('Laptop network settings changed; stop sharing before starting again.')
    for table, base in [('filter', 'FORWARD'), ('nat', 'POSTROUTING')]:
        if ipt('-t', table, '-C', base, '-j', CHAIN, check=False).returncode:
            raise RuntimeError('Saved sharing state exists, but its firewall rules are missing.')
    if run('sysctl', '-n', 'net.ipv4.ip_forward', capture=True).stdout.strip() != '1':
        raise RuntimeError('IPv4 forwarding is disabled despite the saved sharing state.')
    saved.update(robot=cfg['robot'], ssh_user=cfg['ssh_user'])
    return saved


def cleanup(cfg):
    # Only remove the rules and chains created by this script.
    for table, base, chain in [('filter', 'FORWARD', CHAIN), ('nat', 'POSTROUTING', CHAIN)]:
        if ipt('-t', table, '-C', base, '-j', chain, check=False).returncode == 0:
            ipt('-t', table, '-D', base, '-j', chain)
        if ipt('-t', table, '-n', '-L', chain, check=False).returncode == 0:
            ipt('-t', table, '-F', chain)
            ipt('-t', table, '-X', chain)
    if cfg.get('enabled_forwarding'):
        run('sysctl', '-w', 'net.ipv4.ip_forward=0', capture=True)


def start(cfg):
    if STATE.exists():
        raise RuntimeError('Sharing is already active; stop it first.')
    for table in ('filter', 'nat'):
        if ipt('-t', table, '-n', '-L', CHAIN, check=False).returncode == 0:
            raise RuntimeError(f'Firewall chain {CHAIN} already exists in {table}; refusing to overwrite it.')
    old_forward = run('sysctl', '-n', 'net.ipv4.ip_forward', capture=True).stdout.strip()
    if old_forward not in ('0', '1'):
        raise RuntimeError('Could not determine the IPv4 forwarding state.')
    cfg['enabled_forwarding'] = old_forward == '0'
    try:
        ipt('-t', 'filter', '-N', CHAIN)
        ipt('-t', 'filter', '-A', CHAIN, '-i', cfg['lan'], '-o', cfg['wan'],
            '-s', cfg['subnet'], '-j', 'ACCEPT')
        ipt('-t', 'filter', '-A', CHAIN, '-i', cfg['wan'], '-o', cfg['lan'],
            '-d', cfg['subnet'], '-m', 'conntrack', '--ctstate', 'ESTABLISHED,RELATED', '-j', 'ACCEPT')
        ipt('-t', 'nat', '-N', CHAIN)
        ipt('-t', 'nat', '-A', CHAIN, '-s', cfg['subnet'], '-o', cfg['wan'], '-j', 'MASQUERADE')
        ipt('-t', 'filter', '-I', 'FORWARD', '1', '-j', CHAIN)
        ipt('-t', 'nat', '-I', 'POSTROUTING', '1', '-j', CHAIN)
        if cfg['enabled_forwarding']:
            run('sysctl', '-w', 'net.ipv4.ip_forward=1', capture=True)
        STATE.write_text(json.dumps(cfg))
    except Exception:
        cleanup(cfg)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'stop', 'check'))
    parser.add_argument('--robot', default='192.168.123.164', help='G1 IPv4 address (default: %(default)s)')
    parser.add_argument('--ssh-user', default='unitree', help='G1 SSH username')
    args = parser.parse_args()
    ipaddress.IPv4Address(args.robot)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*\$?', args.ssh_user):
        raise RuntimeError('Invalid SSH username.')
    if args.action == 'check':
        cfg = configuration(args.robot)
        print(f"Ethernet: {cfg['lan']} ({cfg['gateway']}); Wi-Fi: {cfg['wan']}")
        print('Wi-Fi DNS: ' + ', '.join(cfg['dns']))
        return
    if os.geteuid() != 0:
        raise RuntimeError('Run start and stop with sudo.')
    if args.action == 'start':
        cfg = configuration(args.robot)
        cfg.update(robot=args.robot, ssh_user=args.ssh_user)
        if not cfg['dns']:
            raise RuntimeError('No usable Wi-Fi DNS server was detected.')
        refresh_time_sync('Laptop')
        newly_started = not STATE.exists()
        if newly_started:
            start(cfg)
        else:
            cfg = active_matches(cfg)
        try:
            remote('start', cfg)
            STATE.write_text(json.dumps(cfg))
        except Exception:
            if newly_started:
                cleanup(cfg)
                STATE.unlink(missing_ok=True)
            raise
        print('Internet sharing is active for the G1. Leave Wi-Fi and Ethernet connected.')
    else:
        if not STATE.exists():
            raise RuntimeError('No active sharing session was found.')
        cfg = json.loads(STATE.read_text())
        cfg.setdefault('robot', args.robot)
        cfg.setdefault('ssh_user', args.ssh_user)
        remote('stop', cfg)
        cleanup(cfg)
        STATE.unlink()
        print('Laptop sharing stopped.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError, KeyError, IndexError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        print(f'Error: {detail}', file=sys.stderr)
        sys.exit(1)
