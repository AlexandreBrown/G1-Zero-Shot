import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sync_time
import _configure_ntp


class SetG1ClockTests(unittest.TestCase):
    def agent(self, commands):
        return subprocess.run([sys.executable, str(ROOT / 'robot_clock_agent.py'), 'check'],
                              input=commands, text=True, capture_output=True, timeout=5)

    def test_cached_sudo_password_does_not_break_clock_probe(self):
        result = self.agent('123\n{"action":"probe"}\n{"action":"quit"}\n')
        self.assertEqual(result.returncode, 0)
        ready, probe = (json.loads(line) for line in result.stdout.splitlines())
        self.assertEqual(ready, {'ready': True})
        self.assertIsInstance(probe['unix_ns'], int)

    def test_readonly_agent_rejects_clock_setting(self):
        result = self.agent('{"action":"set","unix_ns":1800000000000000000}\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unauthorized', result.stderr)

    def test_no_reachable_time_server_never_connects_to_set_clock(self):
        with mock.patch.object(sys, 'argv', ['sync_time.py']), \
             mock.patch.object(sync_time.os, 'geteuid', return_value=1000), \
             mock.patch.object(sync_time, 'laptop_synchronized', return_value=False), \
             mock.patch.object(sync_time.time_sync_agent, 'service_running', return_value=True), \
             mock.patch.object(sync_time.time_sync_agent, 'ntp_sample', side_effect=RuntimeError('offline')), \
             mock.patch.object(sync_time, 'RobotClock') as remote:
            with self.assertRaisesRegex(RuntimeError, 'No laptop time server'):
                sync_time.main()
        remote.assert_not_called()

    def test_unsynchronized_laptop_configures_reachable_server(self):
        with mock.patch.object(sync_time, 'laptop_synchronized', side_effect=[False, True]), \
             mock.patch.object(sync_time.time_sync_agent, 'service_running', return_value=True), \
             mock.patch.object(sync_time, 'discover_time_servers', return_value=['10.1.2.3', 'ntp.ubuntu.com']), \
             mock.patch.object(sync_time.time_sync_agent, 'ntp_sample', return_value={'server': '10.1.2.3'}) as probe, \
             mock.patch.object(sync_time.subprocess, 'run') as command:
            sync_time.ensure_laptop_time()
        probe.assert_called_once_with(['10.1.2.3', 'ntp.ubuntu.com'])
        self.assertEqual(command.call_args.args[0][0], 'sudo')
        self.assertEqual(command.call_args.args[0][-4:],
                         ['install', '10.1.2.3', 'ntp.ubuntu.com', 'time.cloudflare.com'])

    def test_recovery_failure_never_connects_to_robot(self):
        with mock.patch.object(sync_time.os, 'geteuid', return_value=1000), \
             mock.patch.object(sync_time, 'laptop_synchronized', return_value=False), \
             mock.patch.object(sync_time.time_sync_agent, 'service_running', return_value=True), \
             mock.patch.object(sync_time.time_sync_agent, 'ntp_sample', return_value={'server': 'ntp.ubuntu.com'}), \
             mock.patch.object(sync_time.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'sudo')), \
             mock.patch.object(sync_time, 'RobotClock') as remote:
            with self.assertRaises(subprocess.CalledProcessError):
                sync_time.main([])
        remote.assert_not_called()

    def test_default_sets_robot_clock_and_checks_result(self):
        with mock.patch.object(sync_time.os, 'geteuid', return_value=1000), \
             mock.patch.object(sync_time.subprocess, 'run',
                               return_value=subprocess.CompletedProcess([], 0, stdout='yes\n', stderr='')), \
             mock.patch.object(sync_time, 'RobotClock') as connection, \
             mock.patch.object(sync_time, 'sample', side_effect=[(4_000_000, 1_000_000_000),
                                                                (4_000_000, 1_000_000)]):
            clock = connection.return_value.__enter__.return_value
            clock.send.return_value = {'set': True}
            sync_time.main([])
        connection.assert_called_once_with('192.168.123.164', 'unitree', True)
        self.assertEqual(clock.send.call_args.args[0]['action'], 'set')

    def test_ntp_option_forwards_arguments(self):
        with mock.patch.object(_configure_ntp, 'main') as configure, \
             mock.patch.object(sync_time, 'RobotClock') as remote:
            sync_time.main(['--configure-ntp', '--check', '--robot', '192.168.123.165',
                            '--servers', 'ntp.example.org'])
        configure.assert_called_once_with(['--robot', '192.168.123.165', '--ssh-user', 'unitree',
                                           '--check', '--servers', 'ntp.example.org'])
        remote.assert_not_called()

    def test_sudo_rejected_for_one_shot(self):
        with mock.patch.object(sync_time.os, 'geteuid', return_value=0), \
             mock.patch.object(sync_time, 'RobotClock') as remote:
            with self.assertRaisesRegex(RuntimeError, 'without sudo'):
                sync_time.main([])
        remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
