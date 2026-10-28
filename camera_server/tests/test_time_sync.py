import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _configure_ntp as sync_time
import time_sync_agent as agent


class TimeSyncTests(unittest.TestCase):
    def test_install_refuses_to_replace_an_unmanaged_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'time.conf'
            config.write_text('existing local settings\n')
            with mock.patch.object(agent, 'CONFIG', config), mock.patch.object(agent.os, 'geteuid', return_value=0), \
                 mock.patch.object(agent, 'check', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'not managed'):
                    agent.install(['ntp.example.org'])
            self.assertEqual(config.read_text(), 'existing local settings\n')

    def test_restart_failure_restores_previous_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'time.conf'
            old = agent.MARKER + '[Time]\nNTP=old.example.org\n'
            config.write_text(old)
            def run(*args, **kwargs):
                if kwargs.get('check', True):
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 0)
            with mock.patch.object(agent, 'CONFIG', config), mock.patch.object(agent.os, 'geteuid', return_value=0), \
                 mock.patch.object(agent, 'check', return_value={}), mock.patch.object(agent, 'command', side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    agent.install(['ntp.example.org'])
            self.assertEqual(config.read_text(), old)

    def test_clock_difference_requires_same_reference(self):
        robot = {'ntp': {'server': 'one', 'offset_ms': 1200, 'round_trip_ms': 4}}
        laptop = {'ntp': {'server': 'one', 'offset_ms': 200, 'round_trip_ms': 2}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sync_time.difference(robot, laptop), 1003)
            laptop['ntp']['server'] = 'other'
            self.assertIsNone(sync_time.difference(robot, laptop))

    def test_setup_checks_both_machines_and_configures_them(self):
        before = {'ntp': {'server': 'one', 'offset_ms': 1000, 'round_trip_ms': 3}, 'synchronized': False}
        after = {'ntp': {'server': 'one', 'offset_ms': 10, 'round_trip_ms': 3}, 'synchronized': True}
        with mock.patch.object(sys, 'argv', ['sync_time.py']), \
             mock.patch.object(sync_time.os, 'geteuid', return_value=0), \
             mock.patch.object(sync_time, 'remote', side_effect=[before, after, after]) as remote, \
             mock.patch.object(agent, 'check', side_effect=[before, after]), \
             mock.patch.object(agent, 'install', return_value=after) as install, \
             contextlib.redirect_stdout(io.StringIO()):
            sync_time.main()
        self.assertEqual([call.args[0] for call in remote.call_args_list], ['check', 'install', 'check'])
        install.assert_called_once()

    def test_failed_g1_preflight_never_changes_laptop(self):
        with mock.patch.object(sys, 'argv', ['sync_time.py']), \
             mock.patch.object(sync_time.os, 'geteuid', return_value=0), \
             mock.patch.object(sync_time, 'remote', side_effect=RuntimeError('offline')), \
             mock.patch.object(agent, 'install') as install, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                sync_time.main()
        install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
