import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_network


class InternetTimeSyncTests(unittest.TestCase):
    def test_does_not_touch_time_service_without_our_time_configuration(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(g1_network, 'TIME_CONFIG', Path(directory) / 'absent'), \
             mock.patch.object(g1_network, 'run') as run:
            self.assertIsNone(g1_network.refresh_time_sync('test'))
            run.assert_not_called()

    def test_restarts_time_service_and_confirms_synchronization(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'time.conf'
            config.touch()
            with mock.patch.object(g1_network, 'TIME_CONFIG', config), \
                 mock.patch.object(g1_network, 'run', side_effect=['active', '', 'yes']) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(g1_network.refresh_time_sync('G1'))
                self.assertEqual(run.call_args_list[1].args,
                                 ('systemctl', 'restart', 'systemd-timesyncd.service'))

    def test_time_service_failure_is_reported_without_breaking_sharing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'time.conf'
            config.touch()
            failure = subprocess.CalledProcessError(1, 'systemctl')
            with mock.patch.object(g1_network, 'TIME_CONFIG', config), \
                 mock.patch.object(g1_network, 'run', side_effect=['active', failure]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(g1_network.refresh_time_sync('G1'))
                self.assertIn('could not refresh', output.getvalue())


if __name__ == '__main__':
    unittest.main()
