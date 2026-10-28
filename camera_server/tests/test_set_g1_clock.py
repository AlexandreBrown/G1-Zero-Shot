import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import set_g1_clock


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

    def test_unsynchronized_laptop_never_connects_to_set_clock(self):
        with mock.patch.object(sys, 'argv', ['set_g1_clock.py']), \
             mock.patch.object(set_g1_clock.subprocess, 'run',
                               return_value=subprocess.CompletedProcess([], 0, stdout='no\n', stderr='')), \
             mock.patch.object(set_g1_clock, 'RobotClock') as remote:
            with self.assertRaisesRegex(RuntimeError, 'not synchronized'):
                set_g1_clock.main()
        remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
