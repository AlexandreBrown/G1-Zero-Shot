import json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from tests import test_laptop as fixtures
from sonic_laptop.idle import resolve_idle
from sonic_laptop.motions import stage_library

UPSTREAM=Path(__file__).resolve().parents[2]/'GR00T-WholeBodyControl'

class IdleTests(unittest.TestCase):
    setUp=fixtures.LaptopTests.setUp
    tearDown=fixtures.LaptopTests.tearDown
    clip=fixtures.LaptopTests.clip
    def test_default_is_converted_and_staged_without_motion_folder(self):
        source=self.p/'idle.csv';np.savetxt(source,np.tile(np.r_[0,0,.793,1,0,0,0,self.q],(61,1)),delimiter=',')
        with patch('sonic_laptop.idle.DEFAULT_IDLE',source):
            idle=resolve_idle(None,self.p/'session',UPSTREAM)
        catalog=stage_library(idle,None,self.p/'session/motions',self.model)
        self.assertEqual(len(catalog),1)
        self.assertEqual(catalog[0]['frames'],101)
        self.assertTrue((self.p/'session/init.txt').is_file())
        report=json.loads((idle/'import-report.json').read_text())
        self.assertEqual(report['source_fps'],30)
    def test_override_does_not_read_default(self):
        with patch('sonic_laptop.idle.DEFAULT_IDLE',self.p/'missing.csv'):
            self.assertEqual(resolve_idle(self.p/'custom',self.p,UPSTREAM),self.p/'custom')
            with self.assertRaisesRegex(ValueError,'Default idle missing'):resolve_idle(None,self.p,UPSTREAM)
    def test_rejected_default_does_not_create_staged_library(self):
        source=self.p/'idle.csv';raw=np.tile(np.r_[0,0,.793,1,0,0,0,self.q],(61,1));raw[:,7]=100
        np.savetxt(source,raw,delimiter=',')
        with patch('sonic_laptop.idle.DEFAULT_IDLE',source):
            with self.assertRaisesRegex(ValueError,'Default idle rejected'):resolve_idle(None,self.p/'session',UPSTREAM)
        self.assertFalse((self.p/'session/init.txt').exists())
