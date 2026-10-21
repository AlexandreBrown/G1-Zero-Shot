import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sonic_laptop.limits import DEFAULTS,load,save,validate
from sonic_laptop.motions import read_clip,stage_library
from tests import test_laptop as fixtures

class LimitsTests(unittest.TestCase):
    setUp=fixtures.LaptopTests.setUp
    tearDown=fixtures.LaptopTests.tearDown
    clip=fixtures.LaptopTests.clip
    def test_profile_validation(self):
        for values in ({'reference_speed':float('nan')},{'reference_speed':99},
                       {'hand_clearance':0},{'unknown':1},{'root_tilt':True}):
            with self.assertRaises(ValueError):validate(values)
        file=self.p/'limits.json';save(file,{'reference_speed':7})
        self.assertEqual(load(file)['reference_speed'],7)
        self.assertEqual(load(file)['hand_clearance'],.025)
    def test_limits_change_actual_screening_and_snapshot(self):
        idle=self.clip('idle');clip=self.clip('reach')
        body=np.loadtxt(clip/'body_pos.csv',delimiter=',',skiprows=1)
        body[25,0]=.2
        np.savetxt(clip/'body_pos.csv',body,delimiter=',',header='x,y,z',comments='')
        with self.assertRaisesRegex(ValueError,'root translation'):read_clip(clip,self.model)
        limits=validate({'root_displacement':.25})
        read_clip(clip,self.model,limits)
        stage_library(idle,self.p,self.p/'session/motions',self.model,limits)
        self.assertEqual(json.loads((self.p/'session/acceptance-limits.json').read_text()),limits)
        limits['root_displacement']=.9
        self.assertEqual(json.loads((self.p/'session/acceptance-limits.json').read_text())['root_displacement'],.25)

    def test_cli_import_uses_profile(self):
        from sonic_laptop.import_kimodo import main
        raw=np.tile(np.r_[0,0,.793,1,0,0,0,self.q],(61,1));raw[30,0]=.2
        source=self.p/'source.csv';np.savetxt(source,raw,delimiter=',')
        profile=self.p/'profile.json';save(profile,{'root_displacement':.3})
        output=self.p/'imported'
        result=main([str(source),'--source-fps','30','--limits',str(profile),'--output',str(output)])
        self.assertEqual(result,0)
        report=json.loads((output/'import-report.json').read_text())
        self.assertEqual(report['acceptance_limits']['root_displacement'],.3)
        self.assertTrue(report['commissioning_checks_passed'])
