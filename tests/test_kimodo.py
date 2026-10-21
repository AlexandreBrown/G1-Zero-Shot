import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from sonic_laptop.import_kimodo import convert, load_qpos, main, resample
from sonic_laptop.model import load_model
from sonic_laptop.motions import read_clip, stage_library

UPSTREAM=Path(__file__).resolve().parents[2]/'GR00T-WholeBodyControl'

class KimodoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=load_model(UPSTREAM)
        q=np.array(self.model['default']);q[[15,22]]=-.4;q[16]=.4;q[23]=-.4;q[[18,25]]=1
        self.qpos=np.tile(np.r_[0,0,.793,1,0,0,0,q],(91,1))
    def tearDown(self):self.temp.cleanup()
    def save(self,name,values=None):
        path=self.root/name;np.savetxt(path,self.qpos if values is None else values,delimiter=',');return path
    def test_mapping_velocity_and_library(self):
        source=self.save('idle.csv');library=self.root/'library'
        report=convert(source,library/'idle',30,UPSTREAM)
        self.assertTrue(report['commissioning_checks_passed']);self.assertEqual(report['target_frames'],151)
        wave=self.qpos.copy();time=np.arange(91)/30
        window=(time>1)&(time<2)
        wave[window,7+19]+=.06*np.sin(np.pi*(time[window]-1))**2
        report=convert(self.save('wave.csv',wave),library/'wave',30,UPSTREAM)
        self.assertTrue(report['commissioning_checks_passed'])
        arrays,q=read_clip(library/'wave',self.model)
        np.testing.assert_allclose(q[0],self.qpos[0,7:],atol=1e-8)
        np.testing.assert_allclose(arrays['joint_pos'][:,self.model['mapping'][19]],q[:,19])
        self.assertGreater(np.max(np.abs(arrays['joint_vel'])),.1)
        catalog=stage_library(library/'idle',library,self.root/'session/motions',self.model)
        self.assertEqual([c['name'] for c in catalog],['idle','wave'])
        self.assertEqual((library/'wave/metadata.txt').read_text().split('Body part indexes:\n')[1].splitlines()[0],'[0]')
    def test_time_and_quaternion_shortest_path(self):
        raw=self.qpos[:2].copy();raw[0,3:7]=Rotation.from_euler('z',170,degrees=True).as_quat()[[3,0,1,2]]
        raw[1,3:7]=-Rotation.from_euler('z',-170,degrees=True).as_quat()[[3,0,1,2]]
        raw[1,0]=1
        result=resample(raw,2)
        self.assertEqual(len(result),26)
        np.testing.assert_allclose(result[:,0],np.arange(26)/25)
        angle=Rotation.from_quat(result[12,[4,5,6,3]]).as_euler('xyz',degrees=True)[2]
        self.assertGreater(abs(angle),179)
        self.assertTrue(np.all(np.sum(result[1:,3:7]*result[:-1,3:7],axis=1)>0))
        uneven=resample(self.qpos[:3],30)
        self.assertEqual(len(uneven),5)
        np.testing.assert_allclose(uneven[-1],self.qpos[2])
        with self.assertRaises(ValueError):resample(raw,float('nan'))
    def test_linear_velocity_units(self):
        raw=self.qpos.copy();raw[:,7]+=.2*np.arange(91)/30
        result=resample(raw,30)
        from sonic_laptop.import_kimodo import derivative
        np.testing.assert_allclose(derivative(result[:,7]),.2,atol=1e-12)
    def test_invalid_csv_and_no_overwrite(self):
        for label,data in [('nan',self.qpos.copy()),('quat',self.qpos.copy()),('shape',self.qpos[:,:35])]:
            if label=='nan':data[0,7]=np.nan
            if label=='quat':data[0,3:7]=0
            with self.subTest(label=label),self.assertRaises(ValueError):load_qpos(self.save(label+'.csv',data))
        file=self.root/'header.csv';file.write_text('not,a,numeric,header\n')
        with self.assertRaises(ValueError):load_qpos(file)
        source=self.save('ok.csv');dest=self.root/'out';convert(source,dest,30,UPSTREAM)
        before=(dest/'joint_pos.csv').read_bytes()
        with self.assertRaises(FileExistsError):convert(source,dest,30,UPSTREAM)
        self.assertEqual(before,(dest/'joint_pos.csv').read_bytes())
    def test_failed_guard_preserves_unmodified_reference(self):
        bad=self.qpos.copy();bad[:,7]=100
        report=convert(self.save('bad.csv',bad),self.root/'review',30,UPSTREAM)
        self.assertFalse(report['commissioning_checks_passed'])
        self.assertIn('limit',report['commissioning_error'])
        actual=np.loadtxt(self.root/'review/joint_pos.csv',delimiter=',',skiprows=1)
        self.assertTrue(np.all(actual[:,0]==100))
        self.assertEqual(json.loads((self.root/'review/import-report.json').read_text())['target_fps'],50)
    def test_npz_routes_to_official_exporter_and_rejects_human(self):
        human=self.root/'human.npz';np.savez(human,local_rot_mats=np.zeros((2,22,3,3)),root_positions=np.zeros((2,3)))
        with self.assertRaisesRegex(ValueError,'34-joint'):load_qpos(human)
        g1=self.root/'g1.npz';np.savez(g1,local_rot_mats=np.tile(np.eye(3),(91,34,1,1)),root_positions=np.zeros((91,3)))
        def fake_export(args,**kwargs):
            self.assertEqual(args[0],'/kimodo/bin/python')
            self.assertTrue(args[1].endswith('kimodo_npz.py'))
            np.savetxt(args[3],self.qpos,delimiter=',')
            from subprocess import CompletedProcess
            return CompletedProcess(args,0,'','')
        with patch('sonic_laptop.import_kimodo.subprocess.run',side_effect=fake_export):
            np.testing.assert_allclose(load_qpos(g1,Path('/kimodo/bin/python')),self.qpos)
    def test_folder_import_and_duplicate_names(self):
        folder=self.root/'inputs';folder.mkdir()
        for name in ('b','a'):np.savetxt(folder/(name+'.csv'),self.qpos,delimiter=',')
        self.assertEqual(main([str(folder),'--output',str(self.root/'converted'),'--source-fps','30','--upstream',str(UPSTREAM)]),0)
        self.assertTrue((self.root/'converted/a/import-report.json').exists())
        np.savez(folder/'a.npz',local_rot_mats=np.zeros((2,34,3,3)))
        with self.assertRaises(SystemExit):main([str(folder),'--output',str(self.root/'duplicates'),'--source-fps','30'])

if __name__=='__main__':unittest.main()
