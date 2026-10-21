"""Offline Kimodo-G1 CSV/NPZ -> 50 Hz SONIC reference directories. Never controls hardware."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .model import load_model
from .motions import read_clip

ROOT=Path(__file__).resolve().parents[1]
# Default Kimodo MuJoCo CSV traversal order. Do not infer ordering from numeric data.
G1_JOINTS=tuple(
    f'{side}_{part}_joint' for side in ('left','right')
    for part in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')
)+('waist_yaw_joint','waist_roll_joint','waist_pitch_joint')+tuple(
    f'{side}_{part}_joint' for side in ('left','right')
    for part in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')
)


def load_qpos(source, kimodo_python=None):
    source=Path(source)
    if source.suffix.lower()=='.npz':
        # Reject human/object/malformed archives before invoking the optional exporter.
        with np.load(source,allow_pickle=False) as archive:
            if 'local_rot_mats' not in archive or 'root_positions' not in archive:
                raise ValueError('Expected Kimodo-G1 NPZ with local_rot_mats and root_positions')
            r=archive['local_rot_mats'];p=archive['root_positions']
            if r.ndim!=4 or r.shape[1:]!=(34,3,3) or p.shape!=(len(r),3):
                raise ValueError('Only Kimodo-G1 (34-joint) single clips are supported; retarget SOMA/SMPL-X first')
        with tempfile.TemporaryDirectory(prefix='kimodo-export-') as temp:
            csv=Path(temp)/'qpos.csv'
            result=subprocess.run([str(kimodo_python or sys.executable),str(Path(__file__).with_name('kimodo_npz.py')),
                                   str(source.resolve()),str(csv)],capture_output=True,text=True,timeout=180)
            if result.returncode:
                raise ValueError('Kimodo NPZ export failed. Supply --kimodo-python /path/to/kimodo-env/bin/python '
                                 'or export default G1 CSV in Kimodo.\n'+result.stderr[-3000:])
            return load_qpos(csv)
    if source.suffix.lower()!='.csv':raise ValueError('Input must be Kimodo-G1 .csv or .npz')
    # Official save_csv writes headerless numeric rows. Never skip an unrecognized row.
    try:qpos=np.loadtxt(source,delimiter=',',ndmin=2)
    except ValueError as e:raise ValueError(f'{source}: expected headerless Kimodo-G1 numeric CSV') from e
    if qpos.shape[1]!=36 or len(qpos)<2:
        raise ValueError(f'{source}: expected at least two rows of 36 values; got {qpos.shape}')
    if not np.isfinite(qpos).all():raise ValueError('Nonfinite qpos values')
    norms=np.linalg.norm(qpos[:,3:7],axis=1)
    if np.any(np.abs(norms-1)>.005):raise ValueError('Root quaternion must be unit length, in wxyz order')
    qpos[:,3:7]/=norms[:,None]
    return qpos


def resample(qpos,source_fps):
    if not math.isfinite(source_fps) or source_fps<=0:
        raise ValueError('source_fps must be finite and positive')
    source_time=np.arange(len(qpos),dtype=float)/source_fps
    duration=source_time[-1]
    # Preserve endpoint, without speeding up the clip: at most <20 ms terminal hold.
    count=int(math.ceil(duration*50-1e-9))+1
    if count>1_000_000:raise ValueError('Output would exceed one million frames; verify source FPS')
    target_time=np.arange(count,dtype=float)/50
    sample_time=np.minimum(target_time,duration)
    result=np.empty((count,36))
    for j in [0,1,2,*range(7,36)]:
        result[:,j]=np.interp(sample_time,source_time,qpos[:,j])
    # Bounded joints interpolate as real angles, not cyclic shortest paths.
    # Quaternion interpolation does use the shortest rotation and handles q/-q.
    rotations=Rotation.from_quat(qpos[:,[4,5,6,3]])
    result[:,3:7]=Slerp(source_time,rotations)(sample_time).as_quat()[:,[3,0,1,2]]
    for i in range(1,len(result)):
        if result[i,3:7]@result[i-1,3:7]<0:result[i,3:7]*=-1
    return result


def derivative(values):
    return np.gradient(values,.02,axis=0,edge_order=2 if len(values)>2 else 1)


def convert(source,output,source_fps,upstream,kimodo_python=None):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    model=load_model(upstream)
    if tuple(model['names'])!=G1_JOINTS:raise ValueError('Unexpected upstream hardware joint order')
    raw=load_qpos(source,kimodo_python)
    qpos=resample(raw,source_fps)
    joints=np.empty((len(qpos),29));joints[:,model['mapping']]=qpos[:,7:]
    root=qpos[:,3:7]
    rotations=Rotation.from_quat(root[:,[1,2,3,0]])
    # Adjacent orientation increments expressed in world coordinates, then centered.
    increments=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/.02
    angular=np.empty((len(qpos),3));angular[0]=increments[0];angular[-1]=increments[-1]
    if len(qpos)>2:angular[1:-1]=(increments[:-1]+increments[1:])/2
    arrays={'joint_pos':joints,'joint_vel':derivative(joints),
            'body_pos':qpos[:,:3],'body_quat':root,
            'body_lin_vel':derivative(qpos[:,:3]),'body_ang_vel':angular}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.kimodo-import-',dir=output.parent) as temp:
        staging=Path(temp)/'clip';staging.mkdir()
        for name,values in arrays.items():
            np.savetxt(staging/(name+'.csv'),values,delimiter=',',fmt='%.10g',comments='',
                       header=','.join(f'{name}_{i}' for i in range(values.shape[1])))
        (staging/'metadata.txt').write_text(f'Metadata for: {output.name}\n\nBody part indexes:\n[0]\n\nTotal timesteps: {len(qpos)}\n')
        error=None
        try:read_clip(staging,model)
        except ValueError as e:error=str(e).replace(str(staging),output.name)
        report=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_fps=source_fps,source_frames=len(raw),target_fps=50,target_frames=len(qpos),
                    source_duration_seconds=(len(raw)-1)/source_fps,
                    output_duration_seconds=(len(qpos)-1)/50,
                    convention='Kimodo default G1 MuJoCo CSV: z-up, x-forward, wxyz, raw joint DOFs',
                    body_indexes=[0],encoder_mode=0,commissioning_checks_passed=error is None,
                    commissioning_error=error,
                    notes='Root-only body data is for SONIC joint-based encoder mode 0. '
                          'No retargeting, clipping, standing transitions or endpoint dwell was added. '
                          'Idle matching and playlist checks still run when opening the control panel.')
        (staging/'import-report.json').write_text(json.dumps(report,indent=2)+'\n')
        # Destination must still be absent; normal use never replaces existing clips.
        if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
        os.rename(staging,output)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,help='One Kimodo-G1 CSV/NPZ, or a folder (nonrecursive)')
    p.add_argument('--output',type=Path,required=True,help='New clip directory, or new library directory for folder import')
    p.add_argument('--source-fps',type=float,required=True,help='Actual generation/export FPS; never inferred from CSV')
    p.add_argument('--upstream',type=Path,default=ROOT.parent/'GR00T-WholeBodyControl')
    p.add_argument('--kimodo-python',type=Path,help='Python executable with Kimodo installed; only needed for NPZ')
    a=p.parse_args(argv)
    if not math.isfinite(a.source_fps) or a.source_fps<=0:p.error('--source-fps must be finite and positive')
    if a.output.exists():p.error('Output already exists; choose a new destination')
    inputs=sorted(f for f in a.source.iterdir() if f.is_file() and f.suffix.lower() in ('.csv','.npz')) if a.source.is_dir() else [a.source]
    if not inputs:p.error('No CSV/NPZ files found')
    if len({f.stem for f in inputs})!=len(inputs):p.error('Duplicate clip stems (e.g. both wave.csv and wave.npz); select one export format')
    failed=False
    for source in inputs:
        output=a.output/source.stem if a.source.is_dir() else a.output
        try:
            report=convert(source,output,a.source_fps,a.upstream,a.kimodo_python)
            if report['commissioning_checks_passed']:
                print(f'CONVERTED: {output} — {report["target_frames"]} frames at 50 Hz; per-clip checks passed')
            else:
                failed=True
                print(f'CONVERTED FOR REVIEW: {output}\n  Commissioning check failed: {report["commissioning_error"]}')
        except (ValueError,OSError,KeyError,subprocess.TimeoutExpired) as e:
            failed=True;print(f'IMPORT FAILED: {source}: {e}',file=sys.stderr)
    return 2 if failed else 0

if __name__=='__main__':raise SystemExit(main())
