"""Offline numeric screening of Kimodo G1 exports; never publishes robot commands."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .import_kimodo import load_qpos, resample, derivative, G1_JOINTS
from .model import load_model, clearance
from .limits import validate as validate_limits, load as load_limits


def audit(source, fps, upstream, limits=None):
    limits=validate_limits(limits)
    source=Path(source);model=load_model(upstream)
    if tuple(model['names'])!=G1_JOINTS:raise ValueError('Unsupported joint ordering')
    raw=load_qpos(source);sample=resample(raw,fps);q=sample[:,7:];velocity=derivative(q)
    speed=np.abs(np.diff(q,axis=0)*50)
    at=np.unravel_index(speed.argmax(),speed.shape)
    distances=np.array([clearance(model,pose) for pose in q])
    tilt=np.arccos(np.clip(1-2*(sample[:,4]**2+sample[:,5]**2),-1,1))
    displacement=np.linalg.norm(sample[:,:2]-sample[0,:2],axis=1)
    violations=(q<model['lower'])|(q>model['upper'])
    mismatch=float(np.abs(np.diff(q,axis=0)*50-velocity[:-1]).max())
    checks={
        'at_least_one_second':len(q)>=50,
        'joint_limits_with_margin':not bool(violations.any()),
        'reference_speed':bool(max(speed.max(),np.abs(velocity).max())<=limits['reference_speed']),
        'velocity_consistency':mismatch<=limits['velocity_mismatch'],
        'root_tilt':bool(tilt.max()<=limits['root_tilt']),
        'root_displacement':bool(displacement.max()<=limits['root_displacement']),
        'hand_clearance':bool(distances.min()>=limits['hand_clearance']),
    }
    windows={}
    for name,window in [('start',slice(0,25)),('end',slice(-25,None))]:
        deviation=float(np.abs(q[window]-q[window][0]).max());v=float(np.abs(velocity[window]).max())
        checks[name+'_stationary_dwell']=deviation<=.025 and v<=.10
        windows[name]=dict(max_joint_deviation_rad=deviation,max_joint_speed_rad_s=v)
    return dict(acceptance_limits=limits,source=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_fps=fps,source_frames=len(raw),nominal_duration_seconds=len(raw)/fps,
        sampled_time_span_seconds=(len(raw)-1)/fps,output_frames=len(q),output_fps=50,
        checks=checks,passed=all(checks.values()),
        maximum_joint_speed_rad_s=float(speed.max()),fastest_joint=model['names'][at[1]],
        fastest_interval_start_seconds=float(at[0]/50),minimum_hand_envelope_clearance_m=float(distances.min()),
        minimum_clearance_time_seconds=float(distances.argmin()/50),
        max_root_tilt_rad=float(tilt.max()),max_horizontal_displacement_m=float(displacement.max()),
        endpoint_dwell=windows,
        limitations='Reference screening only. Hand envelopes are approximate, not calibrated Dex3 collision geometry. '
        'Idle matching, full SONIC GPU inference, dynamic stability, physical stop response and network failures require separate validation.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    p.add_argument('--source-fps',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path(__file__).resolve().parents[2]/'GR00T-WholeBodyControl')
    p.add_argument('--limits',type=Path,help='Acceptance-limit JSON; defaults to saved app settings')
    a=p.parse_args();report=audit(a.source,a.source_fps,a.upstream,load_limits(a.limits))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
