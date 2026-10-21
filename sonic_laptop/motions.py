"""Validate and snapshot 50 Hz SONIC CSV directories before controller launch."""
import json
import re
import shutil
from pathlib import Path
import numpy as np
from .model import clearance


def read_clip(path, model):
    path = Path(path)
    arrays = {}
    for name, width in [('joint_pos',29), ('joint_vel',29), ('body_pos',3), ('body_quat',4)]:
        file = path / (name+'.csv')
        with file.open() as f:
            header = f.readline().strip().split(',')
        a = np.loadtxt(file, delimiter=',', skiprows=1, ndmin=2)
        if len(header) != a.shape[1] or not np.isfinite(a).all():
            raise ValueError(f'{file}: malformed header or nonfinite data')
        if (name.startswith('joint') and a.shape[1] != width) or a.shape[1] % width:
            raise ValueError(f'{file}: wrong shape {a.shape}')
        arrays[name] = a
    n = len(arrays['joint_pos'])
    if n < 50 or any(len(a) != n for a in arrays.values()):
        raise ValueError(f'{path}: require >=1 second and matching frame counts')
    metadata = (path/'metadata.txt').read_text()
    match = re.search(r'Body part indexes:\s*\[([^\]]+)\]', metadata)
    indices = [int(x) for x in match[1].split()] if match else []
    if not indices or indices[0] != 0 or len(set(indices)) != len(indices):
        raise ValueError(f'{path}: require unique metadata body indexes, pelvis first')
    if arrays['body_pos'].shape[1] != len(indices)*3 or arrays['body_quat'].shape[1] != len(indices)*4:
        raise ValueError(f'{path}: body metadata/CSV mismatch')
    quat = arrays['body_quat'].reshape(n,-1,4)
    if np.max(np.abs(np.linalg.norm(quat,axis=2)-1)) > .005:
        raise ValueError(f'{path}: non-unit quaternions')
    # Restrict commissioning clips to upright, near-stationary root trajectories.
    root = quat[:,0]
    if np.any(1-2*(root[:,1]**2+root[:,2]**2) < np.cos(.20)):
        raise ValueError(f'{path}: root tilt exceeds 0.20 rad')
    if np.max(np.linalg.norm(arrays['body_pos'][:,:2]-arrays['body_pos'][0,:2],axis=1)) > .15:
        raise ValueError(f'{path}: root translation exceeds 15 cm commissioning envelope')
    q = arrays['joint_pos'][:,model['mapping']]
    if np.any(q < model['lower']) or np.any(q > model['upper']):
        raise ValueError(f'{path}: joint limit violation (includes 0.03 rad margin)')
    if np.max(np.abs(arrays['joint_vel'])) > 4 or np.max(np.abs(np.diff(q,axis=0)*50)) > 4:
        raise ValueError(f'{path}: reference joint speed exceeds 4 rad/s')
    if np.max(np.abs(np.diff(arrays['joint_pos'],axis=0)*50-arrays['joint_vel'][:-1])) > 1:
        raise ValueError(f'{path}: joint_pos/joint_vel disagree by >1 rad/s')
    for i, pose in enumerate(q):
        if clearance(model,pose) < model['clearance']:
            raise ValueError(f'{path}: hand envelope clearance <25 mm at frame {i}')
    # Endpoints must dwell, rather than freeze a dynamic mid-stride posture.
    for window in (slice(0,25), slice(-25,None)):
        if np.max(np.abs(q[window]-q[window][0])) > .025 or np.max(np.abs(arrays['joint_vel'][window])) > .10:
            raise ValueError(f'{path}: require stationary first/last 0.5 seconds')
    return arrays, q


def stage_library(idle, folder, destination, model):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    candidates = [Path(idle)] + [p for p in sorted(Path(folder).iterdir())
                                if p.is_dir() and (p/'joint_pos.csv').exists() and p.resolve()!=Path(idle).resolve()]
    if len(candidates) < 2:
        raise ValueError('Motion folder contains no clips other than idle')
    catalog, anchor, anchor_arrays = [], None, None
    for i, path in enumerate(candidates):
        # Copy first, validate the immutable session snapshot, never symlink mutable motions.
        dest = destination/f'{i:04d}'
        dest.mkdir()
        for source in path.iterdir():
            if source.suffix == '.csv' or source.name == 'metadata.txt':
                shutil.copyfile(source,dest/source.name)
        arrays, q = read_clip(dest,model)
        if i == 0:
            anchor, anchor_arrays = q[0], arrays
            if np.max(np.abs(q-anchor)) > .025:
                raise ValueError('Idle must be stationary throughout')
            if np.max(np.abs(anchor[:15]-np.asarray(model['default'])[:15])) > .15:
                raise ValueError('Idle lower body must be within 0.15 rad of SONIC nominal stance')
        if np.max(np.abs(q[[0,-1]]-anchor)) > .10:
            raise ValueError(f'{path}: endpoints must match idle within 0.10 rad')
        if np.max(np.abs(arrays['body_pos'][[0,-1],:3]-anchor_arrays['body_pos'][0,:3])) > .05:
            raise ValueError(f'{path}: root endpoints must match idle within 5 cm')
        qa = arrays['body_quat'][[0,-1],:4]; qb = anchor_arrays['body_quat'][0,:4]
        if np.any(np.abs(qa @ qb) < np.cos(.10/2)):
            raise ValueError(f'{path}: root endpoint orientation mismatch >0.10 rad')
        # The held terminal reference must also have zero velocities for observation windows.
        arrays['joint_vel'][-25:] = 0
        np.savetxt(dest/'joint_vel.csv',arrays['joint_vel'],delimiter=',',comments='',
                   header=','.join(f'joint_{j}' for j in range(29)))
        catalog.append(dict(index=i,name=path.name,frames=len(q)))
    (destination.parent/'init.txt').write_text(' '.join(map(str,anchor))+'\n')
    (destination.parent/'catalog.json').write_text(json.dumps(catalog,indent=2))
    return catalog
