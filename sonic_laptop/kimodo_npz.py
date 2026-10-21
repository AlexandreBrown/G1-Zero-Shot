"""Helper executed in a Kimodo Python environment; no model inference or GPU needed."""
import argparse
import numpy as np


def export(source, destination):
    # Use NVIDIA's converter rather than duplicating skeleton/rest-pose transforms.
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import G1Skeleton34
    with np.load(source, allow_pickle=False) as archive:
        rotations=np.asarray(archive['local_rot_mats'],dtype=np.float64)
        roots=np.asarray(archive['root_positions'],dtype=np.float64)
    if rotations.ndim!=4 or rotations.shape[1:]!=(34,3,3):
        raise ValueError('Only single-clip Kimodo-G1 NPZ (T,34,3,3) is supported; human motions need retargeting')
    if roots.shape!=(len(rotations),3) or len(roots)<2:
        raise ValueError('root_positions must be (T,3), with at least two frames')
    if not np.isfinite(rotations).all() or not np.isfinite(roots).all():
        raise ValueError('NPZ contains nonfinite motion data')
    if (np.max(np.abs(rotations.swapaxes(-1,-2)@rotations-np.eye(3)))>.005 or
            np.max(np.abs(np.linalg.det(rotations)-1))>.005):
        raise ValueError('NPZ local_rot_mats must be proper rotation matrices')
    converter=MujocoQposConverter(G1Skeleton34())
    qpos=converter.dict_to_qpos(dict(local_rot_mats=rotations.astype(np.float32),
                                    root_positions=roots.astype(np.float32)),
                                  device='cpu',root_quat_w_first=True,numpy=True,mujoco_rest_zero=False)
    qpos=np.asarray(qpos)
    if qpos.shape==(1,len(roots),36):qpos=qpos[0]
    if qpos.shape!=(len(roots),36):raise ValueError(f'Unexpected Kimodo exporter shape: {qpos.shape}')
    np.savetxt(destination,qpos,delimiter=',')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    export(a.source,a.destination)
