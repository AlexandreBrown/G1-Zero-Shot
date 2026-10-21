"""Shared model extraction: hardware order comes from upstream policy_parameters.hpp."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def load_model(upstream):
    deploy = Path(upstream) / 'gear_sonic_deploy'
    text = (deploy / 'src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
    defaults = re.search(r'default_angles = \{(.*?)\};', text, re.S)[1]
    pairs = re.findall(r'([-\d.]+),?\s*//\s*(\w+)', defaults)
    names = [p[1] for p in pairs]
    assert len(names) == 29, 'Unknown upstream joint order'
    mapping = [int(x) for x in re.search(r'isaaclab_to_mujoco = \{(.*?)\};', text, re.S)[1].split(',')]
    tree = ET.parse(deploy / 'g1/g1_29dof.xml')
    bodies, limits, torques = [], {}, {}
    def visit(node, parent):
        index = len(bodies)
        joint = node.find('joint')
        name = joint.get('name') if joint is not None else None
        j = names.index(name) if name in names else -1
        if j >= 0:
            assert joint.get('pos', '0 0 0') == '0 0 0'
            limits[name] = list(map(float, joint.get('range').split()))
            torques[name] = max(map(abs, map(float, joint.get('actuatorfrcrange').split())))
        def vec(attr, default):
            return list(map(float, node.get(attr, default).split()))
        bodies.append(dict(name=node.get('name'), parent=parent, joint=j,
                           pos=vec('pos', '0 0 0'), quat=vec('quat', '1 0 0 0'),
                           axis=list(map(float, joint.get('axis', '0 0 1').split())) if j >= 0 else [0,0,1]))
        for child in node.findall('body'):
            visit(child, index)
    for body in tree.findall('./worldbody/body'):
        visit(body, -1)
    return dict(names=names, default=[float(p[0]) for p in pairs], mapping=mapping,
                lower=[limits[n][0]+0.03 for n in names], upper=[limits[n][1]-0.03 for n in names],
                torque=[torques[n] for n in names], bodies=bodies,
                # Preliminary conservative envelopes, not a calibrated U4 collision model.
                hand_radius=0.12, hand_offset=[0.08,0,0], clearance=0.025,
                boxes=[['torso_link', [0,0,0.16], [0.14,0.14,0.22]],
                       ['pelvis', [0,0,-0.045], [0.14,0.13,0.12]],
                       ['torso_link', [0,0,0.43], [0.12,0.12,0.13]],
                       *[[side+'_hip_yaw_link',[0,0,-0.10],[0.08,0.08,0.14]] for side in ('left','right')],
                       *[[side+'_knee_link',[0,0,-0.15],[0.07,0.07,0.17]] for side in ('left','right')]])


def rotation(q):
    w,x,y,z = np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def clearance(model, q):
    transforms = []
    for b in model['bodies']:
        r = rotation(b['quat'])
        if b['joint'] >= 0:
            a = np.asarray(b['axis']); angle = q[b['joint']]/2
            r = r @ rotation([np.cos(angle), *(a*np.sin(angle))])
        t = np.eye(4); t[:3,:3] = r; t[:3,3] = b['pos']
        transforms.append((transforms[b['parent']] @ t) if b['parent'] >= 0 else t)
    by_name = {b['name']: t for b,t in zip(model['bodies'], transforms)}
    hands = [(by_name[s+'_wrist_yaw_link'] @ [*model['hand_offset'],1])[:3] for s in ('left','right')]
    distances = [np.linalg.norm(hands[0]-hands[1])-2*model['hand_radius']]
    for hand in hands:
        for name, center, half in model['boxes']:
            t = by_name[name]; p = t[:3,:3].T @ (hand-t[:3,3])
            delta = np.abs(p-center)-half
            distances.append(np.linalg.norm(np.maximum(delta,0))+min(max(delta),0)-model['hand_radius'])
    return min(distances)
