"""Motion acceptance settings, separate from the controller's fixed runtime guards."""
import json
import math
from pathlib import Path

from .paths import RUNTIME
DEFAULT_PATH=RUNTIME/'acceptance-limits.json'
# key: label, default, minimum, maximum
FIELDS={
    'reference_speed':('Reference joint speed (rad/s)',4.0,0.1,8.0),
    'velocity_mismatch':('Position/velocity mismatch (rad/s)',1.0,0.1,4.0),
    'root_displacement':('Root travel from start (m)',0.15,0.01,1.0),
    'root_tilt':('Root tilt (rad)',0.20,0.01,0.50),
    'hand_clearance':('Minimum hand envelope clearance (m)',0.025,0.025,0.20),
}
DEFAULTS={key:spec[1] for key,spec in FIELDS.items()}


def validate(values=None):
    values={} if values is None else values
    if not isinstance(values,dict):raise ValueError('Limits must be a JSON object')
    if set(values)-set(FIELDS):raise ValueError('Unknown acceptance limit: '+', '.join(set(values)-set(FIELDS)))
    result=dict(DEFAULTS)
    for key,value in values.items():
        label,_,low,high=FIELDS[key]
        if isinstance(value,bool):raise ValueError(f'{label}: enter a number')
        try:value=float(value)
        except (ValueError,TypeError) as error:raise ValueError(f'{label}: enter a number') from error
        if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{label}: allowed range {low:g}–{high:g}')
        result[key]=value
    return result


def load(path=None):
    path=Path(path or DEFAULT_PATH)
    if not path.exists():
        if path!=DEFAULT_PATH:raise FileNotFoundError(path)
        return dict(DEFAULTS)
    return validate(json.loads(path.read_text()))


def save(path,values):
    path=Path(path);values=validate(values);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(values,indent=2)+'\n');temporary.replace(path)
    return values
