"""Resolve the bundled Kimodo idle into a fresh session-local SONIC reference."""
from pathlib import Path
from .import_kimodo import convert

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_IDLE=ROOT/'motions/kimodo/idle.csv'
DEFAULT_IDLE_FPS=30  # Kimodo G1 export convention; CSV itself has no time metadata.


def resolve_idle(override,directory,upstream,limits=None):
    if override is not None:return Path(override).expanduser().resolve()
    if not DEFAULT_IDLE.is_file():
        raise ValueError(f'Default idle missing: {DEFAULT_IDLE}. Restore it or choose a custom SONIC idle directory.')
    output=Path(directory)/'default-idle'
    report=convert(DEFAULT_IDLE,output,DEFAULT_IDLE_FPS,upstream,limits=limits)
    if not report['commissioning_checks_passed']:
        raise ValueError('Default idle rejected: '+report['commissioning_error']+'. Choose another idle or correct the source motion.')
    return output
