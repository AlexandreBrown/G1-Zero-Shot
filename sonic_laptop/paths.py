"""Persistent per-checkout runtime, separate from source assets and disposable reports."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'.runtime/sonic'
DEPLOY=RUNTIME/'deploy'
SESSIONS=RUNTIME/'sessions'
