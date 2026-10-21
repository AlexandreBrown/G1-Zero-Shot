"""Default idle loads without a picker; setup/keyboard can use an idle-only library."""
import tempfile,time
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
import numpy as np
from sonic_laptop.app import Panel,DemoSession
from sonic_laptop.model import load_model

model=load_model(Path(__file__).resolve().parents[2]/'GR00T-WholeBodyControl')
q=np.array(model['default']);q[[15,22]]=-.4;q[16]=.4;q[23]=-.4;q[[18,25]]=1
with tempfile.TemporaryDirectory() as temp:
    source=Path(temp)/'idle.csv';np.savetxt(source,np.tile(np.r_[0,0,.793,1,0,0,0,q],(61,1)),delimiter=',')
    root=tk.Tk();s=DemoSession([dict(index=0,name='Idle',frames=50)]);s.library_valid=False
    p=Panel(root,s,s.catalog);root.update()
    with patch('sonic_laptop.idle.DEFAULT_IDLE',source),patch('tkinter.filedialog.askdirectory',side_effect=AssertionError('No idle picker expected')):
        p.load_library(default_only=True)
        deadline=time.monotonic()+15
        while p.library_busy:root.update();time.sleep(.01);assert time.monotonic()<deadline
    assert p.library_valid and len(p.catalog)==1
    p.arm();p.start_grounded();p.tick()
    assert str(p.planner_button['state'])=='normal'
    assert str(p.play_button['state'])=='disabled'
    p.enable_keyboard();assert s.status['planner']
    root.destroy()
print('Default idle UI passed: no picker, fresh conversion, idle-only setup and keyboard.')
