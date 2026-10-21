"""Virtual-display import workflow test. Only synthetic motion data, no hardware."""
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
import numpy as np
from sonic_laptop.app import Panel,DemoSession
from sonic_laptop.model import load_model

ROOT=Path(__file__).resolve().parents[1]
model=load_model(ROOT.parent/'GR00T-WholeBodyControl')
q=np.array(model['default']);q[[15,22]]=-.4;q[16]=.4;q[23]=-.4;q[[18,25]]=1

def wait(root,predicate):
    deadline=time.monotonic()+15
    while not predicate():
        root.update();time.sleep(.01)
        assert time.monotonic()<deadline,'UI worker timed out'
    root.update()

with tempfile.TemporaryDirectory() as temp:
    path=Path(temp)
    raw=np.tile(np.r_[0,0,.793,1,0,0,0,q],(61,1))
    for name in ('idle','wave'):np.savetxt(path/(name+'.csv'),raw,delimiter=',')
    catalog=[dict(index=0,name='Idle',frames=50)]
    session=DemoSession(catalog);session.offline=True
    root=tk.Tk();panel=Panel(root,session,catalog);root.update()
    assert str(panel.arm_button['state'])=='disabled'
    panel.open_import();dialog=panel.import_dialog
    assert str(panel.load_button['state'])=='disabled'
    dialog.set_sources([path/'idle.csv',path/'wave.csv'])
    dialog.output.set(str(path/'converted'));dialog.fps.set('-1');dialog.start()
    assert not dialog.running and not (path/'converted').exists()
    dialog.fps.set('30');dialog.start()
    assert dialog.running
    panel.arm();assert not session.armed
    wait(root,lambda:not dialog.running)
    assert len(dialog.results.get_children())==2
    assert all(dialog.results.item(row,'values')[1]=='Checks passed' for row in dialog.results.get_children())
    assert (path/'converted/wave/import-report.json').exists()
    dialog.close();assert panel.import_dialog is None
    with patch('tkinter.filedialog.askdirectory',return_value=str(path/'converted/idle')):
        panel.load_library(path/'converted')
    wait(root,lambda:not panel.library_busy)
    assert [c['name'] for c in panel.catalog]==['idle','wave']
    assert panel.listbox.size()==1
    assert str(panel.arm_button['state'])=='disabled'
    panel.arm();assert not session.armed
    # Guard methods themselves, not only disabled widgets, against live changes.
    session.armed=True;panel.update_buttons(None)
    assert str(panel.import_button['state'])=='disabled'
    panel.open_import();assert panel.import_dialog is None
    root.destroy()
print('Offline UI import passed: file conversion, validation results, library loading and arm lockout.')
