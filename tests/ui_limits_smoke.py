"""Offline acceptance editor: persistence, invalidation and lockout."""
import tempfile
from pathlib import Path
import tkinter as tk
from sonic_laptop.app import DemoSession,Panel
from sonic_laptop.limits import load

with tempfile.TemporaryDirectory() as temp:
    root=tk.Tk();session=DemoSession([dict(index=0,name='Idle',frames=50)])
    session.limits_path=Path(temp)/'limits.json';session.library_valid=False
    panel=Panel(root,session,session.catalog);root.update()
    assert str(panel.arm_button['state'])=='disabled'
    panel.open_limits()
    d=panel.limits_dialog
    d.values['reference_speed'].set('nan');d.apply()
    assert panel.limits_dialog is d and not session.limits_path.exists()
    d.values['reference_speed'].set('7');d.apply();root.update()
    assert panel.limits_dialog is None and load(session.limits_path)['reference_speed']==7
    assert not panel.library_valid and str(panel.arm_button['state'])=='disabled'
    panel.arm();assert not session.armed
    panel.open_import();assert panel.import_dialog.limits['reference_speed']==7;panel.import_dialog.close()
    # A new launch uses the saved profile; only a revalidated library can unlock setup.
    panel.library_valid=True;panel.arm();assert session.armed
    panel.open_limits();assert panel.limits_dialog is None
    try:panel.apply_limits({'reference_speed':4})
    except RuntimeError:pass
    else:raise AssertionError('Live limits editing must fail')
    root.destroy()
print('Limits UI passed: validation, persistence, import propagation, reload requirement and setup lockout.')
