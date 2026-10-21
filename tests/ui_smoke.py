"""Run under xvfb-run; offline demo only, never launches a controller."""
import tkinter as tk
from types import SimpleNamespace
from sonic_laptop.app import Panel,DemoSession
catalog=[dict(index=i,name=name,frames=60) for i,name in enumerate(['Idle','Wave demo','Raise demo'])]
root=tk.Tk();session=DemoSession(catalog);p=Panel(root,session,catalog)
root.update()
assert p.stop_button.winfo_height()>=50
assert p.stop_button.winfo_y()+p.stop_button.winfo_height()<=root.winfo_height()
assert str(p.start_button['state'])=='disabled'
p.arm_button.focus_force()
root.update()
p.arm_button.event_generate('<KeyPress-space>');root.update()
p.arm_button.event_generate('<KeyRelease-space>');root.update()
assert not session.armed, 'Deadman Space must not click the focused ARM button'
p.arm();session.status['setup_ready']=False;p.update_buttons(session.status)
assert str(p.start_button['state'])=='disabled'
p.start_grounded();assert session.status['state']==1
session.status['setup_ready']=True;p.start_grounded();assert session.status['state']==2
p.enqueue([1,2]);p.tick()
assert session.status['index']==1
# Force acknowledged final frame, then verify queue proceeds from state, not time.
session.status.update(frame=59,playing=False);p.tick();p.tick()
assert session.status['index']==2
p.keys.update({'space','w'});p.enable_keyboard();assert not p.keys;p.tick()
p.key_down(SimpleNamespace(keysym='space'));p.key_down(SimpleNamespace(keysym='w'))
assert not p.keys, 'Autorepeat of held keys must not enable movement'
p.key_up(SimpleNamespace(keysym='space'));p.key_up(SimpleNamespace(keysym='w'))
import time
time.sleep(.03);root.update()
p.key_down(SimpleNamespace(keysym='space'));p.key_down(SimpleNamespace(keysym='w'))
assert {'space','w'} <= p.keys
from unittest.mock import Mock
original=session.command;session.command=Mock(wraps=original)
session.pending=True
p.key_up(SimpleNamespace(keysym='w'))
import time
time.sleep(.03);root.update()
assert 'w' not in p.keys
assert any(call.args==('idle',0) for call in session.command.call_args_list), 'Release must request idle even while pending'
session.pending=False
p.keys.update({'space','w'})
old_focus=root.focus_displayof;root.focus_displayof=lambda:None
p.check_focus();assert not p.keys
assert session.command.call_args.args==('idle',0)
root.focus_displayof=old_focus
p.stop();assert session.stopped and not p.keys
assert str(p.play_button['state'])=='disabled'
root.destroy();print('UI smoke test passed: rendering, state gating, deadman, queue and stop.')
