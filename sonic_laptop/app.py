"""Local SONIC operator panel. --demo previews the UI without GPU or robot access."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk, messagebox
from .session import Session
from .limits import DEFAULTS, DEFAULT_PATH, load as load_limits, save as save_limits

BG='#f4f6f8'; CARD='#ffffff'; TEXT='#172b3a'; MUTED='#475b6b'; ACCENT='#087f6b'


class DemoSession:
    """UI-only state machine. Never creates a subprocess, file lease or DDS channel."""
    demo=True
    sim=True
    directory=Path('OFFLINE DEMO — no log files')
    process=None
    stopped=False
    pending=False
    status=None
    def __init__(self,catalog): self.catalog=catalog;self.previous=time.monotonic();self.seq=0;self.armed=False
    def arm(self):
        self.armed=True
        self.status=dict(seq=0,state=1,index=0,frame=0,playing=False,planner=False,fault=False,setup_ready=True,result='ok',reason='')
    def heartbeat(self): pass
    def stop(self):
        self.stopped=True
        if self.status:self.status.update(fault=True,reason='Demo stop latched')
    def terminate_supported(self):self.stop()
    def command(self,command,argument=0):
        if not self.status or self.stopped:raise RuntimeError('Arm the demo first')
        self.seq+=1;self.status.update(seq=self.seq,result='ok')
        if command=='grounded_start':self.status['state']=2
        elif command=='planner':self.status['planner']=True
        elif command=='motion':self.status.update(index=argument,frame=0,playing=True)
        return True
    def poll(self):
        now=time.monotonic();delta=now-self.previous;self.previous=now
        if self.status and self.status['playing'] and not self.stopped:
            self.status['frame']=min(self.catalog[self.status['index']]['frames']-1,self.status['frame']+max(1,int(delta*50)))
            if self.status['frame']==self.catalog[self.status['index']]['frames']-1:self.status['playing']=False
        return self.status


class Panel:
    def __init__(self,root,session,catalog):
        self.root,self.session,self.catalog=root,session,catalog
        self.limits=dict(getattr(session,'acceptance_limits',DEFAULTS))
        self.limits_path=getattr(session,'limits_path',DEFAULT_PATH)
        self.library_valid=getattr(session,'library_valid',True);self.limits_dialog=None
        self.idle_override=getattr(session,'idle_override',None)
        self.keys=set();self.blocked_keys=set();self.release_jobs={};self.queue=[];self.waiting_clip=None;self.idle_requested=False
        self.import_dialog=None;self.library_busy=False
        self.activity_lines=[];self.last_activity_read=0;self.setup_started=None
        self.notice=tk.StringVar(value='Import Kimodo exports or load a SONIC library to get started.')
        self.state_text=tk.StringVar(value='Disarmed');self.source_text=tk.StringVar(value='Reference motions')
        self.motion_text=tk.StringVar(value='No motion playing');self.queue_text=tk.StringVar(value='0 remaining')
        self.progress_text=tk.StringVar(value='Playback keeps SONIC active when the motion ends.')
        root.title('SONIC · Motion studio');root.geometry('1180x820');root.minsize(1000,740);root.configure(bg=BG)
        from tkinter import font
        available=set(font.families(root))
        family=next((f for f in ('DejaVu Sans','Arial','Liberation Sans') if f in available),'TkDefaultFont')
        self.font=family
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=(family,12),background=BG,foreground=TEXT)
        style.configure('TFrame',background=BG);style.configure('Card.TFrame',background=CARD)
        style.configure('TLabel',background=BG,foreground=TEXT)
        style.configure('Muted.TLabel',foreground=MUTED)
        style.configure('Card.TLabel',background=CARD,foreground=TEXT)
        style.configure('Caption.TLabel',background=CARD,foreground=MUTED,font=(family,11))
        style.configure('TButton',background='#e8edf1',foreground=TEXT,borderwidth=0,padding=(14,10),font=(family,12))
        style.map('TButton',background=[('active','#dce5eb'),('disabled','#f0f3f5')],foreground=[('disabled','#9aa7b2')])
        style.configure('Accent.TButton',background=ACCENT,foreground='white',font=(family,12,'bold'))
        style.map('Accent.TButton',background=[('active','#0b675b'),('disabled','#dceae6')],foreground=[('disabled','#879b96')])
        style.configure('TNotebook',background=BG,borderwidth=0)
        style.configure('TNotebook.Tab',padding=(20,12),background=BG,foreground=MUTED,borderwidth=0)
        style.map('TNotebook.Tab',background=[('selected',CARD)],foreground=[('selected',ACCENT)])
        style.configure('Horizontal.TProgressbar',background=ACCENT,troughcolor='#e8edf1',borderwidth=0)
        style.configure('Treeview',background=CARD,fieldbackground=CARD,foreground=TEXT,rowheight=34,borderwidth=0)
        style.configure('Treeview.Heading',background=BG,foreground=MUTED,font=(family,12,'bold'),padding=8)
        outer=ttk.Frame(root,padding=(28,24));outer.pack(fill='both',expand=True)
        header=ttk.Frame(outer);header.pack(fill='x')
        ttk.Label(header,text='SONIC',font=(family,23,'bold')).pack(side='left')
        ttk.Label(header,text='Motion studio',style='Muted.TLabel',font=(family,14)).pack(side='left',padx=16)
        offline=getattr(session,'offline',False)
        mode='OFFLINE LIBRARY' if offline else ('DEMO · NO ROBOT' if getattr(session,'demo',False) else ('SIMULATION' if session.sim else 'HARDWARE · '+session.interface))
        tk.Label(header,text=mode,bg='#e0eee9',fg=ACCENT,padx=12,pady=8,font=(family,11,'bold')).pack(side='right')
        ttk.Label(outer,text='G1 EDU U4  /  Dex3 fingers: no motion commands; damping on stop',style='Muted.TLabel').pack(anchor='w',pady=(5,20))
        status=ttk.Frame(outer,style='Card.TFrame',padding=(16,12));status.pack(fill='x',pady=(0,20))
        ttk.Label(status,textvariable=self.state_text,style='Card.TLabel',font=(family,11,'bold')).pack(side='left')
        ttk.Label(status,text='  /  ',style='Card.TLabel').pack(side='left')
        ttk.Label(status,textvariable=self.source_text,style='Card.TLabel').pack(side='left')
        ttk.Label(status,textvariable=self.queue_text,style='Caption.TLabel').pack(side='right')
        self.stop_button=tk.Button(outer,text='EMERGENCY STOP  ·  Esc',bg='#bb2636',activebackground='#951b28',
            fg='white',activeforeground='white',relief='flat',borderwidth=0,font=(family,18,'bold'),
            command=self.stop,pady=15,takefocus=True)
        self.stop_button.pack(side='bottom',fill='x',pady=(12,0))
        content=ttk.Frame(outer,height=360);content.pack(fill='both',expand=True);content.grid_propagate(False)
        content.columnconfigure(0,weight=1);content.columnconfigure(1,weight=0);content.rowconfigure(0,weight=1)
        tabs=ttk.Notebook(content);tabs.grid(row=0,column=0,sticky='nsew',padx=(0,20));self.tabs=tabs
        clips=ttk.Frame(tabs,style='Card.TFrame',padding=18);keyboard=ttk.Frame(tabs,style='Card.TFrame',padding=18)
        tabs.add(clips,text='Motions');tabs.add(keyboard,text='Keyboard')
        activity=ttk.Frame(tabs,style='Card.TFrame',padding=12);tabs.add(activity,text='Live activity')
        from tkinter.scrolledtext import ScrolledText
        self.activity=ScrolledText(activity,font=(family,11),bg=CARD,fg=TEXT,relief='flat',wrap='word',height=8,state='disabled')
        self.activity.pack(fill='both',expand=True)
        toolbar=ttk.Frame(clips,style='Card.TFrame');toolbar.pack(fill='x',pady=(0,14))
        self.import_button=ttk.Button(toolbar,text='+ Import Kimodo',style='Accent.TButton',command=self.open_import);self.import_button.pack(side='left')
        self.load_button=ttk.Button(toolbar,text='Load library',command=self.load_library);self.load_button.pack(side='left',padx=8)
        self.idle_menu=ttk.Menubutton(toolbar,text='Idle: default ▾')
        idle_options=tk.Menu(self.idle_menu,tearoff=False)
        idle_options.add_command(label='Use bundled Kimodo idle',command=lambda:self.select_idle(None))
        idle_options.add_command(label='Choose custom SONIC idle…',command=self.choose_idle)
        self.idle_menu.configure(menu=idle_options);self.idle_menu.pack(side='left',padx=8)
        if self.idle_override:self.idle_menu.configure(text='Idle: '+Path(self.idle_override).name+' ▾')
        self.library_label=tk.StringVar(value=f'{max(0,len(catalog)-1)} motions')
        ttk.Label(clips,textvariable=self.library_label,style='Caption.TLabel').pack(anchor='w',pady=(0,8))
        listing=ttk.Frame(clips,style='Card.TFrame');listing.pack(fill='both',expand=True)
        self.listbox=tk.Listbox(listing,font=(family,12),bg=CARD,fg=TEXT,selectbackground='#e0eee9',selectforeground='#13594d',
                               relief='flat',highlightthickness=0,exportselection=False,height=6,activestyle='none',selectborderwidth=4)
        scroll=ttk.Scrollbar(listing,orient='vertical',command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');self.listbox.pack(side='left',fill='both',expand=True)
        self.refresh_library()
        controls=ttk.Frame(clips,style='Card.TFrame');controls.pack(fill='x',pady=(14,0))
        self.play_button=ttk.Button(controls,text='Play once',style='Accent.TButton',command=self.play);self.play_button.pack(side='left')
        self.all_button=ttk.Button(controls,text='Play all',command=self.play_all);self.all_button.pack(side='left',padx=8)
        self.clear_button=ttk.Button(controls,text='Clear queue',command=self.clear_queue);self.clear_button.pack(side='right')
        self.presets=[]
        self.planner_button=ttk.Button(keyboard,text='Enable keyboard control',style='Accent.TButton',command=self.enable_keyboard)
        self.planner_button.pack(anchor='w')
        ttk.Label(keyboard,text='Slow-walk planner · 0.15 m/s request',style='Caption.TLabel').pack(anchor='w',pady=(8,18))
        pad=ttk.Frame(keyboard,style='Card.TFrame');pad.pack(pady=(0,12))
        for label,row,col in [('W',0,1),('A',1,0),('S',1,1),('D',1,2)]:
            tk.Label(pad,text=label,bg='#edf2f5',fg=TEXT,font=(family,16,'bold'),width=5,height=2).grid(row=row,column=col,padx=4,pady=4)
        tk.Label(pad,text='Hold SPACE + direction',bg='#e0eee9',fg=ACCENT,font=(family,11),pady=12).grid(row=2,column=0,columnspan=3,sticky='ew',pady=8)
        ttk.Label(keyboard,text='Release a key or leave the window to request idle.\nEsc requests damping. Return to motions with a supported restart.',
                  style='Caption.TLabel',wraplength=490).pack(anchor='w')
        side=ttk.Frame(content,style='Card.TFrame',padding=20);side.grid(row=0,column=1,sticky='nsew');side.configure(width=260)
        ttk.Label(side,text='Robot control',style='Card.TLabel',font=(family,13,'bold')).pack(anchor='w',pady=(0,8))
        ttk.Label(side,text='Set up suspended, then lower.\nPlayback requires feet down.',style='Caption.TLabel').pack(anchor='w',pady=(0,20))
        self.arm_button=ttk.Button(side,text='1  Suspended setup',command=self.arm);self.arm_button.pack(fill='x')
        self.start_button=ttk.Button(side,text='2  Feet down / start',style='Accent.TButton',command=self.start_grounded);self.start_button.pack(fill='x',pady=(10,0))
        ttk.Label(side,text='Setup actuates and holds the body.\nWait for checks, then lower feet.',style='Caption.TLabel').pack(anchor='w',pady=16)
        spacer=ttk.Frame(side,style='Card.TFrame');spacer.pack(fill='both',expand=True)
        ttk.Label(side,text='Emergency stop requests damping\nfor body + both hands.\nKeep the robot supported.',style='Caption.TLabel').pack(anchor='w',pady=(10,0))
        progress=ttk.Frame(outer);progress.pack(fill='x',pady=(18,0))
        self.progress=ttk.Progressbar(progress,maximum=100);self.progress.pack(fill='x')
        ttk.Label(progress,textvariable=self.progress_text,style='Muted.TLabel',wraplength=990).pack(anchor='w',pady=(6,0))
        ttk.Label(outer,textvariable=self.notice,wraplength=990,foreground=ACCENT).pack(fill='x',pady=(10,12))
        footer=ttk.Frame(outer);footer.pack(fill='x')
        self.logs_button=ttk.Button(footer,text='Session files',command=self.open_logs);self.logs_button.pack(side='left')
        self.limits_button=ttk.Button(footer,text='Acceptance limits',command=self.open_limits);self.limits_button.pack(side='left',padx=8)
        self.limits_text=tk.StringVar();self.update_limits_text()
        ttk.Label(footer,textvariable=self.limits_text,style='Muted.TLabel').pack(side='left',padx=8)
        ttk.Button(footer,text='Close',command=self.close).pack(side='right')
        root.bind_class('SonicSafetyKeys','<KeyPress>',self.key_down)
        root.bind_class('SonicSafetyKeys','<KeyRelease>',self.key_up)
        def key_priority(widget):
            widget.bindtags(('SonicSafetyKeys',*widget.bindtags()))
            for child in widget.winfo_children():key_priority(child)
        key_priority(root)
        root.bind('<FocusOut>',self.focus_out);root.protocol('WM_DELETE_WINDOW',self.close)
        self.update_buttons(None);root.after(50,self.tick)

    def update_limits_text(self):
        self.limits_text.set(f"Reference ≤{self.limits['reference_speed']:g} rad/s · travel ≤{self.limits['root_displacement']:g} m")

    def open_limits(self):
        if not self.can_edit_library() or self.import_dialog or self.limits_dialog:return
        from .limits_dialog import LimitsDialog
        def done():self.limits_dialog=None;self.update_buttons(self.session.status)
        self.limits_dialog=LimitsDialog(self.root,self.limits,self.apply_limits,done)
        self.update_buttons(self.session.status)

    def apply_limits(self,values):
        if not self.can_edit_library() or self.import_dialog:raise RuntimeError('Limits are locked during setup/control or import.')
        values=save_limits(self.limits_path,values)
        if values!=self.limits:
            self.limits=values;self.session.acceptance_limits=dict(values);self.library_valid=False
            self.queue.clear();self.waiting_clip=None;self.update_limits_text()
            self.notice.set('Limits saved. Reload the motion library to validate it with these values before setup.')
        self.update_buttons(self.session.status)

    def refresh_library(self):
        self.listbox.delete(0,'end')
        for item in self.catalog[1:]:self.listbox.insert('end',f'{item["name"]}    ·    {item["frames"]/50:.1f} s')
        if len(self.catalog)>1:self.listbox.selection_set(0)
        self.library_label.set(f'{max(0,len(self.catalog)-1)} motions')

    def can_edit_library(self):
        return not (self.session.process or getattr(self.session,'armed',False) or self.session.stopped or self.library_busy)

    def open_import(self):
        if not self.can_edit_library() or self.import_dialog or self.limits_dialog:return
        from .import_dialog import ImportDialog
        def done():self.import_dialog=None;self.update_buttons(self.session.status)
        self.import_dialog=ImportDialog(self.root,done,self.load_library,dict(self.limits))
        self.update_buttons(self.session.status)

    def select_idle(self,path):
        if not self.can_edit_library() or self.import_dialog or self.limits_dialog:return
        self.idle_override=Path(path) if path is not None else None
        self.session.idle_override=self.idle_override;self.library_valid=False
        self.idle_menu.configure(text='Idle: '+(self.idle_override.name if self.idle_override else 'default')+' ▾')
        self.load_library(default_only=True)

    def choose_idle(self):
        if not self.can_edit_library() or self.import_dialog or self.limits_dialog:return
        from tkinter import filedialog
        path=filedialog.askdirectory(parent=self.root,title='Custom SONIC idle (contains joint_pos.csv)')
        if path:self.select_idle(path)

    def load_library(self,folder=None,*,default_only=False):
        if not self.can_edit_library() or self.import_dialog or self.limits_dialog:return
        from tkinter import filedialog
        if not default_only:
            folder=folder or filedialog.askdirectory(parent=self.root,title='SONIC motion library')
            if not folder:return
        import queue,threading
        events=queue.Queue();self.library_busy=True;self.update_buttons(self.session.status)
        self.notice.set('Checking library, endpoints and hand clearance…')
        def worker():
            try:
                from .model import load_model
                from .motions import stage_library
                runs=Path('artifacts/laptop/sessions').resolve();runs.mkdir(parents=True,exist_ok=True)
                directory=Path(tempfile.mkdtemp(prefix='session-',dir=runs));directory.chmod(0o700)
                model=(load_model(Path(__file__).resolve().parents[2]/'GR00T-WholeBodyControl') if getattr(self.session,'demo',False)
                       else json.loads((self.session.build/'guard-model.json').read_text()))
                from .idle import resolve_idle
                upstream=Path(__file__).resolve().parents[2]/'GR00T-WholeBodyControl'
                idle=resolve_idle(self.idle_override,directory,upstream,self.limits)
                catalog=stage_library(idle,folder,directory/'motions',model,self.limits)
                events.put((catalog,directory,None))
            except Exception as e:events.put((None,None,str(e)))
        threading.Thread(target=worker,daemon=True).start()
        def poll():
            try:catalog,directory,error=events.get_nowait()
            except queue.Empty:self.root.after(80,poll);return
            self.library_busy=False
            if error:self.notice.set('Library rejected: '+error)
            else:
                self.library_valid=True;self.catalog=catalog;self.session.directory=directory
                if getattr(self.session,'demo',False):self.session.catalog=catalog
                self.refresh_library();self.notice.set('Library loaded and snapshotted. '+('Offline only; launch simulation separately to test SONIC.' if getattr(self.session,'offline',False) else 'Review setup before arming.'))
            self.update_buttons(self.session.status)
        self.root.after(80,poll)

    def open_logs(self):
        if getattr(self.session,'demo',False):self.notice.set('Demo mode creates no logs.');return
        import subprocess
        subprocess.Popen(['xdg-open',str(self.session.directory)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def send(self,command,argument=0):
        try:return self.session.command(command,argument)
        except Exception as e:self.notice.set(str(e));return False

    def arm(self):
        if not self.library_valid or not self.can_edit_library() or self.import_dialog or self.limits_dialog or getattr(self.session,'offline',False):return
        if not getattr(self.session,'demo',False) and not messagebox.askokcancel('ARM starts motor control',
            'After model loading, ARM releases Unitree control and ramps to your idle pose.\n\n'
            'Confirm: gantry support, clear workspace and hands, physical stop operator, and successful simulation validation.'):
            return
        try:
            self.session.arm();self.setup_started=time.monotonic();self.tabs.select(2)
            self.notice.set('Setup running. Live activity shows model loading and robot checks.');self.update_buttons(self.session.status)
        except Exception as e:self.notice.set('Launch failed: '+str(e))

    def start_grounded(self):
        s=self.session.status
        if not s or s['state']!=1 or not s.get('setup_ready') or self.session.stopped:return
        if not getattr(self.session,'demo',False) and not messagebox.askokcancel('Confirm feet on floor',
            'Lower the robot until both feet are planted and it can bear weight. Keep the gantry as fall protection.\n\n'
            'The app checks joint settling, not foot contact or balance. Confirm feet down before starting SONIC.'):
            return
        self.keys.clear();self.send('grounded_start')

    def enable_keyboard(self):
        self.blocked_keys.update(self.keys);self.keys.clear();self.idle_requested=True
        self.send('planner')

    def request_idle(self):
        self.idle_requested=True
        if self.session.status and self.session.status['planner'] and not self.session.stopped:
            self.send('idle') # Supersedes a pending movement request.

    def enqueue(self,indices):
        s=self.session.status
        if not s or s['state']!=2 or s['planner'] or s['playing'] or self.session.stopped or self.waiting_clip is not None or self.queue:
            self.notice.set('Wait for active reference hold and an empty playlist.');return
        self.queue=list(indices);self.notice.set(f'Queued {len(indices)} clip(s); each will play once.')

    def play(self):
        selected=self.listbox.curselection()
        if selected:self.enqueue([selected[0]+1])
    def play_all(self):self.enqueue([m['index'] for m in self.catalog[1:]])
    def clear_queue(self):self.queue.clear();self.notice.set('Queue cleared. The current clip will finish and hold.')
    def stop(self):
        self.queue.clear();self.keys.clear();self.session.stop();self.state_text.set('STOP LATCHED');self.notice.set('Body + hands damping requested. Keep supported; verify the robot response. No automatic restart.' if self.session.process or getattr(self.session,'demo',False) else 'Stop latched. Controller is not running; no robot commands sent. Use the physical stop if needed.');self.update_buttons(self.session.status)

    def key_down(self,event):
        key=event.keysym.lower()
        if key=='escape':self.stop();return 'break'
        if key in self.release_jobs:self.root.after_cancel(self.release_jobs.pop(key))
        if key in self.blocked_keys:return 'break'
        self.keys.add(key)
        if key in ('space','w','a','s','d'):return 'break' # Space must not activate the focused ARM button.

    def key_up(self,event):
        key=event.keysym.lower()
        def release():
            self.release_jobs.pop(key,None);self.keys.discard(key);self.blocked_keys.discard(key)
            if key in ('space','w','a','s','d'):self.request_idle()
        if key not in self.release_jobs:self.release_jobs[key]=self.root.after(25,release)
        if key in ('space','w','a','s','d'):return 'break'

    def focus_out(self,event):self.root.after_idle(self.check_focus)
    def check_focus(self):
        if self.root.focus_displayof() is None:self.keys.clear();self.request_idle()

    def update_buttons(self,s):
        armed=bool(self.session.process) or getattr(self.session,'armed',False)
        ready=bool(s and s['state']==2 and not s['planner'] and not s['playing'] and not self.queue and self.waiting_clip is None and not self.session.pending and not self.session.stopped)
        def enable(widget,yes):widget.configure(state='normal' if yes else 'disabled')
        editable=self.can_edit_library() and self.import_dialog is None and self.limits_dialog is None
        enable(self.limits_button,editable)
        enable(self.import_button,editable);enable(self.load_button,editable);enable(self.idle_menu,editable)
        enable(self.arm_button,editable and self.library_valid and not getattr(self.session,'offline',False))
        enable(self.start_button,bool(s and s['state']==1 and s.get('setup_ready',False) and not self.session.stopped and not self.session.pending))
        enable(self.planner_button,ready)
        for widget in [self.play_button,self.all_button,*self.presets]:enable(widget,ready and len(self.catalog)>1)
        enable(self.clear_button,bool(self.queue));enable(self.logs_button,not getattr(self.session,'demo',False))

    def update_activity(self):
        if time.monotonic()-self.last_activity_read<.25:return
        self.last_activity_read=time.monotonic()
        lines=self.session.read_activity() if hasattr(self.session,'read_activity') else []
        if lines:
            self.activity_lines=(self.activity_lines+lines)[-200:]
            self.activity.configure(state='normal');self.activity.delete('1.0','end')
            self.activity.insert('end','\n'.join(self.activity_lines));self.activity.see('end');self.activity.configure(state='disabled')

    def tick(self):
        try:
            self.session.heartbeat();s=self.session.poll();self.update_activity()
            if self.session.process and self.session.process.poll() is not None:
                self.session.stop();self.state_text.set('PROCESS EXITED');self.notice.set('Support robot / physical stop. Inspect controller.log.')
            elif s:
                label=['INITIALIZING','CHECKS PASSED · LOWER FEET' if s.get('setup_ready') else 'SETTLING · KEEP SUPPORTED','SONIC ACTIVE'][s['state']]
                item=self.catalog[s['index']]
                self.state_text.set('STOP LATCHED' if self.session.stopped else label)
                self.source_text.set('Planner · slow walk' if s['planner'] else ('Clip playback' if s['playing'] else 'Reference hold'))
                self.motion_text.set('Planner' if s['planner'] else item['name'][:25])
                self.progress['value']=0 if s['planner'] else 100*s['frame']/max(1,item['frames']-1)
                self.progress_text.set('Planner active • hold Space + direction to move' if s['planner'] else f'{item["name"]}  ·  frame {s["frame"]+1} / {item["frames"]}  ·  '+('playing' if s['playing'] else 'active reference hold'))
                if s['state']<2:
                    self.source_text.set('Setup · joint pose hold' if s['state']==1 else 'Setup · pose initialization')
                    self.progress['value']=100 if s.get('setup_ready') else 0
                    self.progress_text.set(s.get('detail') or 'Waiting for fresh, settled joint readings. Keep the gantry supporting the robot.')
                elif s['planner']:
                    self.source_text.set('Keyboard · hold Space + WASD')
                else:
                    self.source_text.set(('Playing · ' if s['playing'] else 'Holding · ')+item['name'][:34])
                if s['fault']:self.notice.set('DAMPING LATCHED: '+s['reason'])
                elif s['result']!='ok' and s['result']!='ready':self.notice.set(s['result'])
                if not self.session.stopped and not self.session.pending:
                    if s['planner']:
                        direction=next((k for k in ('w','s','a','d') if k in self.keys),None)
                        if self.idle_requested or 'space' not in self.keys or not direction:
                            self.session.command('idle');self.idle_requested=False
                        else:self.session.command('move',('w','s','a','d').index(direction))
                    elif self.waiting_clip is not None:
                        if s['result']!='ok':self.queue.clear();self.waiting_clip=None
                        elif s['index']==self.waiting_clip and not s['playing'] and s['frame']==self.catalog[self.waiting_clip]['frames']-1:
                            self.waiting_clip=None;self.notice.set('Clip complete. SONIC continues active reference hold.')
                    elif self.queue and not s['playing']:
                        index=self.queue.pop(0)
                        if self.session.command('motion',index):self.waiting_clip=index
            elif self.session.process:
                self.state_text.set('SETUP RUNNING');self.source_text.set('Loading SONIC')
                elapsed=int(time.monotonic()-(self.setup_started or time.monotonic()))
                self.progress_text.set(f'{getattr(self.session,"setup_phase","Starting controller")} · {elapsed}s elapsed. See Live activity.')
                if self.session.stopped:
                    self.state_text.set('STOP REQUESTED')
                    self.notice.set('Setup cancelled. No damping acknowledgement yet; use physical stop if needed.')
            self.queue_text.set(f'{len(self.queue)} remaining')
            self.update_buttons(s)
        except Exception as e:
            self.stop();self.notice.set('APP FAULT → damping: '+str(e))
        self.root.after(50,self.tick)

    def close(self):
        if self.import_dialog or self.library_busy or self.limits_dialog:
            self.notice.set('Finish importing/loading before closing.');return
        if self.session.process:
            self.stop()
            if not messagebox.askokcancel('Support robot before closing','Damping is latched. Is the robot fully supported so the controller process can exit?'):return
            self.session.terminate_supported()
        self.root.destroy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',action='store_true',help='Offline library/import workspace; cannot arm a robot')
    p.add_argument('--demo',action='store_true',help='Offline UI demo: no GPU, DDS or robot commands')
    p.add_argument('--build',type=Path,default=Path('artifacts/laptop/deploy'))
    p.add_argument('--idle',type=Path,help='Override bundled Kimodo idle with a stationary SONIC directory')
    p.add_argument('--motions',type=Path,help='Folder of SONIC CSV motion directories')
    p.add_argument('--interface',help='LOCAL Ethernet interface; lo for simulation')
    p.add_argument('--sim',action='store_true')
    p.add_argument('--limits',type=Path,help='Motion acceptance JSON; defaults to saved app settings')
    args=p.parse_args()
    limits=load_limits(args.limits)
    if args.demo or args.library:
        catalog=[dict(index=i,name=name,frames=frames) for i,(name,frames) in enumerate([
            ('Standing reference',50),('Small wave · demo',150),('Arm raise · demo',200),('Return to neutral · demo',100),('Gesture · demo',175)])]
        if args.library:catalog=[dict(index=0,name='Idle reference',frames=50)]
        session=DemoSession(catalog);session.offline=args.library
        session.acceptance_limits=limits;session.limits_path=args.limits or DEFAULT_PATH
        session.idle_override=args.idle
        if args.library:session.library_valid=False
        root=tk.Tk();panel=Panel(root,session,catalog)
        if args.library:root.after(0,lambda:panel.load_library(default_only=True))
        root.mainloop();return
    if not args.interface:p.error('--interface is required for the GPU controller')
    from .motions import stage_library
    import fcntl,os
    lock=Path(f'/tmp/sonic-laptop-{os.getuid()}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runs=Path('artifacts/laptop/sessions').resolve();runs.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='session-',dir=runs));directory.chmod(0o700)
    model=json.loads((args.build/'guard-model.json').read_text())
    catalog=[dict(index=0,name='Default idle',frames=50)]
    session=Session(args.build,directory,args.interface,args.sim,lock.fileno())
    session.acceptance_limits=limits;session.limits_path=args.limits or DEFAULT_PATH
    session.library_valid=False;session.idle_override=args.idle
    root=tk.Tk();panel=Panel(root,session,catalog)
    root.after(0,lambda:panel.load_library(args.motions,default_only=not bool(args.motions)))
    root.mainloop()

if __name__=='__main__':main()
