"""Pre-deployment motion acceptance editor."""
import tkinter as tk
from tkinter import ttk
from .limits import FIELDS,DEFAULTS,validate


class LimitsDialog:
    def __init__(self,parent,values,on_apply,on_close):
        self.on_apply,self.on_close=on_apply,on_close
        self.window=tk.Toplevel(parent);self.window.title('Motion acceptance limits')
        self.window.transient(parent);self.window.grab_set();self.window.protocol('WM_DELETE_WINDOW',self.close)
        frame=ttk.Frame(self.window,padding=24);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='Motion acceptance limits',font=('DejaVu Sans',18,'bold')).grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,12))
        ttk.Label(frame,text='Edit before setup. Applying changes requires reloading the library.\nRuntime joint, heartbeat, damping and endpoint guards remain fixed.',wraplength=650).grid(row=1,column=0,columnspan=3,sticky='w',pady=(0,14))
        self.values={};self.entries={}
        for row,(key,(label,default,low,high)) in enumerate(FIELDS.items(),2):
            ttk.Label(frame,text=label).grid(row=row,column=0,sticky='w',padx=(0,16),pady=8)
            var=tk.StringVar(value=f'{values[key]:g}');self.values[key]=var
            entry=ttk.Entry(frame,textvariable=var,width=10);entry.grid(row=row,column=1,pady=8);self.entries[key]=entry
            ttk.Label(frame,text=f'{low:g}–{high:g}').grid(row=row,column=2,padx=12)
        self.message=tk.StringVar(value='A higher speed/travel/tilt limit permits more motion; clearance can only be tightened.')
        ttk.Label(frame,textvariable=self.message,wraplength=650).grid(row=8,column=0,columnspan=3,sticky='w',pady=14)
        buttons=ttk.Frame(frame);buttons.grid(row=9,column=0,columnspan=3,sticky='ew')
        ttk.Button(buttons,text='Restore defaults',command=self.defaults).pack(side='left')
        ttk.Button(buttons,text='Apply limits',style='Accent.TButton',command=self.apply).pack(side='right')
        ttk.Button(buttons,text='Cancel',command=self.close).pack(side='right',padx=8)
    def defaults(self):
        for key,var in self.values.items():var.set(f'{DEFAULTS[key]:g}')
    def apply(self):
        try:
            values=validate({key:var.get() for key,var in self.values.items()})
            self.on_apply(values)
        except (ValueError,OSError,RuntimeError) as e:self.message.set(str(e));return
        self.close()
    def close(self):
        self.window.grab_release();self.window.destroy();self.on_close()
