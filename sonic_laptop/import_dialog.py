"""Offline, threaded Kimodo import dialog. Workers never touch Tk or robot sessions."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

ROOT=Path(__file__).resolve().parents[1]


class ImportDialog:
    def __init__(self,parent,on_close,on_library,limits=None):
        self.limits=limits
        self.on_close,self.on_library=on_close,on_library
        self.sources=[];self.events=queue.Queue();self.running=False;self.output_path=None
        self.window=tk.Toplevel(parent);self.window.title('Import Kimodo motions')
        self.window.geometry('780x610');self.window.minsize(740,580);self.window.transient(parent)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        frame=ttk.Frame(self.window,padding=24);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='Import Kimodo',font=('DejaVu Sans',20,'bold')).pack(anchor='w')
        ttk.Label(frame,text='Convert G1 exports to SONIC. No robot connection required.',style='Muted.TLabel').pack(anchor='w',pady=(4,18))
        row=ttk.Frame(frame);row.pack(fill='x')
        self.file_button=ttk.Button(row,text='Choose files',command=self.choose_files);self.file_button.pack(side='left')
        self.folder_button=ttk.Button(row,text='Choose folder',command=self.choose_folder);self.folder_button.pack(side='left',padx=8)
        self.selected=tk.StringVar(value='No files selected')
        ttk.Label(frame,textvariable=self.selected,wraplength=700,style='Muted.TLabel').pack(anchor='w',pady=(8,14))
        form=ttk.Frame(frame);form.pack(fill='x');form.columnconfigure(1,weight=1)
        self.fps=tk.StringVar();self.output=tk.StringVar();self.python=tk.StringVar()
        self.entries=[];self.browse_buttons=[]
        for i,(label,var,action) in enumerate([
            ('Source FPS',self.fps,None),('New output folder',self.output,self.choose_output),
            ('Kimodo Python (NPZ only)',self.python,self.choose_python)]):
            ttk.Label(form,text=label).grid(row=i,column=0,sticky='w',padx=(0,12),pady=7)
            entry=ttk.Entry(form,textvariable=var);entry.grid(row=i,column=1,sticky='ew',pady=7);self.entries.append(entry)
            if action:
                b=ttk.Button(form,text='Browse',command=action);b.grid(row=i,column=2,padx=(8,0));self.browse_buttons.append(b)
        ttk.Label(frame,text='Use the FPS shown by Kimodo. For cluster-generated motions, CSV is the easiest local import.',
                  style='Muted.TLabel',wraplength=700).pack(anchor='w',pady=10)
        self.results=ttk.Treeview(frame,columns=('motion','result'),show='headings',height=4)
        self.results.heading('motion',text='Motion');self.results.heading('result',text='Import result')
        self.results.column('motion',width=190);self.results.column('result',width=460)
        self.results.pack(fill='both',expand=True)
        self.results.bind('<<TreeviewSelect>>',self.show_detail)
        self.details={};self.message=tk.StringVar(value='Converted clips still need idle/transition checks and simulation testing.')
        ttk.Label(frame,textvariable=self.message,wraplength=700,style='Muted.TLabel').pack(anchor='w',pady=12)
        bottom=ttk.Frame(frame);bottom.pack(fill='x')
        self.convert_button=ttk.Button(bottom,text='Import motions',style='Accent.TButton',command=self.start);self.convert_button.pack(side='left')
        self.use_button=ttk.Button(bottom,text='Load output library',command=self.use,state='disabled');self.use_button.pack(side='left',padx=8)
        self.close_button=ttk.Button(bottom,text='Done',command=self.close);self.close_button.pack(side='right')
        self.window.grab_set();self.poll_id=self.window.after(80,self.poll)

    def choose_files(self):
        paths=filedialog.askopenfilenames(parent=self.window,title='Kimodo-G1 exports',filetypes=[('Kimodo G1','*.csv *.npz')])
        if paths:self.set_sources([Path(p) for p in paths])
    def choose_folder(self):
        folder=filedialog.askdirectory(parent=self.window,title='Folder of Kimodo exports')
        if folder:self.set_sources(sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ('.csv','.npz')))
    def set_sources(self,sources):
        self.sources=sources;self.selected.set(f'{len(sources)} file(s) selected'+(' • '+sources[0].name if sources else ''))
    def choose_output(self):
        parent=filedialog.askdirectory(parent=self.window,title='Choose parent folder for converted motions')
        if parent:self.output.set(str(Path(parent)/'sonic_import'))
    def choose_python(self):
        path=filedialog.askopenfilename(parent=self.window,title='LOCAL Python executable with Kimodo installed')
        if path:self.python.set(path)
    def show_detail(self,event=None):
        selected=self.results.selection()
        if selected:self.message.set(self.details.get(selected[0],''))
    def start(self):
        import math
        try:
            fps=float(self.fps.get())
            if not math.isfinite(fps) or fps<=0:raise ValueError('Enter the actual positive source FPS.')
            if not self.sources:raise ValueError('Choose at least one CSV or NPZ file.')
            if len({p.stem for p in self.sources})!=len(self.sources):raise ValueError('Duplicate names: choose either CSV or NPZ for each motion.')
            if not self.output.get().strip():raise ValueError('Choose a new output folder.')
            output=Path(self.output.get()).expanduser().resolve()
            if output.exists():raise ValueError('Output already exists. Choose a new folder; existing files are never overwritten.')
            python=Path(self.python.get()).expanduser() if self.python.get().strip() else None
        except ValueError as e:self.message.set(str(e));return
        self.output_path=output;self.running=True
        for item in self.results.get_children():self.results.delete(item)
        for w in [self.convert_button,self.file_button,self.folder_button,self.close_button,*self.entries,*self.browse_buttons]:w.configure(state='disabled')
        self.use_button.configure(state='disabled');self.message.set('Converting and checking motions…')
        sources=list(self.sources)
        def worker():
            # All expensive work is off the Tk event loop.
            try:
                from .import_kimodo import convert
                for source in sources:
                    try:
                        report=convert(source,output/source.stem,fps,ROOT.parent/'GR00T-WholeBodyControl',python,self.limits)
                        self.events.put(('result',source.stem,report,None))
                    except Exception as e:self.events.put(('result',source.stem,None,str(e)))
            except Exception as e:self.events.put(('error',str(e)))
            finally:self.events.put(('done',))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            while True:
                event=self.events.get_nowait()
                if event[0]=='result':
                    _,name,report,error=event
                    result='Import failed' if error else ('Checks passed' if report['commissioning_checks_passed'] else 'Needs review')
                    detail=error or report.get('commissioning_error') or 'Per-clip checks passed. Idle matching and simulation are still required.'
                    iid=self.results.insert('', 'end',values=(name,result));self.details[iid]=detail
                elif event[0]=='error':self.message.set(event[1])
                elif event[0]=='done':
                    self.running=False
                    for w in [self.convert_button,self.file_button,self.folder_button,self.close_button,*self.entries,*self.browse_buttons]:w.configure(state='normal')
                    self.use_button.configure(state='normal' if self.output_path.exists() else 'disabled')
                    if self.results.get_children():self.message.set('Import complete. Select a row for details. Reports are saved beside each converted clip.')
        except queue.Empty:pass
        if self.window.winfo_exists():self.poll_id=self.window.after(80,self.poll)
    def use(self):
        output=self.output_path;self.close();self.on_library(output)
    def close(self):
        if self.running:self.message.set('Wait for the current offline import to finish.');return
        self.window.after_cancel(self.poll_id);self.window.grab_release();self.window.destroy();self.on_close()
