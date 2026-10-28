"""Live camera window, serviced by the runner while it maintains the controller lease."""
import io
import time

from PIL import Image


class CameraPreview:
    def __init__(self, backend):
        try:
            import tkinter as tk
            from PIL import ImageTk
        except ImportError:
            raise RuntimeError("Camera preview needs Tk: install python3-tk for this Python environment") from None
        self.tk, self.ImageTk = tk, ImageTk
        try:
            self.root = tk.Tk()
        except tk.TclError:
            raise RuntimeError("Cannot open camera preview; run from a graphical desktop with DISPLAY available") from None
        self.root.title(f"SONIC camera — {backend}")
        self.root.resizable(False, False)
        self.root.configure(background="#111827")
        self.status = tk.Label(self.root, text="Waiting for camera…", background="#111827",
                               foreground="#fbbf24", font=("sans", 12), padx=12, pady=10)
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(self.root, width=640, height=480, background="black", highlightthickness=0)
        self.canvas.pack()
        self.item = self.canvas.create_image(320, 240, anchor="center")
        tk.Label(self.root, text="Confirm setup steps with ENTER in the terminal. Keep this window open.",
                 background="#111827", foreground="white", padx=12, pady=12).pack(fill="x")
        self.closed = False
        self.frame = self.photo = None
        self.rendered = None
        self.last_draw = -float("inf")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update()

    @property
    def ready(self):
        return (not self.closed and self.rendered is not None
                and time.monotonic() - self.rendered <= .5
                and bool(self.root.winfo_viewable()))

    def update(self, frame=None, stopped=False):
        if self.closed:
            if stopped:
                return
            raise RuntimeError("Camera preview closed; ending decisions")
        try:
            now = time.monotonic()
            if frame is not None:
                self.frame = frame
            # Bound JPEG/Tk work; the main loop continues servicing the native lease.
            if self.frame is not None and self.frame.received != self.rendered and now-self.last_draw >= .1:
                with Image.open(io.BytesIO(self.frame.jpeg)) as image:
                    self.photo = self.ImageTk.PhotoImage(image.convert("RGB"), master=self.root)
                self.canvas.itemconfigure(self.item, image=self.photo)
                self.rendered = self.frame.received
                self.last_draw = now
            if stopped:
                self.status.configure(text="STOPPED — follow the terminal instructions", foreground="#f87171")
            elif self.frame is not None:
                age = max(0, now-self.frame.received)
                live = age <= .5
                self.status.configure(text=f"{'LIVE' if live else 'STALE'} · {self.frame.name} · {age:.2f}s old",
                                      foreground="#4ade80" if live else "#f87171")
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            raise RuntimeError("Camera preview unavailable; ending decisions") from None
        if self.closed and not stopped:
            raise RuntimeError("Camera preview closed; ending decisions")

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.root.destroy()
            except self.tk.TclError:
                pass
