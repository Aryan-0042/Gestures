import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from cursor_control import CursorControl

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Control App")
        self.geometry("1000x700")
        self.configure(bg="#1e1e1e")
        self.resizable(True, True)

        self.container = tk.Frame(self, bg="#1e1e1e")
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (HomePage, CursorControlPage, PresentationControlPage, TeachingModePage):
            frame = F(parent=self.container, controller=self)
            self.frames[F.__name__] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e1e")
        self.controller = controller

        tk.Label(
            self,
            text="Gesture Control App",
            font=("Helvetica", 28, "bold"),
            fg="#ffffff",
            bg="#1e1e1e"
        ).pack(pady=60)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14), padding=10)

        ttk.Button(self, text="Cursor Control", command=lambda: controller.show_frame("CursorControlPage")).pack(pady=15)
        ttk.Button(self, text="Presentation Control", command=lambda: controller.show_frame("PresentationControlPage")).pack(pady=15)
        ttk.Button(self, text="Teaching Mode", command=lambda: controller.show_frame("TeachingModePage")).pack(pady=15)

class ModePageBase(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller

        self.cap = None
        self.running = False
        self.overlay_enabled = True

        # Toolbar
        toolbar = tk.Frame(self, bg="#222222", height=45)
        toolbar.pack(side="top", fill="x")

        self.start_btn = tk.Button(toolbar, text="Start", command=self.toggle_camera,
                                   bg="#4caf50", fg="white", font=("Helvetica", 10, "bold"))
        self.start_btn.pack(side="left", padx=5, pady=6)

        tk.Button(toolbar, text="Toggle Overlay", command=self.toggle_overlay,
                  bg="#03a9f4", fg="white", font=("Helvetica", 10)).pack(side="left", padx=5, pady=6)

        tk.Button(toolbar, text="Settings", command=self.open_settings,
                  bg="#673ab7", fg="white", font=("Helvetica", 10)).pack(side="left", padx=5, pady=6)

        tk.Button(toolbar, text="Back", command=self.go_back,
                  bg="#e53935", fg="white", font=("Helvetica", 10)).pack(side="right", padx=5, pady=6)

        self.video_label = tk.Label(self, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

    def on_show(self):
        self.running = False
        self.start_btn.config(text="Start")

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.start_btn.config(text="Stop")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        self.start_btn.config(text="Start")
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled

    def open_settings(self):
        print("Settings clicked (extend this if needed)")

    def go_back(self):
        self.stop_camera()
        self.controller.show_frame("HomePage")

    def update_frame(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_frame)
            return

        # Resize frame to fit the label
        width = self.video_label.winfo_width()
        height = self.video_label.winfo_height()
        if width > 0 and height > 0:
            frame = cv2.resize(frame, (width, height))

        processed = self.process_mode_logic(frame)

        if processed is None:
            processed = frame

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.after(10, self.update_frame)

    def process_mode_logic(self, frame):
        return frame  # To be overridden

class CursorControlPage(ModePageBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.cursor_control = CursorControl()

    def process_mode_logic(self, frame):
        return self.cursor_control.process_frame(frame, overlay=self.overlay_enabled)

class PresentationControlPage(ModePageBase):
    def process_mode_logic(self, frame):
        return frame  # Placeholder

class TeachingModePage(ModePageBase):
    def process_mode_logic(self, frame):
        return frame  # Placeholder

if __name__ == "__main__":
    app = App()
    app.mainloop()
