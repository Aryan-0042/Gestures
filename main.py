import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os, sys

from cursor_control import CursorControl
from presentation_control import PresentationControl
from teaching_control import TeachingControl


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller bundles.
    """
    try:
        # PyInstaller stores temp path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


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
        for Page in (HomePage, CursorControlPage, PresentationControlPage, TeachingModePage):
            frame = Page(parent=self.container, controller=self)
            self.frames[Page.__name__] = frame
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

        tk.Label(self, text="Gesture Control App", font=("Helvetica", 28, "bold"), fg="white", bg="#1e1e1e").pack(pady=60)

        style = ttk.Style()
        style.configure("Home.TButton", font=("Helvetica", 14), padding=12)

        ttk.Button(self, text="Cursor Control", style="Home.TButton",
                   command=lambda: controller.show_frame("CursorControlPage")).pack(pady=15)

        ttk.Button(self, text="Presentation Control", style="Home.TButton",
                   command=lambda: controller.show_frame("PresentationControlPage")).pack(pady=15)

        ttk.Button(self, text="Teaching Mode", style="Home.TButton",
                   command=lambda: controller.show_frame("TeachingModePage")).pack(pady=15)

        self.about_button = ttk.Button(self, text="About", command=self.toggle_about)
        self.about_button.place(relx=0.98, rely=0.98, anchor="se")

        self.about_label = tk.Label(self,
            text="Gesture Control App\nDeveloped by: Aryan & Dhairya\nGitHub: github.com/Aryan-0042/Gestures",
            font=("Helvetica", 10), fg="white", bg="#1e1e1e", justify="right")
        self.about_visible = False

    def toggle_about(self):
        if self.about_visible:
            self.about_label.place_forget()
            self.about_visible = False
        else:
            self.about_label.place(relx=0.98, rely=0.9, anchor="se")
            self.about_visible = True


class ModePageBase(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller
        self.cap = None
        self.running = False
        self.overlay_enabled = True
        self.logic = None

        toolbar = tk.Frame(self, bg="#222222", height=45)
        toolbar.pack(side="top", fill="x")

        self.start_btn = tk.Button(toolbar, text="Start", command=self.toggle_camera,
                                   bg="#4caf50", fg="white", font=("Helvetica", 10, "bold"))
        self.start_btn.pack(side="left", padx=5, pady=6)

        tk.Button(toolbar, text="Toggle Overlay", command=self.toggle_overlay,
                  bg="#03a9f4", fg="white", font=("Helvetica", 10)).pack(side="left", padx=5, pady=6)

        self.info_btn = tk.Button(toolbar, text="Info", command=self.show_info_overlay,
                                  bg="#ff9800", fg="white", font=("Helvetica", 10))
        self.info_btn.pack(side="left", padx=5, pady=6)

        tk.Button(toolbar, text="Back", command=self.go_back,
                  bg="#e53935", fg="white", font=("Helvetica", 10)).pack(side="right", padx=5, pady=6)

        self.video_label = tk.Label(self, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        self.hint_label = tk.Label(self.video_label, text="▶ Press Start to Begin",
                                   font=("Helvetica", 16, "italic"),
                                   fg="#AAAAAA", bg="#000000")
        self.hint_label.place(relx=0.5, rely=0.5, anchor="center")

        self._build_info_overlay()

    def _build_info_overlay(self):
        """Construct a translucent, scrollable grid overlay (hidden by default)."""
        # Full‑screen overlay frame (semi‑transparent)
        self.info_overlay = tk.Frame(self, bg="#000000")
        try:
            self.info_overlay.tk.call(self.info_overlay, 'attributes', '-alpha', 0.8)
        except:
            pass
        self.info_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.info_overlay.lower()

        # Header bar
        header = tk.Frame(self.info_overlay, bg="#333333")
        header.pack(fill="x")
        tk.Label(header, text="Gesture Info", fg="white", bg="#333333",
                 font=("Helvetica", 14, "bold")).pack(side="left", padx=10, pady=6)
        tk.Button(header, text="✖", command=self.hide_info_overlay,
                  fg="white", bg="#333333", bd=0, font=("Helvetica", 14, "bold")).pack(side="right", padx=10)

        # Scrollable canvas + scrollbar
        self.info_canvas = tk.Canvas(self.info_overlay, bg="#1a1a1a", highlightthickness=0)
        self.info_scrollbar = tk.Scrollbar(self.info_overlay, orient="vertical", command=self.info_canvas.yview)
        self.info_canvas.configure(yscrollcommand=self.info_scrollbar.set)

        self.info_canvas.pack(side="left", fill="both", expand=True, padx=20, pady=(10,10))
        self.info_scrollbar.pack(side="right", fill="y", pady=(10,10))

        # Frame inside the canvas
        self.info_body = tk.Frame(self.info_canvas, bg="#1a1a1a")
        # Create window and save its ID
        self._canvas_window_id = self.info_canvas.create_window((0,0), window=self.info_body, anchor="nw")

        # When the body frame changes size, update scrollregion
        self.info_body.bind(
            "<Configure>",
            lambda e: self.info_canvas.configure(
                scrollregion=self.info_canvas.bbox("all")
            )
        )
        # When canvas width changes, update the inner window width
        self.info_canvas.bind(
            "<Configure>",
            lambda e: self.info_canvas.itemconfigure(
                self._canvas_window_id, width=e.width
            )
        )

        # Temporary storage for image refs
        self._info_images = []

    def show_info_overlay(self):
        """Populate and raise the info overlay grid."""
        # Clear old content
        for w in self.info_body.winfo_children():
            w.destroy()
        self._info_images.clear()

        # Get mode‑specific gestures (override in subclasses)
        gestures = self.get_gesture_info()  # returns list of (label, img_path)

        # Determine columns based on canvas width
        canvas_width = self.info_canvas.winfo_width() or self.winfo_width()
        max_cell = 200
        cols = max(1, canvas_width // max_cell)
        # Configure grid so columns expand evenly
        for c in range(cols):
            self.info_body.grid_columnconfigure(c, weight=1)

        # Place each gesture in the grid
        for idx, (label, img_path) in enumerate(gestures):
            r, c = divmod(idx, cols)
            cell = tk.Frame(self.info_body, bg="#1a1a1a", padx=5, pady=5)
            cell.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)

            # Load & resize image preserving aspect ratio
            try:
                full = resource_path(img_path)
                pil = Image.open(full)
                w, h = pil.size
                ratio = min(220/w, 220/h)
                resized = pil.resize((int(w*ratio), int(h*ratio)),Image.Resampling.LANCZOS)
                tkimg = ImageTk.PhotoImage(resized)

            except Exception as ex:
                print(f"Error loading {img_path}: {ex}")
                tkimg = None

            self._info_images.append(tkimg)

            if tkimg:
                tk.Label(cell, image=tkimg, bg="#1a1a1a").pack(pady=(0,5))
            tk.Label(cell, text=label.replace("_"," ").title(),
                     fg="white", bg="#1a1a1a",
                     font=("Helvetica", 10)).pack()

        # Finally lift the overlay above video
        self.info_overlay.lift()

    def hide_info_overlay(self):
        self.info_overlay.lower()

    def on_show(self):
        self.running = False
        self.start_btn.config(text="Start")
        self.hint_label.place(relx=0.5, rely=0.5, anchor="center")

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.start_btn.config(text="Stop")
        self.hint_label.place_forget()
        self.update_frame()

    def stop_camera(self):
        self.running = False
        self.start_btn.config(text="Start")
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        if self.logic and hasattr(self.logic, "stop"):
            self.logic.stop()
        self.hint_label.place(relx=0.5, rely=0.5, anchor="center")

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled

    def go_back(self):
        self.stop_camera()
        self.controller.show_frame("HomePage")

    def update_frame(self):
        if not (self.running and self.cap):
            return
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_frame)
            return
        w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if w > 0 and h > 0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        processed = self.process_mode_logic(frame)
        if processed is None:
            processed = frame
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tkimg = ImageTk.PhotoImage(img)
        self.video_label.imgtk = tkimg
        self.video_label.config(image=tkimg)
        self.after(10, self.update_frame)

    def process_mode_logic(self, frame):
        return frame
    

class CursorControlPage(ModePageBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.logic = CursorControl()

    def process_mode_logic(self, frame):
        self.logic.label_mapper = self.map_label_for_display
        return self.logic.process_frame(frame, overlay=self.overlay_enabled)

    def map_label_for_display(self, raw_label):
        mapping = {
            "cursor_move": "Cursor Move",
            "left_click": "Left Click",
            "right_click": "Right Click",
            "scroll_up": "Scroll Up",
            "scroll_down": "Scroll Down",
            "double_click": "Double Click",

            # Not used in this mode
            "next_slide": "",
            "prev_slide": "",
            "peace_sign": ""
        }
        return mapping.get(raw_label, raw_label.replace("_", " ").title())

    def get_gesture_info(self):
        return [
            ("Cursor Move", "images/cursor.jpeg"),
            ("Left Click", "images/left_click.jpeg"),
            ("Right Click", "images/right_click.jpeg"),
            ("Scroll Up", "images/scroll_up.jpg"),
            ("Scroll Down", "images/scroll_down.jpeg"),
            ("Double Click", "images/peace_sign.jpeg"),
        ]


class PresentationControlPage(ModePageBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.logic = PresentationControl()

    def process_mode_logic(self, frame):
        self.logic.label_mapper = self.map_label_for_display
        return self.logic.process_frame(frame, overlay=self.overlay_enabled)

    def map_label_for_display(self, raw_label):
        mapping = {
            "next_slide": "Next Slide",
            "prev_slide": "Previous Slide",
            "scroll_down": "Zoom In",
            "scroll_up": "Zoom Out",
            "double_click": "Start Slide Show",
            "peace_sign": "Start Slide Show",
            "left_click": "End Slide Show",

            # Not used in this mode
            "cursor_move": "",
            "right_click": ""
        }
        return mapping.get(raw_label, raw_label.replace("_", " ").title())
    
    def get_gesture_info(self):
        return [
            ("Next Slide", "images/next_slide.jpeg"),
            ("Previous Slide", "images/prev_slide.jpeg"),
            ("Zoom In", "images/scroll_down.jpeg"),
            ("Zoom Out", "images/scroll_up.jpg"),
            ("Start Slide Show", "images/peace_sign.jpeg"),
            ("End Slide Show", "images/left_click.jpeg"),
        ]

class TeachingModePage(ModePageBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.logic = TeachingControl()

    def process_mode_logic(self, frame):
        self.logic.label_mapper = self.map_label_for_display
        return self.logic.process_frame(frame, overlay=self.overlay_enabled)

    def map_label_for_display(self, raw_label):
        mapping = {
            "cursor_move": "Draw",
            "double_click": "Erase",
            "scroll_up": "Clear Canvas",
            "left_click": "Undo",
            "right_click": "Redo",

            # Not used in this mode
            "scroll_down": "",
            "next_slide": "",
            "prev_slide": ""
        }
        return mapping.get(raw_label, raw_label.replace("_", " ").title())
    
    def get_gesture_info(self):
        return [
            ("Draw", "images/cursor.jpeg"),
            ("Erase", "images/peace_sign.jpeg"),
            ("Clear Canvas", "images/scroll_up.jpg"),
            ("Undo", "images/left_click.jpeg"),
            ("Redo", "images/right_click.jpeg"),
        ]


if __name__ == "__main__":
    app = App()
    app.mainloop()
    