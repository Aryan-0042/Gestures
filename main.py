import tkinter as tk
import threading
import time
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

from cursor_control import CursorControl
from presentation_control import PresentationControl
from teaching_control import TeachingControl

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Mode Gesture Control (ANN)")
        self.master.geometry("800x600")
        self.master.configure(bg="#222222")

        # Instantiate modes
        self.cursor_mode = CursorControl()
        self.presentation_mode = PresentationControl()
        self.teaching_mode = TeachingControl()

        self.active_mode = None
        self.mode_thread = None

        # Create UI elements
        self.create_ui()

    def create_ui(self):
        # Title label
        title_label = tk.Label(
            self.master, text="Multi-Mode Gesture Control",
            fg="#FFFFFF", bg="#222222", font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)

        # Frame for mode buttons
        button_frame = tk.Frame(self.master, bg="#222222")
        button_frame.pack(pady=20)

        cursor_btn = tk.Button(
            button_frame, text="Cursor Mode", command=self.start_cursor_mode,
            fg="#000000", bg="#AAAAAA", font=("Arial", 12, "bold"), width=15
        )
        cursor_btn.grid(row=0, column=0, padx=10, pady=10)

        presentation_btn = tk.Button(
            button_frame, text="Presentation Mode", command=self.start_presentation_mode,
            fg="#000000", bg="#AAAAAA", font=("Arial", 12, "bold"), width=15
        )
        presentation_btn.grid(row=0, column=1, padx=10, pady=10)

        teaching_btn = tk.Button(
            button_frame, text="Teaching Mode", command=self.start_teaching_mode,
            fg="#000000", bg="#AAAAAA", font=("Arial", 12, "bold"), width=15
        )
        teaching_btn.grid(row=0, column=2, padx=10, pady=10)

        stop_btn = tk.Button(
            button_frame, text="Stop Mode", command=self.stop_current_mode,
            fg="#000000", bg="#FF5555", font=("Arial", 12, "bold"), width=15
        )
        stop_btn.grid(row=1, column=1, padx=10, pady=10)

        # Info label
        self.info_label = tk.Label(
            self.master,
            text="Usage:\n"
                 " - Cursor Mode: Move, Left/Right Click, Scroll.\n"
                 " - Presentation Mode: Next/Prev Slide, Zoom.\n"
                 " - Teaching Mode: Draw, Erase, Clear Canvas.\n",
            fg="#FFFFFF", bg="#222222", font=("Arial", 12)
        )
        self.info_label.pack()

        # Optional image
        try:
            img = Image.open("Gestures.jpg").resize((300, 200))
            self.img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.master, image=self.img_tk, bg="#222222")
            img_label.pack(pady=10)
        except:
            pass

    def start_cursor_mode(self):
        self.stop_current_mode()
        self.active_mode = "cursor"
        self.mode_thread = threading.Thread(target=self.cursor_mode.start)
        self.mode_thread.start()

    def start_presentation_mode(self):
        self.stop_current_mode()
        self.active_mode = "presentation"
        self.mode_thread = threading.Thread(target=self.presentation_mode.start)
        self.mode_thread.start()

    def start_teaching_mode(self):
        self.stop_current_mode()
        self.active_mode = "teaching"
        self.mode_thread = threading.Thread(target=self.teaching_mode.start)
        self.mode_thread.start()

    def stop_current_mode(self):
        if self.active_mode == "cursor":
            self.cursor_mode.stop()
        elif self.active_mode == "presentation":
            self.presentation_mode.stop()
        elif self.active_mode == "teaching":
            self.teaching_mode.stop()

        self.active_mode = None
        if self.mode_thread and self.mode_thread.is_alive():
            self.mode_thread.join()
            self.mode_thread = None

    def on_closing(self):
        """
        Graceful shutdown when the window is closed.
        """
        self.stop_current_mode()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
