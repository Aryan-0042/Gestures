import tkinter as tk
from tkinter import Label, Button
import threading

from cursor_control import CursorControl
from presentation_control import PresentationControl
from teaching_control import TeachingControl

class SciFiApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Mode Gesture Control")
        self.master.geometry("800x500")
        self.master.configure(bg="#121212")  # Matte black background


        # Instantiate modes
        self.cursor_mode = CursorControl()
        self.presentation_mode = PresentationControl()
        self.teaching_mode = TeachingControl()
        self.active_mode = None
        self.mode_thread = None

        # Title Label
        title_label = Label(
            master, text="Multi-Mode Gesture Control", fg="#00FFFF", bg="#121212",
            font=("Arial", 20, "bold"))
        title_label.pack(pady=20)

        # Button Frame
        button_frame = tk.Frame(master, bg="#121212")
        button_frame.pack(pady=20)

        # Glowing Button Styles
        button_style = {
            "fg": "#000000", "font": ("Arial", 14, "bold"), "width": 18,
            "relief": "flat", "bd": 5
        }

        self.cursor_btn = Button(button_frame, text="🖱 Cursor Mode", bg="#00FFFF", **button_style, command=self.start_cursor_mode)
        self.cursor_btn.grid(row=0, column=0, padx=15, pady=10)

        self.presentation_btn = Button(button_frame, text="📽 Presentation Mode", bg="#FF00FF", **button_style, command=self.start_presentation_mode)
        self.presentation_btn.grid(row=0, column=1, padx=15, pady=10)

        self.teaching_btn = Button(button_frame, text="🎨 Teaching Mode", bg="#FFD700", **button_style, command=self.start_teaching_mode)
        self.teaching_btn.grid(row=0, column=2, padx=15, pady=10)

        # STOP Button
        self.stop_btn = Button(
            master, text="🔴 STOP", fg="#FFFFFF", bg="#FF0000", font=("Arial", 14, "bold"), width=10,
            relief="flat", bd=5, command=self.stop_current_mode)
        self.stop_btn.pack(pady=20)

        # Status Label
        self.status_label = Label(
            master, text="Status: Idle", fg="#00FF00", bg="#121212", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=10)

    def start_mode(self, mode_name, mode_instance):
        self.stop_current_mode()
        self.active_mode = mode_instance
        self.status_label.config(text=f"Status: {mode_name} Mode Active")
        self.mode_thread = threading.Thread(target=self.active_mode.start)
        self.mode_thread.start()

    def start_cursor_mode(self):
        self.start_mode("Cursor", self.cursor_mode)

    def start_presentation_mode(self):
        self.start_mode("Presentation", self.presentation_mode)

    def start_teaching_mode(self):
        self.start_mode("Teaching", self.teaching_mode)

    def stop_current_mode(self):
        if self.active_mode:
            self.active_mode.stop()
            self.active_mode = None
            if self.mode_thread and self.mode_thread.is_alive():
                self.mode_thread.join()
                self.mode_thread = None
            self.status_label.config(text="Status: Stopped")

if __name__ == "__main__":
    root = tk.Tk()
    app = SciFiApp(root)
    root.mainloop()
