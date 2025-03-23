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

        # Flag controlling the auto-switch detection thread
        self.auto_switch_running = True

        # Create UI elements
        self.create_ui()

        # Start two-hands detection in a separate thread
        self.auto_switch_thread = threading.Thread(target=self.detect_two_hands, daemon=True)
        self.auto_switch_thread.start()

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
                 " - Teaching Mode: Draw, Erase, Clear Canvas.\n"
                 "\n"
                 "Auto Mode Switch:\n"
                 " - Show BOTH hands for ~5 seconds at any time.\n"
                 "   A popup will appear, then the mode will change.",
            fg="#FFFFFF", bg="#222222", font=("Arial", 12)
        )
        self.info_label.pack()

        # Optional image
        try:
            img = Image.open(r"C:\Users\aryan\Downloads\Gestures.jpg").resize((300, 200))
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

    def detect_two_hands(self):
        """
        Runs continuously, checking for two hands in frame for ~5s.
        If found, shows a 5s popup, then switches to the next mode.
        """
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        cap = cv2.VideoCapture(0)

        consecutive_frames_with_two_hands = 0
        required_frames = 300  # ~5 seconds at ~60 FPS (or 150 for ~5s at ~30 FPS)

        mode_list = ["cursor", "presentation", "teaching"]
        current_mode_index = 0

        while self.auto_switch_running:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                consecutive_frames_with_two_hands += 1
            else:
                consecutive_frames_with_two_hands = 0

            # If two hands detected for enough consecutive frames
            if consecutive_frames_with_two_hands >= required_frames:
                consecutive_frames_with_two_hands = 0

                # Cycle to next mode
                current_mode_index = (current_mode_index + 1) % len(mode_list)
                next_mode = mode_list[current_mode_index]

                # Show popup for 5 seconds
                self.show_popup(f"Switching to {next_mode.title()} Mode in 5s...", duration=5)

                # Wait 5 seconds so user sees the popup
                time.sleep(5)

                # Switch mode
                if next_mode == "cursor":
                    self.start_cursor_mode()
                elif next_mode == "presentation":
                    self.start_presentation_mode()
                elif next_mode == "teaching":
                    self.start_teaching_mode()

            # Small sleep to reduce CPU usage
            time.sleep(0.001)

        cap.release()

    def show_popup(self, message, duration=5):
        """
        Creates a self-vanishing popup label at the center of the window.
        'duration' is in seconds.
        """
        popup = tk.Toplevel(self.master)
        popup.title("")
        popup.geometry("+400+300")  # approximate center
        popup.config(bg="#333333")

        label = tk.Label(popup, text=message, fg="#FFFFFF", bg="#333333", font=("Arial", 14, "bold"))
        label.pack(padx=20, pady=20)

        # Vanish after 'duration' seconds
        popup.after(duration * 1000, popup.destroy)

    def on_closing(self):
        # Graceful shutdown
        self.auto_switch_running = False
        self.stop_current_mode()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
