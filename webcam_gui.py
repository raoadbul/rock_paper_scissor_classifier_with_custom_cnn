import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

from model import RPSNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/rps_custom_cnn.pth")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Below this confidence, show 'Uncertain / show hand clearly'.")
    return p.parse_args()


class RPSWebcamApp:
    def __init__(self, root, checkpoint_path, camera_index=0, threshold=0.60):
        self.root = root
        self.root.title("Rock • Paper • Scissors — Live Classifier")
        self.root.geometry("920x720")
        self.root.minsize(700, 560)
        self.threshold = threshold
        self.device = torch.device("cpu")  # Designed to work on CPU laptops.

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint["class_names"]
        self.image_size = int(checkpoint.get("image_size", 128))
        self.model = RPSNet(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint.get("normalization_mean", [0.5, 0.5, 0.5]),
                std=checkpoint.get("normalization_std", [0.5, 0.5, 0.5]),
            ),
        ])

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Close other camera apps or try --camera 1."
            )

        self.video_label = tk.Label(root, bg="#171717")
        self.video_label.pack(padx=12, pady=12, fill="both", expand=True)

        self.prediction_label = tk.Label(
            root, text="Show a rock, paper, or scissor sign",
            font=("Arial", 22, "bold"), pady=8
        )
        self.prediction_label.pack()

        self.confidence_label = tk.Label(root, text="Confidence: --",
                                          font=("Arial", 14))
        self.confidence_label.pack(pady=(0, 8))

        self.status_label = tk.Label(
            root, text="CPU inference • Press Q or click Quit to exit",
            font=("Arial", 10), fg="#555555"
        )
        self.status_label.pack(pady=(0, 8))

        self.quit_button = tk.Button(root, text="Quit", command=self.close,
                                     font=("Arial", 12), padx=18, pady=5)
        self.quit_button.pack(pady=(0, 12))

        self.previous_time = time.time()
        self.fps = 0.0
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame()

    @torch.no_grad()
    def classify(self, frame):
        # OpenCV uses BGR; PIL expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
        return self.classes[index.item()], float(confidence.item())

    def update_frame(self):
        if self.closed:
            return

        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            label, confidence = self.classify(frame)

            if confidence >= self.threshold:
                shown_label = label.upper()
                self.prediction_label.config(
                    text=f"Prediction: {shown_label}", fg="#176B35"
                )
                self.confidence_label.config(
                    text=f"Confidence: {confidence:.1%}"
                )
            else:
                self.prediction_label.config(
                    text="Uncertain — show your hand clearly", fg="#A35A00"
                )
                self.confidence_label.config(
                    text=f"Top prediction: {label} ({confidence:.1%})"
                )

            now = time.time()
            dt = now - self.previous_time
            if dt > 0:
                self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)
            self.previous_time = now

            cv2.putText(
                frame, f"FPS: {self.fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 80), 2
            )
            cv2.putText(
                frame, "Rock | Paper | Scissor", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # Fit the camera image into the current GUI display.
            display_w = max(640, self.video_label.winfo_width())
            display_h = max(360, self.video_label.winfo_height())
            h, w = frame.shape[:2]
            scale = min(display_w / w, display_h / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        else:
            self.status_label.config(text="Camera frame unavailable.")

        self.root.after(20, self.update_frame)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model}\nRun train.py first."
        )
    root = tk.Tk()
    try:
        RPSWebcamApp(root, args.model, args.camera, args.threshold)
    except Exception as exc:
        messagebox.showerror("Rock-Paper-Scissors", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
