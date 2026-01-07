import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

class ImageDisplayWidget(QLabel):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.setText(f"[{title}]\nREADY TO START")
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Consolas", 14))
        self.setStyleSheet("""
            ImageDisplayWidget {
                border: 2px solid #333; 
                border-radius: 10px;
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a1a1a, stop:1 #050505);
                color: #555;
            }
        """)
        self.setMinimumSize(800, 600)

    def update_frame(self, frame):
        if frame is None: return
        try:
            h, w = frame.shape[:2]
            if len(frame.shape) == 2: # Depth/Grayscale
                # Normalize depth for visualization if it's uint16
                if frame.dtype == np.uint16:
                    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                q_img = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
            else: # BGR (from SDK) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format_RGB888)
            
            self.setText("")
            self.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass