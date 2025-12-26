import cv2
import time

class FrameIngestor:
    def __init__(self, camera_id, source, sample_rate=3):
        self.camera_id = camera_id
        self.source = source
        self.sample_rate = sample_rate
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream/video: {source}")

    def read(self):
        """
        Generator that yields sampled frames.
        Stops cleanly when video ends.
        """
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("[INFO] Video stream ended")
                return   # ✅ IMPORTANT: stop generator

            self.frame_count += 1

            # Sampling
            if self.frame_count % self.sample_rate != 0:
                continue

            yield {
                "camera_id": self.camera_id,
                "timestamp": time.time(),
                "frame_id": self.frame_count,
                "frame": frame
            }

    def release(self):
        self.cap.release()
