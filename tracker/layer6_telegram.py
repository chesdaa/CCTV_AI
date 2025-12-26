import requests
import cv2
import os
from datetime import datetime, timedelta

class TelegramNotifier:
    def __init__(self, bot_token, chat_id, cooldown_sec=240, save_snapshots=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_sec = cooldown_sec  # 4 min cooldown
        self.save_snapshots = save_snapshots
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        self.tmp_dir = "snapshots"
        self.last_sent = {}  # track_id -> timestamp

        if self.save_snapshots and not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def send_alert(self, frame, track_id, cam_id, decision, reason):
        import time, cv2, requests
        now = time.time()
        last = self.last_sent.get(track_id, 0)

        if now - last < self.cooldown_sec:
            return  # skip sending, cooldown not reached

        self.last_sent[track_id] = now

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tmp_dir}/cam{cam_id}_track{track_id}_{timestamp}.jpg"

        if self.save_snapshots:
            cv2.imwrite(filename, frame)
        else:
            filename = frame

        caption = f"Camera: {cam_id}\nTrack ID: {track_id}\nDecision: {decision}\nReason: {reason}"

        try:
            with open(filename, "rb") as img_file:
                files = {"photo": img_file}
                data = {"chat_id": self.chat_id, "caption": caption}
                requests.post(self.base_url, files=files, data=data, timeout=5)
        except Exception as e:
            print(f"[Telegram] Failed: {e}")
