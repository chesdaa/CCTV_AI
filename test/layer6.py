import numpy as np
from tracker.layer6_telegram import TelegramNotifier
from config.telegram_config import BOT_TOKEN, CHAT_ID

telegram = TelegramNotifier(bot_token=BOT_TOKEN, chat_id=CHAT_ID)
dummy_frame = np.zeros((480,640,3), dtype=np.uint8)

telegram.send_alert(dummy_frame, track_id=1, camera_id="CAM1", decision="Warning", reason="Test Warning")
telegram.send_alert(dummy_frame, track_id=1, camera_id="CAM1", decision="Alert", reason="Test Alert")

print("Telegram test sent. Check your bot.")
