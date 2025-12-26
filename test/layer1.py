import cv2
from ingest.layer1_frame_ingest import FrameIngestor
from config.layer0_cameras import CAMERAS

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=1)

for data in ingestor.read():
    frame = data["frame"]
    frame_id = data["frame_id"]
    print(f"Frame {frame_id}")
    cv2.imshow("Layer1 - Frame Ingestor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

ingestor.release()
cv2.destroyAllWindows()
