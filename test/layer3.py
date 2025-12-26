import cv2
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from config.layer0_cameras import CAMERAS

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=3)
detector = YOLODetector(classes=["person"])
tracker = SortTracker()

for data in ingestor.read():
    frame = data["frame"]
    person_dets = [d for d in detector.detect(frame) if d["class"]=="person"]
    tracks = tracker.update(person_dets, frame)

    for t in tracks:
        x1,y1,x2,y2 = t["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,f"ID {t['track_id']}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    cv2.imshow("Layer3 - Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

ingestor.release()
cv2.destroyAllWindows()
