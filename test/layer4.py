import cv2
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from config.layer0_cameras import CAMERAS

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=3)
detector = YOLODetector(classes=["person"])
tracker = SortTracker()
motion = MotionAnalyzer(frame_gaps=[1,5,10,15,20])

for data in ingestor.read():
    frame = data["frame"]
    person_dets = [d for d in detector.detect(frame) if d["class"]=="person"]
    tracks = tracker.update(person_dets, frame)
    motion_info = motion.update(tracks, data["frame_id"])

    for m in motion_info:
        tid = m["track_id"]
        cv2.putText(frame, f"ID {tid} speed:{m['avg_speed']:.1f}", (10, 20+20*tid), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1)

    cv2.imshow("Layer4 - Motion", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

ingestor.release()
cv2.destroyAllWindows()
