import cv2
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
from config.layer0_cameras import CAMERAS

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=3)
detector = YOLODetector(classes=["person"])
tracker = SortTracker()
motion = MotionAnalyzer(frame_gaps=[1,5,10,15,20])
behavior = BehaviorDecider(warning_time=100, alert_time=200)

for data in ingestor.read():
    frame = data["frame"]
    person_dets = [d for d in detector.detect(frame) if d["class"]=="person"]
    tracks = tracker.update(person_dets, frame)
    motion_info = motion.update(tracks, data["frame_id"])
    behavior_info = behavior.update(tracks, motion_info)

    for t,b in zip(tracks, behavior_info):
        x1,y1,x2,y2 = t["bbox"]
        decision = b["decision"]
        color = (255,0,0) if decision=="Normal" else (0,165,255) if decision=="Warning" else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"ID {t['track_id']} {decision}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,2)

    cv2.imshow("Layer5 - Behavior", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

ingestor.release()
cv2.destroyAllWindows()
