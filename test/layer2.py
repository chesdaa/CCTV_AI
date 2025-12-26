import cv2
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from config.layer0_cameras import CAMERAS

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=3)
detector = YOLODetector(classes=["person","mask","helmet","hat"])

for data in ingestor.read():
    frame = data["frame"]
    detections = detector.detect(frame)

    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, det["class"], (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    cv2.imshow("Layer2 - YOLO", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

ingestor.release()
cv2.destroyAllWindows()
