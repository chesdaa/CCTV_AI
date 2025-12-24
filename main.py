import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
# from tracker.layer6_telegram import TelegramNotifier  # Uncomment when ready

def check_bbox_overlap(person_bbox, ppe_bbox):
    """Check if PPE bbox overlaps with person bbox (PPE center inside person box)"""
    px1, py1, px2, py2 = person_bbox
    dx1, dy1, dx2, dy2 = ppe_bbox
    
    # Calculate center of PPE bbox
    ppe_center_x = (dx1 + dx2) / 2
    ppe_center_y = (dy1 + dy2) / 2
    
    # Check if PPE center is inside person bbox
    if px1 <= ppe_center_x <= px2 and py1 <= ppe_center_y <= py2:
        return True
    
    # Also check if there's any intersection at all
    x_left = max(px1, dx1)
    y_top = max(py1, dy1)
    x_right = min(px2, dx2)
    y_bottom = min(py2, dy2)
    
    if x_right > x_left and y_bottom > y_top:
        return True
    
    return False

def main():
    cam = CAMERAS[0]

    # --- Initialize layers ---
    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person", "mask", "helmet", "hat"])
    tracker = SortTracker()
    motion = MotionAnalyzer(frame_gaps=[1,5,10,15,20])
    
    behavior_decider = BehaviorDecider(
        warning_time=180,   # 3 minutes
        alert_time=300      # 5 minutes
    )
    # notifier = TelegramNotifier(BOT_TOKEN, CHAT_ID)  # Uncomment when ready

    print("Pipeline L1→L5 started... Press ESC to quit")

    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]
        cam_id = data["camera_id"]

        # --- Layer 2: Detection ---
        detections = detector.detect(frame)

        # --- DEBUG: Print all detected classes ---
        detected_classes = set([d["class"] for d in detections])
        if detected_classes:
            print(f"[DEBUG] Frame {frame_id}: Detected classes: {detected_classes}")

        # --- Separate person and PPE detections ---
        person_detections = [d for d in detections if d["class"] == "person"]
        ppe_detections = [d for d in detections if d["class"] in ["mask", "helmet", "hat"]]
        
        if ppe_detections:
            print(f"[DEBUG] Found {len(ppe_detections)} PPE items: {[p['class'] for p in ppe_detections]}")

        # --- Layer 3: Tracking (only track persons) ---
        tracks = tracker.update(person_detections)

        # --- Add PPE attributes using spatial overlap ---
        for t in tracks:
            # Default to False → assume PPE missing until detected
            t["attributes"] = {"mask": False, "helmet": False, "hat": False}
            person_bbox = t["bbox"]

            # Check all PPE detections and match using spatial overlap
            for ppe in ppe_detections:
                ppe_bbox = ppe["bbox"]
                cls = ppe["class"]
                
                # Check if PPE overlaps with person
                if check_bbox_overlap(person_bbox, ppe_bbox):
                    t["attributes"][cls] = True
                    print(f"[DEBUG] Track {t['track_id']}: Detected {cls}")  # Debug output

        # --- Layer 4: Motion Analysis ---
        motion_info = motion.update(tracks, frame_id)

        # --- Layer 5: Behavior Decision ---
        behavior_info = behavior_decider.update(tracks, motion_info)

        # --- Visualization: Bounding boxes + Behavior ---
        for t, b in zip(tracks, behavior_info):
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]
            decision = b["decision"]
            reason = b["reason"]

            # Choose color based on decision
            if decision == "Normal":
                color = (255, 0, 0)  # Blue
            elif decision == "Warning":
                color = (0, 165, 255)  # Orange
            elif decision == "Alert":
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # fallback white

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw track ID + decision (without reason)
            label = f"ID {track_id} | {decision}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Show PPE status for debugging
            ppe_status = []
            if t["attributes"]["mask"]:
                ppe_status.append("MASK")
            if t["attributes"]["helmet"]:
                ppe_status.append("HELMET")
            if t["attributes"]["hat"]:
                ppe_status.append("HAT")
            
            if ppe_status:
                cv2.putText(
                    frame,
                    f"PPE: {', '.join(ppe_status)}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Draw PPE detections separately for debugging
        for ppe in ppe_detections:
            px1, py1, px2, py2 = ppe["bbox"]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)  # Yellow
            cv2.putText(frame, f"{ppe['class'].upper()}", (px1, py1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Optional: show motion gaps for debugging
        for m in motion_info:
            track_id = m["track_id"]
            gap1 = int(m["motion_gaps"][1])
            cv2.putText(frame, f"G1:{gap1}", (10, 50 + 20*track_id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"{cam_id} | Frame {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("CCTV AI Pipeline L1-L5", frame)

        # Optional: Layer 6 Telegram alert (commented for testing)
        # for b in behavior_info:
        #     if b["decision"] in ["Warning", "Alert"]:
        #         track_id = b["track_id"]
        #         bbox = next((t["bbox"] for t in tracks if t["track_id"]==track_id), None)
        #         if bbox:
        #             x1, y1, x2, y2 = bbox
        #             crop = frame[y1:y2, x1:x2]
        #             notifier.send_alert(crop, track_id, cam_id, b["decision"], b["reason"])

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # --- Cleanup ---
    ingestor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()