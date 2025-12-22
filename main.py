import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
# from tracker.layer6_telegram import TelegramNotifier  # Uncomment when ready

def main():
    cam = CAMERAS[0]

    # --- Initialize layers ---
    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person", "mask", "helmet"])
    tracker = SortTracker()
    motion = MotionAnalyzer(frame_gaps=[1,5,10,15,20])
    behavior_decider = BehaviorDecider(motion_threshold=50, loitering_frames=10)
    # notifier = TelegramNotifier(BOT_TOKEN, CHAT_ID)  # Uncomment when ready

    print("Pipeline L1→L5 started... Press ESC to quit")

    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]
        cam_id = data["camera_id"]

        # --- Layer 2: Detection ---
        detections = detector.detect(frame)

        # --- Layer 3: Tracking ---
        tracks = tracker.update(detections)

        # --- Layer 4: Motion Analysis ---
        motion_info = motion.update(tracks, frame_id)

        # --- Layer 5: Behavior Decision ---
        behavior_info = behavior_decider.update(tracks, motion_info)

        # --- Visualization: Bounding boxes + Behavior ---
        for t, b in zip(tracks, behavior_info):
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]
            decision = b["decision"]

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

            # Draw track ID + decision text above bbox
            cv2.putText(
                frame,
                f"ID {track_id} | {decision}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

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
