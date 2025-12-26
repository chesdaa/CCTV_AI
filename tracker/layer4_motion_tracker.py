import numpy as np
import math
from collections import defaultdict, deque

class MotionAnalyzer:
    def __init__(self, frame_gaps=[1,5,10,15,20], history_size=25):
        self.frame_gaps = frame_gaps
        self.history_size = history_size
        self.track_history = defaultdict(lambda: deque(maxlen=history_size))

    def _angle(self, p1, p2):
        return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

    def update(self, tracks, frame_id):
        motion_info = []

        for t in tracks:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.track_history[track_id].append((frame_id, cx, cy))
            history = list(self.track_history[track_id])

            # --- Original motion gaps (KEEP) ---
            motion_gaps = {}
            for gap in self.frame_gaps:
                if len(history) > gap:
                    _, old_x, old_y = history[-gap-1]
                    motion_gaps[gap] = np.hypot(cx-old_x, cy-old_y)
                else:
                    motion_gaps[gap] = 0.0

            # --- New: direction changes ---
            angles = []
            for i in range(1, len(history)):
                angles.append(self._angle(history[i-1][1:], history[i][1:]))

            direction_changes = 0
            for i in range(1, len(angles)):
                if abs(angles[i] - angles[i-1]) > 45:
                    direction_changes += 1

            # --- New: speed ---
            total_dist = 0
            for i in range(1, len(history)):
                x0, y0 = history[i-1][1:]
                x1_, y1_ = history[i][1:]
                total_dist += math.hypot(x1_-x0, y1_-y0)

            total_time = max(len(history)-1, 1)
            avg_speed = total_dist / total_time

            # --- New: displacement ---
            displacement = math.hypot(
                history[-1][1] - history[0][1],
                history[-1][2] - history[0][2]
            )

            motion_info.append({
                "track_id": track_id,
                "frame_id": frame_id,
                "motion_gaps": motion_gaps,
                "direction_changes": direction_changes,
                "avg_speed": avg_speed,
                "displacement": displacement
            })

        return motion_info
