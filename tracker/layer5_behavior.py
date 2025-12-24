import time
import math

class BehaviorDecider:
    def __init__(
        self,
        move_threshold=50,
        warning_time=180,   # 3 min
        alert_time=300      # 5 min
    ):
        self.MOVE_THRESHOLD = move_threshold
        self.WARNING_TIME = warning_time
        self.ALERT_TIME = alert_time

        self.last_position = {}       # track_id -> (x, y)
        self.still_start_time = {}    # track_id -> timestamp

    def _distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def update(self, tracks, motion_info):
        now = time.time()
        results = []

        for t in tracks:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            attr = t.get("attributes", {})

            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            pos = (cx, cy)

            if track_id not in self.last_position:
                self.last_position[track_id] = pos
                self.still_start_time[track_id] = now
                results.append({
                    "track_id": track_id,
                    "decision": "Normal",
                    "reason": "Init"
                })
                continue

            dist = self._distance(self.last_position[track_id], pos)
            still_time = now - self.still_start_time[track_id]

            # --- PPE first ---
            if attr.get("helmet", False):
                decision = "Alert"
                reason = "Helmet worn"
            elif attr.get("mask", False):
                decision = "Warning"
                reason = "Mask worn"
            elif attr.get("hat", False):
                decision = "Warning"
                reason = "Hat worn"
            else:
                # --- Standing still detection ---
                if dist < self.MOVE_THRESHOLD:
                    if still_time >= self.ALERT_TIME:
                        decision = "Alert"
                        reason = "Standing still too long (5 min)"
                    elif still_time >= self.WARNING_TIME:
                        decision = "Warning"
                        reason = "Standing still (3 min)"
                    else:
                        decision = "Normal"
                        reason = "Moving or short stop"
                else:
                    self.still_start_time[track_id] = now
                    decision = "Normal"
                    reason = "Moving"

            self.last_position[track_id] = pos

            results.append({
                "track_id": track_id,
                "decision": decision,
                "reason": reason
            })

        return results
