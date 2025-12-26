import time
import math

class BehaviorDecider:
    def __init__(
        self,
        move_threshold=50,
        warning_time=100,
        alert_time=250,
        dis_dir_changes = 6,
        dis_speed = 30,
        dis_displacement = 150
    ):
        self.MOVE_THRESHOLD = move_threshold
        self.WARNING_TIME = warning_time
        self.ALERT_TIME = alert_time

        self.DIS_DIR = dis_dir_changes
        self.DIS_SPEED = dis_speed
        self.DIS_DIST = dis_displacement

        self.last_position = {}
        self.still_start_time = {}

    def _distance(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def update(self, tracks, motion_info):
        now = time.time()
        results = []

        motion_map = {m["track_id"]: m for m in motion_info}

        for t in tracks:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            attr = t.get("attributes", {})

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            pos = (cx, cy)

            # --- First time seen ---
            if track_id not in self.last_position:
                self.last_position[track_id] = pos
                self.still_start_time[track_id] = None   # IMPORTANT FIX
                results.append({
                    "track_id": track_id,
                    "decision": "Normal",
                    "reason": ""
                })
                continue

            dist = self._distance(self.last_position[track_id], pos)

            # --- MOVEMENT HANDLING (FIX) ---
            if dist >= self.MOVE_THRESHOLD:
                # Person moved → reset still timer
                self.still_start_time[track_id] = None
            else:
                # Person stopped → start timer if not started
                if self.still_start_time[track_id] is None:
                    self.still_start_time[track_id] = now

            still_time = (
                now - self.still_start_time[track_id]
                if self.still_start_time[track_id]
                else 0
            )

            decision = "Normal"
            reason = ""

            m = motion_map.get(track_id, {})

            # --- PPE (highest priority) ---
            if attr.get("helmet", False):
                decision = "Alert"
                reason = "Helmet worn"

            elif attr.get("mask", False) or attr.get("hat", False):
                decision = "Warning"
                reason = "Mask or hat worn"

            # --- Disoriented behavior ---
            elif (
                m.get("avg_speed", 0) > self.DIS_SPEED   # MUST be moving
                and m.get("direction_changes", 0) >= self.DIS_DIR
                and m.get("displacement", 0) <= self.DIS_DIST
            ):

                decision = "Warning"
                reason = "Disoriented navigation"

            # --- Standing still (FIXED) ---
            elif self.still_start_time[track_id] is not None:
                if still_time >= self.ALERT_TIME:
                    decision = "Alert"
                    reason = "Standing still too long"
                elif still_time >= self.WARNING_TIME:
                    decision = "Warning"
                    reason = "Standing still"

            self.last_position[track_id] = pos

            results.append({
                "track_id": track_id,
                "decision": decision,
                "reason": reason
            })

        return results
