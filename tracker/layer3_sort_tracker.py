from deep_sort_realtime.deepsort_tracker import DeepSort

class SortTracker:
    def __init__(self, max_age=30, n_init=3):
        # Initialize DeepSort tracker
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame=None):
        """
        detections: list of dicts with keys ["bbox", "class", "confidence"]
        frame: current frame for embedding generation (required by DeepSort)
        """
        dets = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d.get("confidence", 1.0)
            dets.append(([x1, y1, x2, y2], conf, d["class"]))

        if frame is None:
            # DeepSort requires either embeddings or frame
            raise ValueError("Frame must be provided for tracking")

        tracks = self.tracker.update_tracks(dets, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            results.append({
                "track_id": t.track_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        return results
