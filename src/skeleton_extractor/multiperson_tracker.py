import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import jit

def nms(boxes, iou_threshold=0.5):
    """Non‑Maximum Suppression to remove duplicate boxes."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = w * h
        iou = overlap / (areas[i] + areas[order[1:]] - overlap)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

@jit(nopython=True)
def _iou_numba(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def _iou_matrix_vectorized(tracks_boxes, detections_boxes):
    tracks_exp = tracks_boxes[:, np.newaxis, :]
    dets_exp = detections_boxes[np.newaxis, :, :]
    x1 = np.maximum(tracks_exp[..., 0], dets_exp[..., 0])
    y1 = np.maximum(tracks_exp[..., 1], dets_exp[..., 1])
    x2 = np.minimum(tracks_exp[..., 2], dets_exp[..., 2])
    y2 = np.minimum(tracks_exp[..., 3], dets_exp[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_t = (tracks_exp[..., 2] - tracks_exp[..., 0]) * (tracks_exp[..., 3] - tracks_exp[..., 1])
    area_d = (dets_exp[..., 2] - dets_exp[..., 0]) * (dets_exp[..., 3] - dets_exp[..., 1])
    union = area_t + area_d - inter
    return inter / (union + 1e-6)

class Track:
    __slots__ = ('track_id', 'lost', 'state', 'dt', 'F', 'P', 'Q', 'R')
    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.lost = 0
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.dt = 1.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = self.dt
        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8) * 0.01
        self.R = np.eye(4) * 1.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        z = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32)
        H = np.zeros((4,8))
        H[:4,:4] = np.eye(4)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def bbox(self):
        cx, cy, w, h = self.state[:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class SimpleMPT:
    def __init__(self, iou_threshold=0.4, max_lost=5, nms_threshold=0.5, use_numba=True):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.nms_threshold = nms_threshold
        self.tracks = []
        self.next_id = 0
        self._iou_func = self._iou_matrix_vectorized if not use_numba else self._iou_matrix_numba

    @staticmethod
    def _iou_matrix_vectorized(tracks_boxes, detections_boxes):
        return _iou_matrix_vectorized(tracks_boxes, detections_boxes)

    @staticmethod
    @jit(nopython=True)
    def _iou_matrix_numba(tracks_boxes, detections_boxes):
        N = len(tracks_boxes)
        M = len(detections_boxes)
        iou_mat = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            tbox = tracks_boxes[i]
            for j in range(M):
                dbox = detections_boxes[j]
                x1 = max(tbox[0], dbox[0])
                y1 = max(tbox[1], dbox[1])
                x2 = min(tbox[2], dbox[2])
                y2 = min(tbox[3], dbox[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_t = (tbox[2] - tbox[0]) * (tbox[3] - tbox[1])
                area_d = (dbox[2] - dbox[0]) * (dbox[3] - dbox[1])
                union = area_t + area_d - inter
                iou_mat[i, j] = inter / (union + 1e-6)
        return iou_mat

    def update(self, detections):
        """
        Returns: (tracks, matched_det_boxes)
        matched_det_boxes: list of the raw detection boxes that were used to update each track,
                           in the same order as tracks. For new tracks, it's the detection that created them.
        """
        # Apply NMS
        if len(detections) > 0:
            keep = nms(detections, self.nms_threshold)
            detections = [detections[i] for i in keep]

        # Predict all tracks
        for t in self.tracks:
            t.predict()

        # No tracks -> create from all detections
        if len(self.tracks) == 0:
            new_tracks = []
            matched_boxes = []
            for det in detections:
                new_tracks.append(Track(det, self.next_id))
                matched_boxes.append(det)               # raw detection used
                self.next_id += 1
            self.tracks = new_tracks
            return self.tracks, matched_boxes

        # No detections
        if len(detections) == 0:
            for t in self.tracks:
                t.lost += 1
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
            # No detections, so no matched boxes
            return self.tracks, [None] * len(self.tracks)

        # Build cost matrix
        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        det_boxes = np.array(detections, dtype=np.float32)
        iou_mat = self._iou_func(track_boxes, det_boxes)
        cost_mat = 1.0 - iou_mat

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        # Prepare result containers
        new_tracks = []
        matched_boxes = [None] * len(self.tracks)   # will be filled for matched tracks

        # Process matched pairs
        assigned_tracks = set()
        assigned_dets = set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_threshold:
                # Update track with the detection
                self.tracks[r].update(detections[c])
                self.tracks[r].lost = 0
                new_tracks.append(self.tracks[r])
                matched_boxes[r] = detections[c]      # store the raw detection box
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Unmatched tracks: increase lost counter and keep if still alive
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.lost += 1
                if t.lost <= self.max_lost:
                    new_tracks.append(t)
                    matched_boxes[i] = None           # no detection for this track
                # else discarded

        # Unmatched detections: create new tracks
        for j, d in enumerate(detections):
            if j not in assigned_dets:
                new_track = Track(d, self.next_id)
                new_tracks.append(new_track)
                matched_boxes.append(d)               # raw detection for new track
                self.next_id += 1

        self.tracks = new_tracks
        return self.tracks, matched_boxes