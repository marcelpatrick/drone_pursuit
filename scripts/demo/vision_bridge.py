# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py
# ── This is a NEW file — the whole content, nothing to insert into.
# ── Reads : C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
# ── Imported by: vision_pursuit_demo.py (6.2) and fly_real.py (7.2)

from collections import deque
import numpy as np, onnxruntime as ort


class DroneDetector:
    """Runs the exported YOLO model and returns the single best rectangle."""

    def __init__(self, onnx_path, conf_thresh=0.4):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.conf = conf_thresh

    def detect(self, rgb_uint8_hwc):
        """480x640x3 uint8 (H, W, C) → (x_center, y_center, width, height) in pixels, or None."""
        x = rgb_uint8_hwc.astype(np.float32) / 255.0        # scale to 0..1
        x = np.transpose(x, (2, 0, 1))[None]                # HWC → 1,C,H,W
        out = self.sess.run(None, {self.input_name: x})[0][0]   # (5, 6300)
        boxes, scores = out[:4, :], out[4, :]
        best = int(np.argmax(scores))
        if scores[best] < self.conf:
            return None                                     # attacker not seen this frame
        return boxes[:, best]


class CameraReadingBridge:
    """Rectangle → the seven numbers the policy was trained on. Holds last value on a miss."""

    def __init__(self, img_w=640, img_h=480):   # MUST match the frame you feed it
        self.img_w, self.img_h = img_w, img_h
        self.bx = self.by = self.asz = 0.0
        self.has_seen = False
        self._growth = deque(maxlen=4)   # last 4 size changes between CONSECUTIVE sightings, to smooth jitter
        self._was_visible = False        # was the attacker seen in the previous picture?

    def update(self, bbox):
        prev = (self.bx, self.by, self.asz)
        if bbox is not None:
            x_c, y_c, w, h = bbox
            self.bx = (x_c - self.img_w / 2) / (self.img_w / 2)     # −1 … +1
            self.by = (y_c - self.img_h / 2) / (self.img_h / 2)     # −1 … +1
            self.asz = w / self.img_w                               #  0 … 1
            self.has_seen = True
            if self._was_visible:                                   # growth only between two real sightings:
                self._growth.append(self.asz - prev[2])             # the jump from "nothing" to a first box is not growth
            self._was_visible = True
            visible = 1.0
        else:
            visible = 0.0                                           # values stay frozen
            self._was_visible = False
            self._growth.clear()                                    # a gap breaks the step-by-step growth series
        d = (self.bx - prev[0], self.by - prev[1], self.asz - prev[2])
        return np.array([self.bx, self.by, self.asz, d[0], d[1], d[2], visible],
                        dtype=np.float32)

    def captured(self, threshold):
        """Camera-only victory condition, using the value calibrated in Chapter 3.3."""
        return self.has_seen and self.asz > threshold

    def time_to_contact(self, dt):
        """Seconds until the drones meet at the current closing speed: size ÷ growth per second.
        Needs no attacker width — a bigger drone looks bigger AND grows faster, so the width cancels.
        dt = seconds between control steps (0.05 at 20 decisions per second)."""
        if len(self._growth) < 2:                                   # needs 3 consecutive sightings
            return float("inf")
        n = len(self._growth)
        growth_per_s = float(np.mean(self._growth)) / dt
        if growth_per_s <= 0:
            return float("inf")                                     # not closing → no contact coming
        # Averaging n steps measures growth as it was ~n steps ago, which adds n×dt to the
        # result. Training's 3.3 Step 4 measured over ONE step (adds 1×dt), so remove the
        # extra (n−1)×dt to stay on the same scale as capture_ttc_s.
        return self.asz / growth_per_s - (n - 1) * dt

    def captured_ttc(self, ttc_threshold_s, dt):
        """Camera-only victory condition that works for any attacker size (3.3 Step 4's capture_ttc_s)."""
        return self.time_to_contact(dt) < ttc_threshold_s