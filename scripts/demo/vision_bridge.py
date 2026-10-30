# ── FILE: C:\projects\drone_pursuit\drone_pursuit\scripts\demo\vision_bridge.py
# ── This is a NEW file — the whole content, nothing to insert into.
# ── Reads : C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx
# ── Imported by: vision_pursuit_demo.py (6.2) and fly_real.py (7.2)

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

    def update(self, bbox):
        prev = (self.bx, self.by, self.asz)
        if bbox is not None:
            x_c, y_c, w, h = bbox
            self.bx = (x_c - self.img_w / 2) / (self.img_w / 2)     # −1 … +1
            self.by = (y_c - self.img_h / 2) / (self.img_h / 2)     # −1 … +1
            self.asz = w / self.img_w                               #  0 … 1
            self.has_seen = True
            visible = 1.0
        else:
            visible = 0.0                                           # values stay frozen
        d = (self.bx - prev[0], self.by - prev[1], self.asz - prev[2])
        return np.array([self.bx, self.by, self.asz, d[0], d[1], d[2], visible],
                        dtype=np.float32)

    def captured(self, threshold):
        """Camera-only victory condition, using the value calibrated in Chapter 3.3."""
        return self.has_seen and self.asz > threshold