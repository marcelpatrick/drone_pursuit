# ── FILE: C:\projects\drone_pursuit\drone_pursuit\test_detector_onnx.py
# ── NEW file — the whole content, nothing to insert into.
# ── Environment: env_drone
# ── Checks that models\drone_detector.onnx loads and runs with onnxruntime alone
# ──   (no Ultralytics, no torch), with the camera's frame size, 480 x 640.
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession(
    r"C:\projects\drone_pursuit\drone_pursuit\models\drone_detector.onnx",
    providers=["CPUExecutionProvider"],
)
inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape)

dummy = np.random.rand(1, 3, 480, 640).astype(np.float32)   # one 480x640 frame of random numbers
out = sess.run(None, {inp.name: dummy})
print("output:", out[0].shape)
print("drone_detector.onnx runs inside env_drone")