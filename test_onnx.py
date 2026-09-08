import numpy as np, onnxruntime as ort

sess = ort.InferenceSession(
    r"C:\projects\drone_pursuit\drone_pursuit\models\yolov8n.onnx",
    providers=["CPUExecutionProvider"],
)
inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape)

dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
out = sess.run(None, {inp.name: dummy})
print("output:", out[0].shape)
print("ONNX round-trip works inside env_drone")