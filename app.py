import os
import uuid
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from flask import Flask, render_template, request
from fastai.vision.all import load_learner

import pathlib
import matplotlib.cm as cm

# Fix FastAI model exported on Windows but loaded on Linux
pathlib.WindowsPath = pathlib.PosixPath

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/Human_Face_Emotions/demo_stage-3.pkl"
)

UPLOAD_DIR = Path("static/uploads")
RESULT_DIR = Path("static/results")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

learn = load_learner(MODEL_PATH, cpu=True)
model = learn.model
model.eval()

def find_last_conv_layer(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("找不到 Conv2d layer，無法產生 Grad-CAM")
    return last_conv

target_layer = find_last_conv_layer(model)

def generate_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")

    dl = learn.dls.test_dl([img])
    xb = dl.one_batch()[0]

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()

    output = model(xb)
    class_idx = output.argmax(dim=1).item()
    score = output[0, class_idx]

    score.backward()

    fh.remove()
    bh.remove()

    acts = activations["value"].detach()
    grads = gradients["value"].detach()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam = cam.cpu().numpy()

    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)

    heatmap = Image.fromarray(np.uint8(cam * 255)).resize(
        original.size,
        Image.BILINEAR
    )

    heatmap_np = np.array(heatmap)
    colored_heatmap = cm.jet(heatmap_np / 255.0)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    overlay = np.uint8(original_np * 0.55 + colored_heatmap * 0.45)

    result_name = f"{uuid.uuid4().hex}.jpg"
    result_path = RESULT_DIR / result_name

    Image.fromarray(overlay).save(result_path)

    pred_label = learn.dls.vocab[class_idx] if hasattr(learn.dls, "vocab") else str(class_idx)

    probs = torch.softmax(output, dim=1)
    confidence = probs[0, class_idx].item()

    return result_path, pred_label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result_url = None
    label = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = f"{uuid.uuid4().hex}.jpg"
            upload_path = UPLOAD_DIR / filename
            file.save(upload_path)

            result_path, label, confidence = generate_gradcam(upload_path)
            result_url = "/" + str(result_path)

    return render_template(
        "index.html",
        result_url=result_url,
        label=label,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)