import base64
import io
import json
from datetime import datetime
import hashlib
from typing import Annotated

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context

from classifier import MRIClassifier, TemperatureScaler
from preprocess import CLASSES, IMG_SIZE, NUM_CLASSES

# Initialize FastMCP server
mcp = FastMCP("Brain Tumor Classifier", stateless_http=True, host="0.0.0.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Info
CLASS_INFO = {
    "astrocytoma": {"display": "Astrocytoma", "severity": "high"},
    "ependymoma": {"display": "Ependymoma", "severity": "high"},
    "glioma": {"display": "Glioma", "severity": "high"},
    "meningioma": {"display": "Meningioma", "severity": "medium"},
    "neurocytoma": {"display": "Neurocytoma", "severity": "medium"},
    "oligodendroglioma": {"display": "Oligodendroglioma", "severity": "high"},
    "schwannoma": {"display": "Schwannoma", "severity": "low"},
    "hemangiopericytoma": {"display": "Hemangiopericytoma", "severity": "high"},
    "normal": {"display": "No Tumor", "severity": "none"},
}


def load_model():
    base = MRIClassifier().to(DEVICE)
    base.load_state_dict(torch.load("checkpoints/classifier_final.pt",
                                    map_location=DEVICE, weights_only=True))
    base.eval()
    try:
        scaler = TemperatureScaler(base)
        scaler_state = torch.load("checkpoints/scaler_fold1.pt",
                                  map_location=DEVICE, weights_only=True)
        scaler.load_state_dict(scaler_state)
        scaler.eval()
        return scaler.to(DEVICE)
    except FileNotFoundError:
        return base


def load_thresholds():
    try:
        with open("checkpoints/locked_thresholds.json", "r") as f:
            raw = json.load(f)
        return {CLASSES.index(k): v for k, v in raw.items() if k in CLASSES}
    except FileNotFoundError:
        return {i: 0.5 for i in range(NUM_CLASSES)}


model = load_model()
thresholds = load_thresholds()


def apply_fft(img_tensor):
    fft = torch.fft.fft2(img_tensor)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.log1p(torch.abs(fft_shift))
    magnitude = magnitude - magnitude.min()
    magnitude = magnitude / (magnitude.max() + 1e-8)
    magnitude = magnitude * 2 - 1
    return magnitude


# Patch capabilities to indicate we DON'T need FHIR context
_original_get_capabilities = mcp._mcp_server.get_capabilities


def _patched_get_capabilities(notification_options, experimental_capabilities):
    caps = _original_get_capabilities(notification_options, experimental_capabilities)
    # We don't require FHIR context - brain tumor analysis works standalone
    caps.model_extra["extensions"] = {
        "ai.promptopinion/fhir-context": {
            "scopes": []  # Empty - we don't need FHIR access
        }
    }
    return caps


mcp._mcp_server.get_capabilities = _patched_get_capabilities


@mcp.tool(
    name="analyze_mri",
    description="Neuro-symbolic brain tumor classification from MRI scan. Analyzes MRI images using hybrid AI (neural networks + symbolic clinical reasoning) to classify 9 brain tumor types. Returns FHIR-compliant DiagnosticReport with confidence scores, uncertainty detection, and clinical recommendations."
)
async def analyze_mri(
        image_data: Annotated[str, Field(description="Base64-encoded MRI image (JPG, PNG, or JPEG)")],
        patient_reference: Annotated[
            str, Field(description="FHIR patient reference (e.g., 'Patient/12345')")] = "Patient/unknown",
        ctx: Context = None
) -> str:
    """Analyze brain MRI for tumor classification"""

    # Decode base64 image
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(image).unsqueeze(0)
    tensor = apply_fft(tensor)
    tensor = tensor.to(DEVICE)

    # Neural prediction
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_cls = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        locked_t = thresholds.get(pred_idx, 0.5)
        uncertain = conf < locked_t

    info = CLASS_INFO.get(pred_cls, {"display": pred_cls, "severity": "unknown"})

    # Symbolic reasoning trace
    sorted_probs = sorted(probs, reverse=True)
    top2_diff = sorted_probs[0] - sorted_probs[1]

    reasoning = {
        "rule1_threshold": {
            "confidence": round(conf, 4),
            "threshold": round(locked_t, 2),
            "result": "UNCERTAIN" if uncertain else "CONFIDENT"
        },
        "rule2_severity": {
            "predicted_class": pred_cls,
            "severity": info["severity"],
            "priority": "HIGH" if info["severity"] == "high" else "MEDIUM"
        },
        "rule3_ambiguity": {
            "top2_gap": round(top2_diff, 4),
            "result": "AMBIGUOUS" if top2_diff < 0.15 else "CLEAR"
        }
    }

    # Build FHIR DiagnosticReport
    report_id = hashlib.md5(patient_reference.encode()).hexdigest()[:8]

    fhir_report = {
        "resourceType": "DiagnosticReport",
        "id": f"mri-{report_id}",
        "status": "preliminary" if uncertain else "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "RAD",
                "display": "Radiology"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "25045-6",
                "display": "MRI Brain"
            }],
            "text": "Brain Tumor MRI Classification"
        },
        "subject": {
            "reference": patient_reference
        },
        "issued": datetime.now().isoformat(),
        "conclusion": f"{'Uncertain - refer to specialist' if uncertain else info['display'] + ' detected'}",
        "conclusionCode": [{
            "text": info["display"]
        }],
        "result": [
            {
                "display": CLASS_INFO[CLASSES[i]]["display"],
                "valueQuantity": {
                    "value": round(float(probs[i]), 4),
                    "unit": "probability"
                }
            }
            for i in range(NUM_CLASSES)
        ],
        "extension": [{
            "url": "http://ai-diagnostics.org/fhir/neurosymbolic",
            "extension": [
                {
                    "url": "neuralOutput",
                    "valueString": json.dumps({
                        "topPrediction": pred_cls,
                        "confidence": round(conf, 4),
                        "probabilities": [round(float(p), 4) for p in probs]
                    })
                },
                {
                    "url": "symbolicReasoning",
                    "valueString": json.dumps(reasoning)
                },
                {
                    "url": "uncertaintyFlag",
                    "valueBoolean": uncertain
                },
                {
                    "url": "recommendedAction",
                    "valueString": "Refer to radiologist" if uncertain else "Clinical correlation advised"
                }
            ]
        }]
    }

    return json.dumps(fhir_report, indent=2)