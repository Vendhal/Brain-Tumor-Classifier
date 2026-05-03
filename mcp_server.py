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

# ─────────────────────────────────────────────
# CLASS INFO - full clinical data
# ─────────────────────────────────────────────
CLASS_INFO = {
    "astrocytoma": {
        "display": "Astrocytoma", "severity": "high",
        "description": "A tumor arising from astrocytes (star-shaped glial cells) in the brain or spinal cord.",
        "symptoms": ["Headaches", "Seizures", "Memory problems", "Personality changes"],
        "treatment": "Surgery, radiation therapy, chemotherapy (temozolomide)",
        "urgency": "URGENT", "timeframe": "See specialist within 1-2 weeks",
        "five_year_survival": "20-40% (grade dependent)"
    },
    "ependymoma": {
        "display": "Ependymoma", "severity": "high",
        "description": "A tumor that arises from ependymal cells lining the brain ventricles and spinal cord.",
        "symptoms": ["Headaches", "Nausea/vomiting", "Neck pain", "Balance problems"],
        "treatment": "Surgical resection followed by radiation therapy",
        "urgency": "URGENT", "timeframe": "See specialist within 1-2 weeks",
        "five_year_survival": "50-75% (location dependent)"
    },
    "glioma": {
        "display": "Glioma", "severity": "high",
        "description": "A broad category of tumors arising from glial cells. Includes glioblastoma (GBM), the most aggressive form.",
        "symptoms": ["Progressive headaches", "Seizures", "Cognitive decline", "Weakness/numbness"],
        "treatment": "Surgery + Temozolomide + Radiation (Stupp protocol for GBM)",
        "urgency": "IMMEDIATE", "timeframe": "Emergency referral within 24-48 hours",
        "five_year_survival": "5-30% (grade dependent)"
    },
    "meningioma": {
        "display": "Meningioma", "severity": "medium",
        "description": "A tumor arising from the meninges. Usually benign and slow-growing.",
        "symptoms": ["Headaches", "Vision problems", "Hearing loss", "Memory difficulties"],
        "treatment": "Observation (small/asymptomatic), Surgery, Stereotactic radiosurgery",
        "urgency": "ROUTINE", "timeframe": "See specialist within 4-6 weeks",
        "five_year_survival": "70-80%"
    },
    "neurocytoma": {
        "display": "Neurocytoma", "severity": "medium",
        "description": "A rare benign tumor typically found in the ventricles of young adults.",
        "symptoms": ["Hydrocephalus symptoms", "Headaches", "Vision changes"],
        "treatment": "Surgical resection, often curative",
        "urgency": "URGENT", "timeframe": "See specialist within 2-4 weeks",
        "five_year_survival": "85-90%"
    },
    "oligodendroglioma": {
        "display": "Oligodendroglioma", "severity": "high",
        "description": "A tumor arising from oligodendrocytes. Often chemo-sensitive due to 1p/19q co-deletion.",
        "symptoms": ["Seizures (often first symptom)", "Headaches", "Cognitive changes"],
        "treatment": "Surgery + PCV chemotherapy + Radiation",
        "urgency": "URGENT", "timeframe": "See specialist within 1-2 weeks",
        "five_year_survival": "60-80% (grade/genetics dependent)"
    },
    "schwannoma": {
        "display": "Schwannoma", "severity": "low",
        "description": "A benign tumor arising from Schwann cells. Most common: vestibular schwannoma (acoustic neuroma).",
        "symptoms": ["Hearing loss", "Tinnitus", "Balance problems", "Facial numbness"],
        "treatment": "Observation, Stereotactic radiosurgery (Gamma Knife), Surgery",
        "urgency": "ROUTINE", "timeframe": "See specialist within 4-8 weeks",
        "five_year_survival": "90-95%"
    },
    "hemangiopericytoma": {
        "display": "Hemangiopericytoma", "severity": "high",
        "description": "A rare vascular tumor arising from pericytes around blood vessels. High recurrence rate.",
        "symptoms": ["Headaches", "Neurological deficits", "Seizures"],
        "treatment": "Surgery + Radiation. Close long-term follow-up required.",
        "urgency": "URGENT", "timeframe": "See specialist within 1-2 weeks",
        "five_year_survival": "60-70%"
    },
    "normal": {
        "display": "No Tumor Detected", "severity": "none",
        "description": "No evidence of brain tumor on this MRI scan.",
        "symptoms": [],
        "treatment": "No tumor-specific treatment required. Continue routine monitoring if symptomatic.",
        "urgency": "ROUTINE", "timeframe": "Routine follow-up as clinically indicated",
        "five_year_survival": "N/A"
    },
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


# ═══════════════════════════════════════════════════════
# SHARP EXTENSION SPECS IMPLEMENTATION
# Standardized Healthcare Agent Resource Propagation
# Reads FHIR context headers automatically injected by
# Prompt Opinion when a patient context is active.
# ═══════════════════════════════════════════════════════
def read_sharp_context(ctx: Context) -> dict:
    """
    Read SHARP Extension Spec headers from Prompt Opinion.
    Headers: fhirUrl, fhirToken, patientId
    """
    sharp = {
        "fhirUrl": "",
        "fhirToken": "",
        "patientId": "",
        "has_context": False,
        "error": None
    }

    if not ctx:
        return sharp

    try:
        headers = ctx.request_context.request.headers

        sharp["fhirUrl"] = headers.get("fhirUrl", "")
        sharp["fhirToken"] = headers.get("fhirToken", "")
        sharp["patientId"] = headers.get("patientId", "")

        # Also check X- prefix variants
        if not sharp["fhirUrl"]:
            sharp["fhirUrl"] = headers.get("X-FHIR-Server-URL", "")
        if not sharp["fhirToken"]:
            sharp["fhirToken"] = headers.get("X-FHIR-Access-Token", "")
        if not sharp["patientId"]:
            sharp["patientId"] = headers.get("X-Patient-ID", "")

        if sharp["fhirUrl"] and not sharp["fhirToken"]:
            sharp["error"] = "SHARP_TOKEN_MISSING"

        sharp["has_context"] = bool(sharp["patientId"] or sharp["fhirUrl"])

    except Exception as e:
        sharp["error"] = f"SHARP_CONTEXT_READ_ERROR: {str(e)}"

    return sharp


# ─────────────────────────────────────────────
# GUARD: Detect wrong input type
# Prevents agent from passing FHIR JSON or
# document text as image_data by mistake
# ─────────────────────────────────────────────
def detect_wrong_input(image_data: str) -> dict | None:
    """
    Detects if the agent accidentally passed a JSON document
    or plain text instead of a base64-encoded image.
    This happens when agents confuse patient documents with images.
    Returns an error dict if wrong input detected, None if OK.
    """
    stripped = image_data.strip()

    # Check if it looks like JSON (FHIR document, etc.)
    if stripped.startswith('{') or stripped.startswith('['):
        return {
            "error": "WRONG_INPUT_TYPE",
            "message": (
                "Received JSON/text data instead of a base64-encoded image. "
                "This tool requires an actual MRI image file, not a FHIR document or text. "
                "Please provide a base64-encoded JPG or PNG image."
            ),
            "hint": (
                "If you are trying to read a patient's existing diagnostic report, "
                "use the patient's uploaded FHIR JSON document directly. "
                "To analyze a NEW MRI scan, provide the image as base64."
            ),
            "correct_usage": "image_data should be base64-encoded image bytes, not JSON text"
        }

    # Check if it's plain text (too short or readable text)
    if len(stripped) < 100 and not stripped.replace('+', '').replace('/', '').replace('=', '').isalnum():
        return {
            "error": "WRONG_INPUT_TYPE",
            "message": (
                "Input appears to be plain text rather than a base64-encoded image. "
                "Please provide a valid base64-encoded MRI image (JPG or PNG)."
            ),
            "hint": "Use the web interface at vendhal.github.io/Brain-Tumor-Classifier to generate valid base64 image data."
        }

    return None


# ─────────────────────────────────────────────
# CAPABILITIES PATCH
# ─────────────────────────────────────────────
_original_get_capabilities = mcp._mcp_server.get_capabilities


def _patched_get_capabilities(notification_options, experimental_capabilities):
    caps = _original_get_capabilities(notification_options, experimental_capabilities)
    caps.model_extra["extensions"] = {
        "ai.promptopinion/fhir-context": {
            "scopes": []
        }
    }
    return caps


mcp._mcp_server.get_capabilities = _patched_get_capabilities


# ─────────────────────────────────────────────
# TOOL 1: analyze_mri
# ─────────────────────────────────────────────
@mcp.tool(
    name="analyze_mri",
    description=(
        "Performs 9-class neuro-symbolic brain tumor classification from MRI scan. "
        "Returns a FHIR R4 DiagnosticReport with neural confidence scores, symbolic "
        "clinical reasoning traces (3 rules: confidence threshold, severity assessment, "
        "ambiguity detection), uncertainty flags, and clinical recommendations. "
        "SHARP-on-MCP compliant: automatically reads fhirUrl, fhirToken, and patientId "
        "from SHARP context headers when available in Prompt Opinion workspace. "
        "Input must be a base64-encoded image file (JPG/PNG), NOT a FHIR document or JSON text."
    )
)
async def analyze_mri(
        image_data: Annotated[str, Field(
            description=(
                "Base64-encoded MRI image (JPG, PNG, or JPEG format). "
                "Must be actual image binary data encoded as base64. "
                "Do NOT pass FHIR JSON, patient documents, or plain text here."
            )
        )],
        patient_reference: Annotated[str, Field(
            description="FHIR patient reference e.g. 'Patient/12345'. "
                        "Auto-populated from SHARP patientId header if available."
        )] = "Patient/unknown",
        ctx: Context = None
) -> str:
    """Neuro-symbolic brain MRI analysis. SHARP-on-MCP compliant."""

    # ── Read SHARP context ──
    sharp = read_sharp_context(ctx)

    # Auto-populate patient_reference from SHARP patientId
    if sharp["patientId"] and patient_reference == "Patient/unknown":
        patient_reference = f"Patient/{sharp['patientId']}"

    # ── Strip data URI prefix if present ──
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # ── Guard: detect wrong input type ──
    wrong = detect_wrong_input(image_data)
    if wrong:
        return json.dumps(wrong, indent=2)

    # ── Fix base64 padding ──
    image_data = image_data.strip()
    padding = 4 - len(image_data) % 4
    if padding != 4:
        image_data += "=" * padding

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return json.dumps({
            "error": "IMAGE_DECODE_FAILED",
            "message": f"Could not decode image: {str(e)}",
            "hint": "Ensure you are providing a valid base64-encoded JPG or PNG image."
        }, indent=2)

    # ── Preprocess ──
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(image).unsqueeze(0)
    tensor = apply_fft(tensor)
    tensor = tensor.to(DEVICE)

    # ── Neural prediction ──
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_cls = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        locked_t = float(thresholds.get(pred_idx, 0.5))
        uncertain = bool(conf < locked_t)

    info = CLASS_INFO.get(pred_cls, {"display": pred_cls, "severity": "unknown"})

    # ── Symbolic reasoning trace ──
    sorted_probs = sorted([float(p) for p in probs], reverse=True)
    top2_diff = float(sorted_probs[0] - sorted_probs[1])

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

    # ── Build FHIR R4 DiagnosticReport ──
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
        "subject": {"reference": patient_reference},
        "issued": datetime.now().isoformat(),
        "conclusion": "Uncertain - refer to specialist" if uncertain else info["display"] + " detected",
        "conclusionCode": [{"text": info["display"]}],
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
                {"url": "symbolicReasoning", "valueString": json.dumps(reasoning)},
                {"url": "uncertaintyFlag", "valueBoolean": uncertain},
                {
                    "url": "recommendedAction",
                    "valueString": "Refer to radiologist" if uncertain else "Clinical correlation advised"
                },
                {
                    "url": "sharpContext",
                    "valueString": json.dumps({
                        "sharp_compliant": True,
                        "patient_from_sharp_context": sharp["has_context"],
                        "fhir_server": sharp["fhirUrl"] if sharp["fhirUrl"] else "standalone_mode",
                        "sharp_error": sharp["error"]
                    })
                }
            ]
        }]
    }

    return json.dumps(fhir_report, indent=2)


# ─────────────────────────────────────────────
# TOOL 2: get_tumor_info
# ─────────────────────────────────────────────
@mcp.tool(
    name="get_tumor_info",
    description=(
        "Get detailed clinical information about a specific brain tumor type. "
        "Returns description, common symptoms, standard treatment protocol, "
        "clinical urgency level, recommended timeframe to see a specialist, "
        "and 5-year survival statistics. Use after analyze_mri to explain "
        "findings to clinicians or patients. SHARP-on-MCP compliant."
    )
)
async def get_tumor_info(
        tumor_class: Annotated[str, Field(
            description="Tumor class name. One of: astrocytoma, ependymoma, glioma, "
                        "meningioma, neurocytoma, oligodendroglioma, schwannoma, "
                        "hemangiopericytoma, normal"
        )],
        ctx: Context = None
) -> str:
    """Get clinical info about a tumor type. SHARP-on-MCP compliant."""

    sharp = read_sharp_context(ctx)
    tumor_class = tumor_class.lower().strip()

    if tumor_class not in CLASS_INFO:
        for key in CLASS_INFO:
            if key in tumor_class or tumor_class in key:
                tumor_class = key
                break
        else:
            return json.dumps({
                "error": f"Unknown tumor class: {tumor_class}",
                "available_classes": list(CLASS_INFO.keys())
            }, indent=2)

    info = CLASS_INFO[tumor_class]

    return json.dumps({
        "tumor_class": tumor_class,
        "display_name": info["display"],
        "severity_level": info["severity"].upper(),
        "description": info["description"],
        "common_symptoms": info["symptoms"],
        "standard_treatment": info["treatment"],
        "clinical_urgency": info["urgency"],
        "recommended_timeframe": info["timeframe"],
        "five_year_survival_rate": info["five_year_survival"],
        "sharp_context_active": sharp["has_context"],
        "disclaimer": (
            "This information is for clinical reference only. "
            "Treatment decisions must be made by qualified healthcare professionals."
        )
    }, indent=2)


# ─────────────────────────────────────────────
# TOOL 3: list_tumor_classes
# ─────────────────────────────────────────────
@mcp.tool(
    name="list_tumor_classes",
    description=(
        "List all 9 brain tumor types that this classifier can detect, "
        "with severity levels, urgency levels, and recommended timeframes. "
        "No arguments required. Use before analyze_mri to understand the "
        "classifier scope. SHARP-on-MCP compliant."
    )
)
async def list_tumor_classes(ctx: Context = None) -> str:
    """List all supported tumor classes. SHARP-on-MCP compliant."""

    classes = []
    for cls_key, info in CLASS_INFO.items():
        classes.append({
            "class_id": cls_key,
            "display_name": info["display"],
            "severity": info["severity"].upper(),
            "urgency": info["urgency"],
            "timeframe": info["timeframe"]
        })

    return json.dumps({
        "total_classes": len(classes),
        "classifier_name": "Neuro-Symbolic Brain Tumor Classifier",
        "model_architecture": "FFT Preprocessing + CGAN Augmentation + MobileNetV2 + Temperature Scaling",
        "approach": "Hybrid Neuro-Symbolic AI",
        "fhir_output": "FHIR R4 DiagnosticReport",
        "sharp_compliant": True,
        "extension_url": "http://ai-diagnostics.org/fhir/neurosymbolic",
        "tumor_classes": classes,
        "severity_guide": {
            "HIGH": "Malignant tumors requiring urgent specialist referral",
            "MEDIUM": "Tumors requiring timely evaluation, often treatable",
            "LOW": "Usually benign, monitoring or elective treatment",
            "NONE": "No tumor detected"
        }
    }, indent=2)


# ─────────────────────────────────────────────
# TOOL 4: validate_mri_image
# ─────────────────────────────────────────────
@mcp.tool(
    name="validate_mri_image",
    description=(
        "Validate whether an uploaded image is a suitable brain MRI scan before analysis. "
        "Checks image format, dimensions, quality indicators, brightness, contrast, and "
        "grayscale characteristics. Returns quality score (0-100), issues, and warnings. "
        "Run before analyze_mri to catch unsuitable images early. "
        "Input must be base64-encoded image data, NOT a FHIR document or JSON text."
    )
)
async def validate_mri_image(
        image_data: Annotated[str, Field(
            description=(
                "Base64-encoded image to validate (JPG, PNG, or JPEG). "
                "Must be actual image data, not a FHIR JSON document."
            )
        )],
        ctx: Context = None
) -> str:
    """Validate MRI image quality before analysis."""

    # Strip data URI prefix
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Guard: detect wrong input type
    wrong = detect_wrong_input(image_data)
    if wrong:
        return json.dumps(wrong, indent=2)

    try:
        image_data = image_data.strip()
        padding = 4 - len(image_data) % 4
        if padding != 4:
            image_data += "=" * padding

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        width, height = image.size
        mode = image.mode
        file_size_kb = round(len(image_bytes) / 1024, 2)

        import numpy as np
        gray = image.convert("L")
        gray_array = np.array(gray)

        mean_brightness = float(np.mean(gray_array))
        std_brightness = float(np.std(gray_array))
        dark_ratio = float(np.sum(gray_array < 30) / gray_array.size)

        issues = []
        warnings = []

        if width < 64 or height < 64:
            issues.append("Image too small (minimum 64x64 pixels)")
        if width > 4096 or height > 4096:
            warnings.append("Very large image - will be resized to 224x224 for analysis")
        if file_size_kb < 5:
            warnings.append("Very small file size - image may be low quality")
        if mean_brightness > 220:
            issues.append("Image appears overexposed (too bright for MRI)")
        if std_brightness < 10:
            issues.append("Very low contrast - may not be a valid MRI scan")
        if dark_ratio < 0.1:
            warnings.append("Low dark pixel ratio - typical MRI scans have significant dark areas")

        score = 100
        score -= len(issues) * 30
        score -= len(warnings) * 10
        if dark_ratio > 0.3:
            score += 10
        if 50 < mean_brightness < 180:
            score += 5
        score = max(0, min(100, score))

        is_valid = len(issues) == 0 and score >= 50

        return json.dumps({
            "is_valid": is_valid,
            "quality_score": score,
            "quality_label": "GOOD" if score >= 70 else "ACCEPTABLE" if score >= 50 else "POOR",
            "recommendation": "Safe to analyze with analyze_mri" if is_valid else "Consider a higher quality MRI image",
            "image_properties": {
                "width_px": width,
                "height_px": height,
                "color_mode": mode,
                "file_size_kb": file_size_kb,
                "mean_brightness": round(mean_brightness, 2),
                "contrast_std": round(std_brightness, 2),
                "dark_pixel_ratio": round(dark_ratio, 3)
            },
            "issues": issues,
            "warnings": warnings
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "is_valid": False,
            "error": f"Could not process image: {str(e)}",
            "recommendation": "Please provide a valid base64-encoded JPG or PNG image."
        }, indent=2)


# ─────────────────────────────────────────────
# TOOL 5: assess_urgency
# ─────────────────────────────────────────────
@mcp.tool(
    name="assess_urgency",
    description=(
        "Assess clinical urgency from a FHIR DiagnosticReport generated by analyze_mri. "
        "Returns urgency level (IMMEDIATE/URGENT/ROUTINE), recommended timeframe, "
        "red flags detected, and next clinical steps. Accepts patient age and symptoms "
        "for context-aware urgency escalation. SHARP-on-MCP compliant: reads patientId "
        "from SHARP context headers for seamless multi-agent workflow integration."
    )
)
async def assess_urgency(
        fhir_report_json: Annotated[str, Field(
            description="FHIR DiagnosticReport JSON string output from analyze_mri tool"
        )],
        patient_age: Annotated[int, Field(
            description="Patient age in years. Pediatric (<18) and elderly (>65) patients "
                        "with high-severity tumors receive automatic urgency escalation.",
            ge=0, le=120
        )] = 0,
        additional_symptoms: Annotated[str, Field(
            description="Comma-separated symptoms e.g. 'seizures, vision loss, headaches'. "
                        "Critical symptoms trigger IMMEDIATE urgency escalation."
        )] = "",
        ctx: Context = None
) -> str:
    """Context-aware urgency assessment. SHARP-on-MCP compliant."""

    sharp = read_sharp_context(ctx)

    try:
        report = json.loads(fhir_report_json)
    except json.JSONDecodeError:
        return json.dumps({
            "error": "INVALID_FHIR_REPORT",
            "message": "Invalid FHIR report JSON. Please provide output directly from analyze_mri tool."
        }, indent=2)

    status = report.get("status", "unknown")
    extensions = report.get("extension", [{}])[0].get("extension", [])
    neural_output = {}
    symbolic_reasoning = {}
    uncertain = False

    for ext in extensions:
        if ext.get("url") == "neuralOutput":
            neural_output = json.loads(ext.get("valueString", "{}"))
        elif ext.get("url") == "symbolicReasoning":
            symbolic_reasoning = json.loads(ext.get("valueString", "{}"))
        elif ext.get("url") == "uncertaintyFlag":
            uncertain = ext.get("valueBoolean", False)

    pred_cls = neural_output.get("topPrediction", "unknown")
    confidence = float(neural_output.get("confidence", 0))
    ambiguity = symbolic_reasoning.get("rule3_ambiguity", {}).get("result", "CLEAR")

    info = CLASS_INFO.get(pred_cls, {})
    base_urgency = info.get("urgency", "ROUTINE")
    base_timeframe = info.get("timeframe", "See specialist as clinically indicated")
    severity = info.get("severity", "none")

    red_flags = []
    urgency_upgrade = False

    if uncertain:
        red_flags.append("AI confidence below clinical threshold - radiologist review required")
        urgency_upgrade = True
    if ambiguity == "AMBIGUOUS":
        red_flags.append("Ambiguous classification - differential diagnosis needed")
        urgency_upgrade = True
    if confidence < 0.4:
        red_flags.append(f"Low confidence ({round(confidence * 100, 1)}%) - second opinion recommended")
    if patient_age > 0:
        if patient_age > 65 and severity == "high":
            red_flags.append(f"Elderly patient (age {patient_age}) with high-severity tumor - expedited referral")
            urgency_upgrade = True
        if patient_age < 18 and pred_cls != "normal":
            red_flags.append(f"Pediatric patient ({patient_age}y) - pediatric neuro-oncology referral required")
            urgency_upgrade = True

    if additional_symptoms:
        symptom_lower = additional_symptoms.lower()
        for s in ["seizure", "vomiting", "vision loss", "paralysis", "unconscious", "stroke"]:
            if s in symptom_lower:
                red_flags.append(f"Critical symptom reported: {s} - emergency evaluation may be needed")
                base_urgency = "IMMEDIATE"
                urgency_upgrade = False
                break

    if urgency_upgrade and base_urgency == "ROUTINE":
        base_urgency = "URGENT"
        base_timeframe = "See specialist within 1-2 weeks (upgraded due to uncertainty)"
    elif urgency_upgrade and base_urgency == "URGENT":
        base_urgency = "IMMEDIATE"
        base_timeframe = "Emergency referral recommended within 24-48 hours"

    next_steps = {
        "IMMEDIATE": [
            "Contact neurosurgeon or neuro-oncologist immediately",
            "Consider emergency MRI with contrast if not already done",
            "Admit to hospital if neurological symptoms present",
            "Begin dexamethasone if significant edema suspected"
        ],
        "URGENT": [
            "Refer to neurology or neuro-oncology within 1-2 weeks",
            "Order contrast-enhanced MRI for better characterization",
            "Review full neurological history",
            "Consider biopsy planning"
        ],
        "ROUTINE": [
            "Schedule outpatient neurology consultation",
            "Routine follow-up MRI in 3-6 months if asymptomatic",
            "Monitor for symptom progression",
            "Patient education about warning signs"
        ]
    }.get(base_urgency, [])

    return json.dumps({
        "urgency_level": base_urgency,
        "urgency_color": {"IMMEDIATE": "RED", "URGENT": "ORANGE", "ROUTINE": "GREEN"}.get(base_urgency, "GREY"),
        "recommended_timeframe": base_timeframe,
        "tumor_detected": info.get("display", pred_cls),
        "confidence": f"{round(confidence * 100, 1)}%",
        "report_status": status,
        "red_flags": red_flags,
        "next_steps": next_steps,
        "patient_age_considered": patient_age if patient_age > 0 else "Not provided",
        "additional_symptoms_considered": additional_symptoms if additional_symptoms else "None provided",
        "sharp_patient_context": sharp["patientId"] if sharp["patientId"] else "Not provided via SHARP",
        "disclaimer": "This urgency assessment is AI-generated and must be reviewed by a qualified clinician."
    }, indent=2)


# ─────────────────────────────────────────────
# TOOL 6: generate_clinical_summary
# ─────────────────────────────────────────────
@mcp.tool(
    name="generate_clinical_summary",
    description=(
        "Generate a plain-English clinical summary from a FHIR DiagnosticReport "
        "produced by analyze_mri. Converts technical FHIR output and symbolic reasoning "
        "traces into readable summaries. Supports two formats: "
        "'clinical' (structured SOAP-style for doctors, includes differential diagnosis "
        "and symbolic reasoning summary) and "
        "'patient' (simplified language for patients explaining findings and next steps). "
        "SHARP-on-MCP compliant."
    )
)
async def generate_clinical_summary(
        fhir_report_json: Annotated[str, Field(
            description="FHIR DiagnosticReport JSON string from analyze_mri tool"
        )],
        format: Annotated[str, Field(
            description="Summary format: 'clinical' for medical professionals (default), "
                        "'patient' for simplified patient-facing language"
        )] = "clinical",
        ctx: Context = None
) -> str:
    """Generate readable clinical summary. SHARP-on-MCP compliant."""

    try:
        report = json.loads(fhir_report_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid FHIR report JSON."}, indent=2)

    report_id = report.get("id", "unknown")
    status = report.get("status", "unknown")
    issued = report.get("issued", "")[:10]
    patient_ref = report.get("subject", {}).get("reference", "Unknown patient")

    extensions = report.get("extension", [{}])[0].get("extension", [])
    neural_output = {}
    symbolic_reasoning = {}
    uncertain = False
    recommended_action = ""

    for ext in extensions:
        if ext.get("url") == "neuralOutput":
            neural_output = json.loads(ext.get("valueString", "{}"))
        elif ext.get("url") == "symbolicReasoning":
            symbolic_reasoning = json.loads(ext.get("valueString", "{}"))
        elif ext.get("url") == "uncertaintyFlag":
            uncertain = ext.get("valueBoolean", False)
        elif ext.get("url") == "recommendedAction":
            recommended_action = ext.get("valueString", "")

    pred_cls = neural_output.get("topPrediction", "unknown")
    confidence = float(neural_output.get("confidence", 0))
    probs = neural_output.get("probabilities", [])
    info = CLASS_INFO.get(pred_cls, {})

    class_probs = [(CLASSES[i], float(probs[i])) for i in range(len(probs))] if probs else []
    class_probs.sort(key=lambda x: x[1], reverse=True)
    top3 = class_probs[:3]

    ambiguity = symbolic_reasoning.get("rule3_ambiguity", {}).get("result", "CLEAR")
    severity = info.get("severity", "unknown")

    if format.lower() == "patient":
        if pred_cls == "normal":
            finding = "Your MRI scan does not show any signs of a brain tumor."
            next_step = "Continue with routine follow-up as recommended by your doctor."
        elif uncertain:
            finding = "Your MRI scan shows some unusual features, but our AI system is not certain about the exact nature."
            next_step = "Your doctor will need to review these results and may recommend additional tests."
        else:
            finding = f"Your MRI scan shows features that may indicate {info.get('display', pred_cls)}."
            next_step = f"Please speak with your doctor as soon as possible. {info.get('timeframe', '')}"

        summary = {
            "summary_type": "Patient-Friendly Summary",
            "report_date": issued,
            "finding": finding,
            "confidence_note": f"The AI system is {'uncertain' if uncertain else str(round(confidence*100,1))+'% confident'} in this finding.",
            "what_this_means": info.get("description", ""),
            "common_symptoms": info.get("symptoms", []),
            "next_step": next_step,
            "important_note": (
                "This is an AI-assisted analysis designed to support (not replace) clinical judgment. "
                "Your doctor will review these results and discuss them with you. "
                "Do not make medical decisions based on this report alone."
            )
        }
    else:
        uncertainty_note = ""
        if uncertain:
            uncertainty_note = f"UNCERTAIN: Confidence ({round(confidence*100,1)}%) below threshold. Radiologist review recommended."
        if ambiguity == "AMBIGUOUS":
            uncertainty_note += " Ambiguous differential - consider contrast-enhanced MRI."

        differential = [
            f"{CLASS_INFO.get(cls, {}).get('display', cls)} ({round(prob*100,1)}%)"
            for cls, prob in top3
        ]

        summary = {
            "summary_type": "Clinical Summary",
            "report_id": report_id,
            "patient_reference": patient_ref,
            "report_date": issued,
            "report_status": status.upper(),
            "primary_finding": f"{info.get('display', pred_cls)} (confidence: {round(confidence*100,1)}%)",
            "severity_classification": severity.upper(),
            "differential_diagnosis": differential,
            "uncertainty_flags": uncertainty_note if uncertainty_note else "None",
            "symbolic_reasoning_summary": {
                "confidence_check": symbolic_reasoning.get("rule1_threshold", {}).get("result", ""),
                "severity_priority": symbolic_reasoning.get("rule2_severity", {}).get("priority", ""),
                "ambiguity_status": ambiguity
            },
            "recommended_action": recommended_action,
            "clinical_notes": [
                f"Primary diagnosis: {info.get('display', pred_cls)}",
                f"Standard treatment: {info.get('treatment', 'See specialist')}",
                f"Typical urgency: {info.get('urgency', 'ROUTINE')} - {info.get('timeframe', '')}",
                f"5-year survival reference: {info.get('five_year_survival', 'N/A')}"
            ],
            "disclaimer": (
                "AI-generated report. This tool is designed to assist (not replace) qualified clinicians. "
                "Must be reviewed and confirmed by a radiologist or neuro-oncologist before clinical use."
            )
        }

    return json.dumps(summary, indent=2)