# 🧠 Brain Tumor MRI Classifier - MCP Server

> **Neuro-Symbolic AI for brain tumor classification, deployed as an MCP server on Prompt Opinion.**  
> Built for the **Agents Assemble Hackathon 2026** by **Kambhampati Sai Sandeep** And **Maramreddy Sasank**  (KLU)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue)](https://vendhal.github.io/Brain-Tumor-Classifier/)
[![MCP Server](https://img.shields.io/badge/MCP-Prompt%20Opinion-green)](https://app.promptopinion.ai/marketplace)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-purple)](https://brain-tumor-classifier-g48f.onrender.com)

---

## 🌐 Live Test Interface

**Try it instantly — no setup, no download:**

👉 **[https://vendhal.github.io/Brain-Tumor-Classifier/](https://vendhal.github.io/Brain-Tumor-Classifier/)**

Upload any brain MRI image → get a full FHIR DiagnosticReport in seconds.

---

## 🎯 What It Does

The **only medical imaging AI tool** in the Prompt Opinion marketplace.  
While every other submission handles text-based clinical workflows (drug interactions, prior auth, lab results), we tackle the hardest problem: **real-time brain tumor classification from raw MRI scans**.

### Core Pipeline:
```
MRI Image (JPG/PNG)
    ↓
FFT Preprocessing (frequency domain analysis)
    ↓
CGAN Augmentation (synthetic data enrichment)
    ↓
MobileNetV2 Neural Network (9-class classification)
    ↓
Temperature Scaling (calibrated confidence)
    ↓
Symbolic Reasoning Engine (3 clinical rules)
    ↓
FHIR R4 DiagnosticReport (EHR-ready output)
    ↓
MCP Tool Response (agent-consumable)
```

---

## 🛠️ 6 MCP Tools

| Tool | Description |
|------|-------------|
| `analyze_mri` | Classify brain MRI → returns full FHIR DiagnosticReport |
| `get_tumor_info` | Clinical details for any of the 9 tumor classes |
| `list_tumor_classes` | List all detectable tumor types with severity info |
| `validate_mri_image` | Check image quality before analysis |
| `assess_urgency` | Urgency level (IMMEDIATE/URGENT/ROUTINE) from report |
| `generate_clinical_summary` | Plain-English summary (clinical or patient format) |

---

## 🔬 9 Tumor Classes

| Class | Severity | Urgency |
|-------|----------|---------|
| Glioma | HIGH | IMMEDIATE (24-48 hrs) |
| Astrocytoma | HIGH | URGENT (1-2 weeks) |
| Ependymoma | HIGH | URGENT (1-2 weeks) |
| Oligodendroglioma | HIGH | URGENT (1-2 weeks) |
| Hemangiopericytoma | HIGH | URGENT (1-2 weeks) |
| Meningioma | MEDIUM | ROUTINE (4-6 weeks) |
| Neurocytoma | MEDIUM | URGENT (2-4 weeks) |
| Schwannoma | LOW | ROUTINE (4-8 weeks) |
| Normal | NONE | ROUTINE |

---

## 🧩 Neuro-Symbolic Architecture

Unlike pure deep learning approaches, we combine:

**Neural (What):** MobileNetV2 classifies 9 tumor types with calibrated confidence  
**Symbolic (Why):** 3 clinical rules explain every decision:

```
Rule 1 - Confidence Check:    Is the model confident enough for clinical use?
Rule 2 - Severity Assessment: What priority level does this tumor require?
Rule 3 - Ambiguity Detection: Is the differential diagnosis clear or ambiguous?
```

This gives **explainable AI** — not just a prediction, but a **reasoning trace**.

---

## 📄 FHIR-Compliant Output

Every `analyze_mri` call returns a **FHIR R4 DiagnosticReport** with:
- `resourceType`: DiagnosticReport
- `status`: final / preliminary (based on confidence)
- `conclusion`: Human-readable finding
- `result`: All 9 class probabilities
- `extension`: Neural output + symbolic reasoning trace

Ready for direct integration with **Epic**, **Cerner**, or any FHIR-compliant EHR.

---

## 🚀 Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/Vendhal/Brain-Tumor-Classifier.git
cd Brain-Tumor-Classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `index.html` in your browser — or just visit the [live demo](https://vendhal.github.io/Brain-Tumor-Classifier/).

---

## 🔗 MCP Server Endpoint

```
URL: https://brain-tumor-classifier-g48f.onrender.com/mcp
Transport: Streamable HTTP
Auth: Open Access (no API key required)
```

Connect directly in Prompt Opinion → MCP Servers → Add New.

---

## 🗂️ Project Structure

```
Brain-Tumor-Classifier/
│
├── 🧠 Core Server
│   ├── main.py                  # FastAPI app with MCP mount
│   ├── mcp_server.py            # 6 MCP tools (FastMCP)
│   ├── classifier.py            # MobileNetV2 + TemperatureScaler
│   ├── preprocess.py            # FFT preprocessing pipeline
│   ├── gan.py                   # CGAN augmentation
│   └── app.py                   # Alternative entry point
│
├── 🌐 Interface
│   ├── index.html               # Hosted interface (GitHub Pages)
│   └── Test Interface.html      # Original test interface
│
├── 📊 Training Artifacts
│   ├── classifier_curves.png    # Training loss/accuracy curves
│   ├── confusion_matrix.png     # Model evaluation matrix
│   ├── gan_loss_curve.png       # GAN training curve
│   ├── sample_preview.png       # Sample output preview
│   ├── cls_training_log.txt     # Full training logs
│   ├── fid_score.txt            # GAN FID evaluation score
│   └── fs_score.txt             # Additional metrics
│
├── 🔧 Config
│   ├── requirements.txt         # Python dependencies
│   ├── runtime.txt              # Python version (Render)
│   ├── .gitignore
│   └── architecture_diagram.py  # Architecture visualization
│
└── checkpoints/
    ├── classifier_final.pt      # Trained model weights
    ├── scaler_fold1.pt          # Temperature scaler
    └── locked_thresholds.json   # Per-class confidence thresholds
```

---

## 🧪 Testing End-to-End

### Via Live Interface:
1. Visit [https://vendhal.github.io/Brain-Tumor-Classifier/](https://vendhal.github.io/Brain-Tumor-Classifier/)
2. Upload a brain MRI image (JPG/PNG)
3. Click **Analyze MRI**
4. Download the FHIR Report (JSON)
5. Upload to a Prompt Opinion patient record
7. Connect the MCP server in your workspace
1. validate_mri_image  → check quality first
2. analyze_mri         → classify tumor → FHIR report
3. assess_urgency      → how urgent is it?
4. generate_clinical_summary → explain to doctor + patient
5. get_tumor_info      → deep dive on detected class

### Sample Test Images:
Download brain MRI test images from [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Model | MobileNetV2 (PyTorch) |
| Preprocessing | FFT + CGAN (custom) |
| Calibration | Temperature Scaling |
| Reasoning | Custom symbolic rules |
| Output Format | FHIR R4 DiagnosticReport |
| MCP Framework | FastMCP (Python) |
| API | FastAPI + Uvicorn |
| Deployment | Render.com |
| Interface | HTML/JS (GitHub Pages) |

---

## 🔮 Future Enhancements

- **DICOM Support** — Parse raw radiology files directly
- **Radiologist Feedback Loop** — Fine-tune model from corrections
- **Multi-modal Fusion** — Combine CT + MRI for better accuracy
- **Epic/Cerner Integration** — Direct EHR write-back
- **Confidence Calibration v2** — Per-patient threshold adaptation

---

## 🏆 Hackathon

**Agents Assemble Healthcare AI Challenge 2026**  
Organized by: Prompt Opinion  
Team: Kambhampati Sai Sandeep And Maramreddy Sasank — Koneru Lakshmaiah University (KLU)

**Unique value:** Only medical imaging AI in the marketplace. Every other submission handles text workflows. We classify brain tumors from raw MRI scans.

---

## ⚠️ Disclaimer

This tool is for **research and demonstration purposes only**. All AI-generated findings must be reviewed by qualified radiologists and clinicians before any clinical use. Not approved for medical diagnosis.

---

*Powered by MCP + FHIR + PyTorch*