import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────
# Feltrin dataset root — update this to your actual path
DATASET_ROOT = r"C:\Users\saisa.DESKTOP-IRA1I5U\Documents\Mri\Brain Tumor MRI Dataset"

# Feltrin folder names use "TumorType Weighting" format
# We use T1C+ only (contrast-enhanced — best for tumor boundaries)
# Mapping: Feltrin folder name → our class name
FELTRIN_FOLDER_MAP = {
    "Astrocytoma T1C+":        "astrocytoma",
    "Ependymoma T1C+":         "ependymoma",
    "Glioma T1C+":             "glioma",
    "Hemangiopericytoma T1C+": "hemangiopericytoma",
    "Meningioma T1C+":         "meningioma",
    "Neurocytoma T1C+":        "neurocytoma",
    "Normal T1C+":             "normal",
    "Oligodendroglioma T1C+":  "oligodendroglioma",
    "Schwannoma T1C+":         "schwannoma",
    "Other T1C+":              "schwannoma",   # Acoustic Neuroma = Schwannoma, merged
}

CLASSES      = ["astrocytoma", "ependymoma", "glioma", "hemangiopericytoma",
                "meningioma", "neurocytoma", "normal", "oligodendroglioma", "schwannoma"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES  = len(CLASSES)
IMG_SIZE     = 128

# ── Augmentation transforms ─────────────────────────────
TRAIN_AUGMENT = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),   # slightly larger then crop
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

CLEAN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ── FFT transform ───────────────────────────────────────
def apply_fft(img_tensor):
    """Convert spatial image to frequency domain magnitude spectrum."""
    fft       = torch.fft.fft2(img_tensor)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.log1p(torch.abs(fft_shift))
    magnitude = magnitude - magnitude.min()
    magnitude = magnitude / (magnitude.max() + 1e-8)
    magnitude = magnitude * 2 - 1
    return magnitude

# ── Dataset ─────────────────────────────────────────────
class BrainMRIDataset(Dataset):
    """
    Loads Feltrin T1C+ images mapped to 8 classes.
    mode: 'train' (augmented), 'val'/'test' (clean), 'gan' (clean, for GAN training)
    samples: optional list of (path, label) tuples — used for OOF fold splitting
    """
    def __init__(self, root_dir=None, mode="train", samples=None):
        self.mode    = mode
        self.samples = []

        if samples is not None:
            # OOF mode: samples list passed directly
            self.samples = samples
        else:
            # Build from folder structure
            for folder_name, class_name in FELTRIN_FOLDER_MAP.items():
                cls_dir = os.path.join(root_dir, folder_name)
                if not os.path.isdir(cls_dir):
                    print(f"  [WARNING] Folder not found: {cls_dir}")
                    continue
                label = CLASS_TO_IDX[class_name]
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(
                            (os.path.join(cls_dir, fname), label)
                        )

        self.transform = TRAIN_AUGMENT if mode == "train" else CLEAN_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if self.mode in ("train", "val", "test", "gan"):
            img = apply_fft(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# ── Weighted sampler (fixes class imbalance) ─────────────
def get_weighted_sampler(dataset):
    labels      = dataset.get_labels()
    class_count = [labels.count(i) for i in range(NUM_CLASSES)]
    weights     = [1.0 / class_count[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ── All-samples list builder (for OOF splitting) ─────────
def get_all_samples(root_dir):
    """Returns full list of (path, label) tuples from Feltrin dataset."""
    all_samples = []
    for folder_name, class_name in FELTRIN_FOLDER_MAP.items():
        cls_dir = os.path.join(root_dir, folder_name)
        if not os.path.isdir(cls_dir):
            continue
        label = CLASS_TO_IDX[class_name]
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_samples.append((os.path.join(cls_dir, fname), label))
    return all_samples


# ── Class counts ─────────────────────────────────────────
def get_class_counts(root_dir):
    counts = {}
    for folder_name, class_name in FELTRIN_FOLDER_MAP.items():
        cls_dir = os.path.join(root_dir, folder_name)
        if os.path.isdir(cls_dir):
            counts[class_name] = len([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        else:
            counts[class_name] = 0
    return counts


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":
    print("=== Class counts (Feltrin T1C+) ===")
    counts = get_class_counts(DATASET_ROOT)
    total = 0
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} images")
        total += cnt
    print(f"  TOTAL: {total} images")

    print("\n=== Testing dataset (train mode) ===")
    ds = BrainMRIDataset(DATASET_ROOT, mode="train")
    print(f"  Total samples: {len(ds)}")
    img, lbl = ds[0]
    print(f"  Image shape: {img.shape}, Label: {lbl} ({CLASSES[lbl]})")
    print(f"  FFT range: [{img.min():.2f}, {img.max():.2f}]")

    print("\n=== Testing all_samples builder (for OOF) ===")
    all_s = get_all_samples(DATASET_ROOT)
    print(f"  Total samples from get_all_samples: {len(all_s)}")

    print("\n✅ Preprocessing ready for Feltrin 8-class OOF pipeline!")
