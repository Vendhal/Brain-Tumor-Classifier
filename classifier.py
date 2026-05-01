import torch
import torch.nn as nn
import torchvision.models as models
from preprocess import NUM_CLASSES

# ── Classifier ────────────────────────────────────────────
class MRIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Adapt first conv: RGB (3ch) → Grayscale (1ch)
        self.base.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Stronger classifier head — Dropout 0.5 + hidden layer
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.5),                              # was 0.3 — stronger regularization
            nn.Linear(self.base.last_channel, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.base(x)


# ── Temperature Scaler ────────────────────────────────────
# Applied POST training — fixes overconfident softmax outputs
# (fixes the 23.4% pituitary bleed you saw in the app)
class TemperatureScaler(nn.Module):
    def __init__(self, model, temperature=1.5):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        return self.model(x) / self.temperature

    def calibrate(self, val_loader, device):
        """Find optimal temperature T on validation set using NLL loss."""
        self.to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll_criterion = nn.CrossEntropyLoss()

        all_logits, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                logits = self.model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        def eval_fn():
            optimizer.zero_grad()
            loss = nll_criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        print(f"  Calibrated temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


# ── Confidence Weighted Focal Loss ────────────────────────
# Per-class weights: harder/rarer classes get higher weight
# astrocytoma=2.0, ependymoma=2.0, glioma=1.0, hemangiopericytoma=2.5,
# meningioma=1.0, neurocytoma=2.0, normal=1.0, oligodendroglioma=2.0, schwannoma=1.5
CLASS_WEIGHTS = torch.tensor([2.0, 2.0, 1.0, 2.5, 1.0, 2.0, 1.0, 2.0, 1.5])

class ConfidenceWeightedLoss(nn.Module):
    def __init__(self, real_weight=1.0, synthetic_weight=0.7, gamma=2.0):
        super().__init__()
        self.real_weight      = real_weight
        self.synthetic_weight = synthetic_weight
        self.gamma            = gamma

    def forward(self, logits, labels, is_synthetic, device):
        cw       = CLASS_WEIGHTS.to(device)
        ce_loss  = nn.CrossEntropyLoss(weight=cw, reduction='none')(logits, labels)

        # Focal scaling — down-weight easy examples
        probs      = torch.softmax(logits, dim=1)
        true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_w    = (1 - true_probs) ** self.gamma
        focal_loss = focal_w * ce_loss

        # Real vs synthetic weighting
        sample_w = torch.where(
            is_synthetic,
            torch.tensor(self.synthetic_weight, device=device),
            torch.tensor(self.real_weight,      device=device)
        )
        return (focal_loss * sample_w).mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MRIClassifier().to(device)
    x      = torch.randn(4, 1, 128, 128).to(device)
    out    = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}  (NUM_CLASSES={NUM_CLASSES})")
    print("✅ Classifier ready!")
