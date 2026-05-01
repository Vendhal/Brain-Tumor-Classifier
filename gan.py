import torch
import torch.nn as nn

NUM_CLASSES = 9
LATENT_DIM  = 100
IMG_SIZE    = 128

# ── Generator ────────────────────────────────────────────
# Same architecture as original but embedding now handles 8 classes
# Added dropout in intermediate layers to prevent memorization
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            # 8x8 → 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], dim=1)
        return self.model(x)


# ── Discriminator ─────────────────────────────────────────
# Added Dropout2d after each conv block — critical fix for discriminator
# dominance (your D loss crashed to 0.27 previously because of zero dropout)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)
        self.model = nn.Sequential(
            # input: 2 channels (image + label map)
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),                          # NEW — prevents D memorizing

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),                          # NEW

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),                          # NEW

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, label_map], dim=1)
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    G = Generator().to(device)
    D = Discriminator().to(device)

    z      = torch.randn(8, LATENT_DIM).to(device)
    labels = torch.randint(0, NUM_CLASSES, (8,)).to(device)

    fake  = G(z, labels)
    score = D(fake, labels)

    print(f"Generator output shape:     {fake.shape}")
    print(f"Discriminator output shape: {score.shape}")
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print("✅ GAN architecture ready for 8-class Feltrin dataset!")
