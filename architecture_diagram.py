import torch
from torchviz import make_dot
from gan import Generator, Discriminator, LATENT_DIM, NUM_CLASSES
from classifier import MRIClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator diagram
G = Generator().to(DEVICE)
G.eval()
z     = torch.randn(1, LATENT_DIM).to(DEVICE)
label = torch.randint(0, NUM_CLASSES, (1,)).to(DEVICE)
out   = G(z, label)
dot   = make_dot(out, params=dict(G.named_parameters()))
dot.render("architecture_generator", format="png", cleanup=True)
print("✅ Generator diagram saved.")

# Discriminator diagram
D     = Discriminator().to(DEVICE)
D.eval()
score = D(out, label)
dot   = make_dot(score, params=dict(D.named_parameters()))
dot.render("architecture_discriminator", format="png", cleanup=True)
print("✅ Discriminator diagram saved.")

# Classifier diagram
model = MRIClassifier().to(DEVICE)
model.eval()
x   = torch.randn(1, 1, 128, 128).to(DEVICE)
out = model(x)
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render("architecture_classifier", format="png", cleanup=True)
print("✅ Classifier diagram saved.")

print(f"\nAll architecture diagrams generated. (NUM_CLASSES={NUM_CLASSES})")
