import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ------- Config -------
IMG_FILENAME =r"E:\2024PHD0053\Winter sem\ADVERSARIAL EXAMPLE GENERATION\adv_ex\dog.jpg"       # your original image (in same folder)
ADV_OUT_FILENAME = "adv_image.png"
EPSILON = 0.02                   # small perturbation (try 0.01,0.02,0.05)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------- Model & transforms -------
weights = models.Inception_V3_Weights.IMAGENET1K_V1
model = models.inception_v3(weights=weights).to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# inverse normalization for saving
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# ------- Load image -------
img = Image.open(IMG_FILENAME).convert("RGB")
x = preprocess(img).unsqueeze(0).to(DEVICE)        # shape [1,3,299,299]

# make a copy that requires grad
x_adv = x.clone().detach().requires_grad_(True)

# ------- Forward pass & original prediction -------
with torch.no_grad():
    out = model(x)                                # logits
orig_label = out.argmax(dim=1)                    # original predicted class (tensor)

print("Original class id:", orig_label.item())

# ------- Compute loss w.r.t original label (untargeted FGSM) -------
criterion = nn.CrossEntropyLoss()
# For FGSM untargeted: maximize loss for correct class => backward on loss
# We do NOT use torch.no_grad() so gradients are tracked
out_for_grad = model(x_adv)                       # forward (grad tracked)
loss = criterion(out_for_grad, orig_label)        # loss w.r.t original predicted label
model.zero_grad()
loss.backward()                                   # populates x_adv.grad

# ------- Create adversarial example -------
grad_sign = x_adv.grad.data.sign()
perturbed = x_adv.data + EPSILON * grad_sign
# clamp to valid range of *normalized* tensors: since we clamped in [0,1] later after inv-norm,
# it's safer to clamp in pixel space after inverse-normalizing
# so we'll convert back to image space, clamp, then renormalize when saving

# Convert perturbed tensor to PIL image for saving
# Step 1: undo normalization -> get into [0,1] float range
perturbed_denorm = inv_normalize(perturbed.squeeze()).clamp(0, 1)  # shape [3,299,299]
# Step 2: to CPU numpy
perturbed_np = perturbed_denorm.cpu().permute(1,2,0).numpy()
# Save as PNG
from PIL import Image as PILImage
out_img = (perturbed_np * 255).round().astype(np.uint8)
PILImage.fromarray(out_img).save(ADV_OUT_FILENAME)
print(f"Saved adversarial image to {ADV_OUT_FILENAME}")

# ------- Check adversarial prediction quickly -------
with torch.no_grad():
    # load saved image through preprocess pipeline for exact check
    reloaded = preprocess(PILImage.open(ADV_OUT_FILENAME).convert("RGB")).unsqueeze(0).to(DEVICE)
    out2 = model(reloaded)
    adv_label = out2.argmax(dim=1).item()
print("Adversarial class id:", adv_label)
import matplotlib.pyplot as plt

# Load human-readable labels for ImageNet (optional, for better output)
# This list is available in torchvision
from torchvision.models import Inception_V3_Weights
labels = Inception_V3_Weights.IMAGENET1K_V1.meta["categories"]

# --- Display images ---
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image (denormalize before display)
orig_denorm = inv_normalize(x.squeeze()).clamp(0, 1)
orig_np = orig_denorm.cpu().permute(1, 2, 0).numpy()
axs[0].imshow(orig_np)
axs[0].axis("off")
axs[0].set_title(f"Original:\n{labels[orig_label.item()]} ({orig_label.item()})")

# Adversarial image
axs[1].imshow(perturbed_np)
axs[1].axis("off")
axs[1].set_title(f"Adversarial:\n{labels[adv_label]} ({adv_label})")

plt.tight_layout()
plt.show()
