# FGSM Adversarial Attack (Fast Gradient Sign Method)

This project demonstrates the **Fast Gradient Sign Method (FGSM)** for generating adversarial examples against an image classification model.  
We use **Inception v3** pretrained on ImageNet to show how small, imperceptible perturbations can cause a model to misclassify an image.

---

## 📂 Project Structure
```
.
├── fgsm_attack.py       # Main FGSM attack script
├── dog.jpg              # Sample input image
├── adv_image.png        # Generated adversarial image (output)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fgsm-attack-demo.git
cd fgsm-attack-demo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python fgsm_attack.py
```

---

## ⚙️ Parameters
You can edit the following in `fgsm_attack.py`:
- `IMG_FILENAME` → Path to the original image.
- `EPSILON` → Perturbation size (e.g., 0.01, 0.02, 0.05).
- `ADV_OUT_FILENAME` → Output filename for adversarial image.

---

## 📊 Example Output

| Original Image | Adversarial Image |
| -------------- | ----------------- |
| ![Original](dog.jpg) | ![Adversarial](adv_image.png) |

---

## 📚 References
- **FGSM Paper**: [Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2014)](https://arxiv.org/abs/1412.6572)
- **PyTorch Models**: [Torchvision Model Zoo](https://pytorch.org/vision/stable/models.html)

---

## 📜 License
This project is released under the MIT License.
