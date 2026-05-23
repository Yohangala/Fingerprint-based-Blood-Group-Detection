# 🩸 Fingerprint-based Blood Group Detection

A deep learning system that predicts **ABO blood group (A, B, AB, O)** from fingerprint ridge patterns using a fine-tuned **ResNet-152** model with oversampling to handle class imbalance — deployed as a Flask web app.

---

## 📌 Research Context

Emerging studies suggest a correlation between fingerprint patterns (ridge density, minutiae distribution) and ABO blood groups. This project explores that link by training a CNN-based classifier on fingerprint images, offering a non-invasive, rapid alternative to conventional blood typing.

> **Note:** This is a research prototype. Results should not be used for medical decisions.

---

## 🧠 Model Architecture

- **Backbone:** ResNet-152 (pretrained on ImageNet, fine-tuned)
- **Input:** Fingerprint images (grayscale → RGB)
- **Output:** 4-class softmax (A, B, AB, O)
- **Class Imbalance Handling:** Oversampling (augmentation-based) to balance blood group distribution
- **Framework:** PyTorch + torchvision

---

## 📁 Repository Structure

```
Fingerprint-based-Blood-Group-Detection/
├── OVERSAMPLING_RESNET152ipynb.ipynb   # Model training, oversampling & evaluation
├── app.py                              # Flask API for inference
├── index.html                          # Frontend upload UI
├── requirements.txt                    # Dependencies
└── README.md
```

---

## ⚙️ Setup & Run

```bash
pip install -r requirements.txt
# Also install PyTorch:
pip install torch torchvision
python app.py
```

Open `http://localhost:5000` in your browser, upload a fingerprint image, and get a predicted blood group.

---

## 🔬 Pipeline

1. **Data Collection** — Fingerprint image dataset with ABO blood group labels
2. **Preprocessing** — Resize, normalize, convert to 3-channel
3. **Oversampling** — Augment minority classes to balance distribution
4. **Model Training** — Fine-tune ResNet-152 with cross-entropy loss
5. **Evaluation** — Accuracy, confusion matrix, per-class F1
6. **Deployment** — Flask endpoint accepts image upload, returns prediction

---

## 📊 Results

| Metric | Score |
|---|---|
| Model | ResNet-152 (fine-tuned) |
| Classes | A, B, AB, O |
| Oversampling | Yes (augmentation-based) |

> Full metrics available in the training notebook.

---

## 🔗 References

- [He et al. — Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [Gnanasivam & Muttan — Fingerprint Ridge Density for Blood Group and Gender Identification](https://arxiv.org/abs/1209.3286)

---

## 👤 Author

**Yohan Gala** — B.Tech Computer Engineering, KJSIT Mumbai  
[GitHub](https://github.com/Yohangala)
