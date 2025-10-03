# 📦 MobileNetV2 Quantization Assignment

---

## 📚 Background
This repository contains code to train **MobileNetV2** on the **CIFAR-10** dataset and perform **quantization**.
You can adjust **quantization bits** for both **weights** and **activations**, and evaluate model performance **before** and **after** quantization.

---

## ▶️ Example Usage

- **Train MobileNetV2:**

    python train.py

- **Quantize Model:**

    python test.py --weight_bits 8,6,4 --act_bits 8,6,4

- **Output:**
  - Training logs: `training_log.txt`
  - Model checkpoints: `./checkpoints/`
  - Quantization parallel coordinates plot: `parallel_coordinates.png`

---

## 📝 Assignment Tasks

### ✅ Task 1: Train MobileNetV2
- Train MobileNetV2 on CIFAR-10 / CIFAR-100.
- Logs both training and test accuracy per epoch.
- Saves model checkpoint in `./checkpoints/`.

### ✅ Task 2: Custom Quantization Implementation
- Implement your own quantization code for model compression.
- Configurable quantization bits for **weights** and **activations**.

### ✅ Task 3: Quantization Analysis Report
- Perform weight & activation quantization sweeps.
- Measure and report:
  - Compression ratio of the model
  - Compression ratio of weights
  - Compression ratio of activations
  - Final model size (MB) after quantization
  - Test accuracy after quantization
- Generate **parallel coordinates plot** to visualize trade-offs.

---

## ⚡ Features

1. **Training (Task 1)**
   - SGD optimizer with momentum and cosine annealing LR scheduler.
   - CrossEntropy loss with label smoothing.
   - Logs training/test accuracy and loss per epoch.

2. **Quantization Sweep (Task 2 & 3)**
   - Supports multiple bit-widths for weights and activations.
   - Activation calibration with configurable batch count.
   - Quantized evaluation on test set.
   - Estimates compression ratio vs FP32.

3. **Visualization (Task 3)**
   - Parallel coordinates plot of quantization metrics:
     - ActBits, WeightBits, CompressionRatio, ModelSizeMB, QuantAcc
   - Colors lines by **quantized accuracy**.

---

## 🖥 Steps to Run Code

### 1 Setup Environment
Install dependencies from `requirements.txt`:

    pip install -r requirements.txt
    
### 2 Train MobileNetV2 (Task 1)

    python train.py

- Logs: `training_log.txt`
- Checkpoint: `./checkpoints/test2_mobilenetv2_cifar10.pth`

### 3 Quantize Model (Task 2)

    python test.py --weight_bits 8,6,4 --act_bits 8,6,4 --calib_batches 5

- Sweep through all combinations of weight/activation bits.
- Calibrates activations using a few training batches.
- Freezes quantization parameters for evaluation.

### 4 Analyze & Plot (Task 3)

- Generates a **parallel coordinates plot**: `parallel_coordinates.png`
- Visualizes the relationship between quantization bits, compression, and accuracy.

---

##  Notes
- Accuracy retention after quantization is critical.
- Maximum points awarded for **best compression ratio while maintaining accuracy**.
- Submit **full code + report** on GitHub and share the link on Moodle.

---

##  Evaluation Criteria
- **Functionality:** Working training + quantization code
- **Compression Performance:** Higher compression ratio scores more
- **Accuracy Retention:** Maintain performance post-quantization
- **Code Quality:** Clean, well-documented, organized
- **Analysis:** Comprehensive quantization metrics

---

 **Outcome**
- Trained MobileNetV2 on CIFAR-10
- Custom quantized models evaluated for multiple bit-widths
- Compression ratios and quantized accuracies visualized
