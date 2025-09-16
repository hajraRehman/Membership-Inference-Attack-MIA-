# TML25_A1_13 - Membership Inference Attack

## Task Overview

In this project, we implement a **Membership Inference Attack (MIA)** on a pretrained ResNet18 model. The objective is to determine whether a given image was part of the training dataset used to train the model.

We are given:
- A **pretrained ResNet18 model**
- A **public dataset** with membership labels (1 for members, 0 for non-members)
- A **private dataset** where the membership status is unknown

Our goal is to:
- **Predict continuous membership confidence scores** ∈ [0, 1] for each sample in the private dataset.
- **Submit** these scores to the evaluation server to compute:
  - **TPR@FPR=0.05**
  - **AUC (Area Under ROC Curve)**

---

## Repository Structure

```

.
├── assignment1_code.py     # Main script for running the full attack pipeline
├── submission.csv         # Final submission file (scores for private dataset)
├── 01\_MIA.pt              # Pretrained ResNet18 model checkpoint
├── pub.pt                 # Public dataset with membership labels
├── priv\_out.pt            # Private dataset (membership unknown)
├── README.md              # This file

````

---

##  Methodology

### 1. **Model and Data Loading**
- Loaded the pretrained ResNet18 model (`01_MIA.pt`).
- Loaded public and private datasets (`pub.pt`, `priv_out.pt`).
- Applied dataset normalization using provided channel-wise mean and std.

### 2. **Feature Extraction**
For each image, we passed it through the model and extracted the following features:

| Feature | Description |
|--------|-------------|
| `prob.max()` | Model's confidence in its top prediction |
| `entropy` | Shannon entropy of the softmax output |
| `top-3 probs` | Top 3 softmax probabilities |
| `conf. gaps` | Differences between top-k probabilities |
| `logits mean/std` | Statistical summary of raw logits |
| `is_correct` | Whether model's prediction was correct |
| `predicted label` | Argmax of softmax output |

These features capture model confidence, uncertainty, and behavior — useful for inferring membership.

### 3. **Attack Model Training**
- Trained a **Random Forest classifier** on features from the **public dataset**.
- Used a train/val split (80/20) to evaluate generalization.
- Achieved validation AUC: **~0.65**

### 4. **Prediction & Submission**
- Extracted features for the **private dataset**.
- Predicted membership confidence scores using the trained attack model.
- Formatted predictions as a `.csv` file and submitted via HTTP POST request.

---

## 📈 Results

**Evaluation (Public Set)**:
- **TPR@FPR=0.05**: `0.1183`
- **AUC**: `0.6539`

These results indicate a successful attack well above random guessing (AUC = 0.5), demonstrating leakage from the target model.

---

## 🚀 How to Run

1. Clone this repo and place the required `.pt` files in the root directory.
2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn pandas requests

3. Run the script:

   ```bash
   python assignment1_code.py
   ```

---

## 📤 Submission

* We submitted the results to:

  ```
  http://34.122.51.94:9090/mia
  ```
* Using the official token assigned to **Team 13**.
* Output format:

  ```csv
  ids,score
  1001,0.87
  1002,0.23
  ...
  ```

---

## 📌 Notes & Recommendations

* We ensured **no data leakage** by strictly separating training (public) and test (private) data.
* All predictions are **continuous scores**, not hard labels.
* The model outputs are properly **normalized and clamped** to avoid instability in entropy calculations.
* Future improvement areas: experimenting with XGBoost, more advanced calibration, and ensemble methods.

---

## 👥 Contributors

* Hafiza Hajrah Rehman   hafizahajra6@gmail.com
* Atqa Rabiya Amir amiratqa@gmail.com

---

## 🔗 Resources

* [Assignment Instructions PDF](./01-MIA-TML2025.pdf)
* [Task Presentation Slides](./TML2025-TASK1.pdf)
* [Code Template Repository](https://github.com/sprintml/tml_2025_tasks)

---

## 🏁 Final Submission

* GitHub Tag: `v1.0`
* Submitted on: **May 28, 2025**
* Submission File: `submission.csv`

```


 
