# Medical Image Classification: Comparative ML Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

A comprehensive comparison of four machine learning approaches for 11-class medical image classification:
- **Support Vector Machine (SVM)** - 94.9% accuracy
- **K-Nearest Neighbors (KNN)** - 93.0% accuracy
- **Convolutional Neural Network (CNN)** - 85.0% accuracy
- **Generative Adversarial Network (GAN)** - 50.6% accuracy

## 🎯 Key Achievements

- ✅ **Best Performance:** 94.9% accuracy with SVM (0.998 AUC)
- ✅ **Feature Engineering:** 357-dimensional feature space (HOG + LBP + Statistics)
- ✅ **Class Imbalance Handling:** GAN-based synthetic data generation (4.7:1 → 1.5:1)
- ✅ **Comprehensive Evaluation:** ROC curves, confusion matrices, per-class metrics
- ✅ **Optimized Hyperparameters:** Grid Search with 3-fold cross-validation

## 📊 Performance Comparison

| Model | Accuracy | AUC | Training Time | Key Feature |
|-------|----------|-----|---------------|-------------|
| **SVM** | 94.9% | 0.998 | 12 min | Near-perfect classification |
| **KNN** | 93.0% | 0.977 | Instant | No training required |
| **CNN** | 85.0% | 0.987 | 5 min | Automatic feature learning |
| **GAN** | 50.6% | 0.892 | 10 min | Synthetic data generation |

## 🛠️ Tech Stack

**Languages & Frameworks:**
- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- OpenCV
- scikit-image

**Libraries:**
- NumPy, Pandas - Data manipulation
- Matplotlib, Seaborn - Visualization
- tqdm - Progress bars
- joblib - Model persistence

## 📁 Project Structure
```
├── notebooks/          # Jupyter notebooks for each approach
├── results/           # Visualizations and metrics
├── models/            # Saved trained models
├── src/              # Source code modules
├── docs/             # Documentation and presentations
└── requirements.txt   # Dependencies
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip or conda package manager
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/medical-image-classification.git
cd medical-image-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place files in appropriate directories

### Usage

Run the Jupyter notebooks:
```bash
jupyter notebook notebooks/01_KNN_Classification.ipynb
jupyter notebook notebooks/02_SVM_Classification.ipynb
jupyter notebook notebooks/03_CNN_Classification.ipynb
jupyter notebook notebooks/04_GAN_Classification.ipynb
```

## 📈 Results

### SVM Results (Best Performance)

**Key Metrics:**
- Accuracy: 94.9%
- Precision: 0.949
- Recall: 0.949
- F1-Score: 0.949
- AUC: 0.998

### Feature Engineering

**357-Dimensional Feature Vector:**
1. **HOG (81 features):** Histogram of Oriented Gradients
   - 9 orientations, 8×8 pixel cells
   - Shape and edge information

2. **LBP (235 features):** Local Binary Patterns
   - Radius 2, 16 points
   - Texture information

3. **Statistical (41 features):**
   - Mean, std, median, quartiles
   - Skewness, kurtosis, 32-bin histogram

## 🔬 Methodology

### 1. Data Preprocessing
- Image resizing: 32×32 pixels
- Normalization: 0-1 range
- Class distribution analysis

### 2. Feature Extraction
- HOG for shape features
- LBP for texture features
- Statistical features for intensity

### 3. Model Training
- **SVM:** RBF kernel with Grid Search
- **KNN:** K=5, distance-weighted
- **CNN:** 3-block architecture with dropout
- **GAN:** 20 epochs adversarial training

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC curves and AUC
- Confusion matrices
- Per-class performance analysis

## 💡 Key Insights

1. **Feature Engineering Matters:** Hand-crafted features (SVM/KNN) outperformed automatic learning (CNN) in this low-resolution scenario
2. **Class Imbalance Challenge:** GAN successfully generated synthetic data but requires more tuning
3. **SVM's Kernel Trick:** Non-linear separation crucial for medical image classification
4. **Confidence Calibration:** Models appropriately express uncertainty on ambiguous samples

## 🎓 Lessons Learned

- Traditional ML with good features can outperform deep learning on small datasets
- Class imbalance requires multiple strategies (weights, synthetic data)
- 32×32 resolution is limiting for medical imaging
- Model interpretability is crucial for healthcare applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: (https://linkedin.com/in/thisas-jayasooriya-b70398337)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: jayasooriyathisas9@gmail.com
## 🙏 Acknowledgments

- Dataset: [Source/Citation]
- Tools: TensorFlow, scikit-learn, OpenCV teams

---

⭐ If you find this project useful, please consider giving it a star!
