**This project was created by Zohaib Hasan Siddiqui & Mohammad Anas Azeez.(https://github.com/AnasAzeez)**

### **ECAPA-TDNN Model for Audio Classification**

---

### **Index**
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Applications](#applications)  
4. [Prerequisites](#prerequisites)  
5. [Dataset Requirements](#dataset-requirements)  
6. [Usage](#usage)  
   - [Clone the Repository](#clone-the-repository)  
   - [Install Dependencies](#install-dependencies)  
   - [Prepare Your Dataset](#prepare-your-dataset)  
   - [Train the Model](#train-the-model)  
   - [Evaluate the Model](#evaluate-the-model)  
7. [Configuration](#configuration)  
8. [Model Performance](#model-performance)  
9. [Visualization](#visualization)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Acknowledgments](#acknowledgments)

---

# Overview
This repository provides an implementation of the **ECAPA-TDNN (Emphasized Channel Attention, Propagation, and Aggregation in Time-Delay Neural Networks)** model. Designed for audio classification tasks such as **language identification**, **speaker recognition**, and **speech pattern analysis**, this model leverages state-of-the-art deep learning techniques to achieve robust and efficient feature extraction from audio signals.

---

# Key Features
- **Channel Attention Mechanism**: Enhances the extraction of critical audio features by assigning higher weights to important channels.
- **Res2Net Architecture**: Captures multi-scale features, improving the model's ability to process complex audio patterns.
- **TDNN Layers**: Efficiently extract sequential time-dependent features from audio data.
- **Statistics Pooling**: Aggregates temporal features into fixed-length embeddings for classification.
- **Customizable Framework**: The implementation supports modifications for diverse audio classification tasks.

---

# Applications
- **Language Identification**: Classify audio into different languages with high accuracy.
- **Speaker Recognition**: Identify speakers based on unique audio features.
- **Speech Pattern Analysis**: Analyze audio patterns for research or diagnostic purposes.

---

# Prerequisites
To run this project, ensure you have the following installed:
- **Python 3.8 or later**
- **PyTorch 1.10+**
- **NumPy**
- **Matplotlib** (for visualizations)
- **Librosa** (for audio processing)
- **Scikit-learn** (for evaluation metrics)

---

# Dataset Requirements
The model works with datasets containing preprocessed audio features such as:
- **MFCCs (Mel-Frequency Cepstral Coefficients)**
- **Spectrograms**
- **Raw audio signals (requires preprocessing)**

Ensure your dataset is structured with:
- **Features**: NumPy arrays of shape `(num_samples, feature_dim)`.
- **Labels**: Corresponding class labels in one-hot or integer-encoded format.

---

# Usage

## **Clone the Repository**
```bash
git clone https://github.com/your-repo/ECAPA-TDNN.git
cd ECAPA-TDNN
```

## **Install Dependencies** 
```bash
pip install -r requirements.txt
```

## **Prepare Your Dataset**
- Place your dataset in the `data/` directory.
- Ensure files are in the correct format.

## **Train the Model**
Run the training script:
```bash
python train.py --dataset_path data/ --epochs 50 --batch_size 64
```

## **Evaluate the Model**
Evaluate the model's performance on test data:
```bash
python evaluate.py --model_path models/ecapa_tdnn.pth --test_data data/test/
```

---

### **Configuration**
You can modify hyperparameters and model settings in the configuration file:
- **`config.py`**:
  ```python
  {
      "learning_rate": 0.001,
      "batch_size": 64,
      "num_epochs": 50,
      "feature_dim": 87,
      "num_classes": 6
  }
  ```

---

### **Model Performance**
The model achieves state-of-the-art performance on audio classification benchmarks:
- **Accuracy**: ~96.4% on language identification datasets.
- **Equal Error Rate (EER)**: ~2.3% for speaker recognition.

---

### **Visualization**
The notebook includes loss curves, confusion matrices, and performance metrics to analyze training and evaluation results. To visualize, run:
```bash
jupyter notebook ECAPA-TDNN_Model.ipynb
```

---

### **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for improvements or bug fixes.

---

### **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### **Acknowledgments**
- Inspired by the original ECAPA-TDNN research paper.
- Special thanks to the open-source community for providing tools like PyTorch and Librosa.

```

