# Language_Classifier_2

# Language Recognition with ECAPA-TDNN

This project provides a GUI-based application for language recognition using ECAPA-TDNN, a deep learning model for speaker recognition, modified for language classification. The application allows users to load audio files, extract features, and train a language recognition model. It supports extracting audio features, training the model, evaluating accuracy, and making predictions.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Load File Paths](#load-file-paths)
  - [Feature Extraction](#feature-extraction)
  - [Train and Evaluate Model](#train-and-evaluate-model)
  - [Prediction](#prediction)
- [Model Details](#model-details)
- [License](#license)

## Features

- **Load File Paths:** Select directories containing MP3 files, associate them with labels, and display the selected files in a list.
- **Feature Extraction:** Extract audio features using ECAPA-TDNN from the SpeechBrain library.
- **Train and Evaluate Model:** Train the ECAPA-TDNN model on the extracted features and evaluate its performance.
- **Prediction:** Load a trained model and use it to predict languages from new audio data.

## Prerequisites

Before running the project, ensure the following libraries are installed:

- Python 3.x
- Tkinter
- SpeechBrain
- TensorFlow
- NumPy
- Torchaudio
- Scikit-learn

You can install the required packages using pip:

```bash
pip install torchaudio tensorflow numpy scikit-learn speechbrain
Setup and Installation
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/yourusername/language-recognition.git
cd language-recognition
Install the dependencies:
bash
Copy code
pip install -r requirements.txt
Launch the GUI:
bash
Copy code
python main.py
Usage
Load File Paths
Open the application and enter the number of file paths to load.
Click the "Load File Paths" button to select directories containing MP3 files.
For each directory, provide a corresponding label.
The selected directories and labels will be displayed in a list box.

Feature Extraction
Click the "Extract Features" button to start extracting audio features.
Enter the maximum number of files to process from each directory.
The progress bar will track the extraction process.
Train and Evaluate Model
Enter the number of languages you want to train on.
For each language, select the corresponding feature and label files.
The application will train the ECAPA-TDNN model and evaluate its performance.
The final training, validation, and test accuracies will be displayed in the GUI.

Prediction
Select the trained model and label encoder files.
Use the model to predict the language of new audio data.
Model Details
This project utilizes the ECAPA-TDNN (Extended Contextualized Attention-augmented Time Delay Neural Network) model, originally used for speaker recognition but adapted for language classification in this implementation. The model is trained on extracted features from audio files and evaluated based on accuracy metrics.

The model uses the following architecture:

Input: Extracted features from audio files (192-dimensional embeddings).
Output: Predicted language label.
