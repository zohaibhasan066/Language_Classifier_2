
```markdown
# Language Recognition with ECAPA-TDNN

This project is an implementation of a language recognition system using the ECAPA-TDNN model, typically used for speaker recognition, modified for language classification. The project includes a GUI-based application that allows users to load audio files, extract features, train a language recognition model, and make predictions.

The key steps in this process include:
- **Loading file paths**: Select directories containing audio files and associate them with language labels.
- **Feature extraction**: Extract audio features from MP3 files using the SpeechBrain library.
- **Model training and evaluation**: Train the ECAPA-TDNN model on the extracted features and evaluate its accuracy.
- **Prediction**: Use the trained model to predict the language of new audio files.

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

- **Load File Paths**: Select directories containing MP3 files and associate each directory with a label.
- **Feature Extraction**: Extract audio features using ECAPA-TDNN from the SpeechBrain library.
- **Train and Evaluate Model**: Train a language classification model using extracted features and evaluate the model’s performance on a test set.
- **Prediction**: Predict the language of new audio files using a pre-trained model.

## Prerequisites

Before running the project, ensure that you have Python 3.x and the following dependencies installed:

- `tkinter` for the graphical user interface (GUI).
- `SpeechBrain` for feature extraction and model training.
- `TensorFlow` for building and training the deep learning model.
- `NumPy` for numerical operations.
- `Torchaudio` for audio processing.
- `Scikit-learn` for evaluation and machine learning utilities.

You can install the required libraries using `pip`:

```bash
pip install torchaudio tensorflow numpy scikit-learn speechbrain
```

## Setup and Installation

1. **Clone the repository** to your local machine:

```bash
git clone https://github.com/yourusername/language-recognition.git
cd language-recognition
```

2. **Install the dependencies** by running the following command:

```bash
pip install -r requirements.txt
```

3. **Launch the GUI application**:

```bash
python main.py
```

The GUI will allow you to interact with the system for loading audio files, extracting features, training the model, and making predictions.

## Usage

### Load File Paths

1. Open the application and enter the number of file paths to load.
2. Click the **"Load File Paths"** button to select directories containing MP3 files.
3. For each directory, provide a corresponding language label.

The selected directories and labels will be displayed in a list box.

### Feature Extraction

1. Click the **"Extract Features"** button to start extracting features from the selected MP3 files.
2. Enter the maximum number of files to process from each directory.
3. The application will extract features, and a progress bar will indicate the status.

### Train and Evaluate Model

1. Enter the number of languages you wish to classify.
2. For each language, select the corresponding feature file and label file.
3. The application will train the ECAPA-TDNN model on the extracted features and evaluate the model on a test set.
4. After training, the model's performance (accuracy) will be displayed.

### Prediction

1. Select a pre-trained model and label encoder.
2. Use the trained model to predict the language of new audio data by providing an audio file.
3. The model will return the predicted language label.

## Model Details

This project uses the **ECAPA-TDNN** (Extended Contextualized Attention-augmented Time Delay Neural Network) model, a robust model for speaker recognition, modified for language classification.

The model is trained on audio features (192-dimensional embeddings) extracted from the provided audio files and predicts one of several possible languages based on the audio content. The steps involved in model training are:

- **Feature Extraction**: Audio files are processed into 192-dimensional embeddings using ECAPA-TDNN.
- **Model Training**: The model is trained on these embeddings using deep learning techniques.
- **Evaluation**: The model's performance is evaluated using accuracy metrics on a test set.
- **Prediction**: A trained model is used to predict languages from new audio inputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SpeechBrain** for providing the powerful feature extraction tools.
- **TensorFlow** and **Keras** for the deep learning model implementation.
- **Torchaudio** for handling audio processing tasks.
- The ECAPA-TDNN model, originally designed for speaker recognition, was adapted for this project for language classification tasks.

---
*For more information, feel free to open an issue or contribute to this repository!*
```

### Explanation:

1. **Project Overview**: A brief introduction to what the project does and how it works (language recognition using ECAPA-TDNN).
2. **Table of Contents**: Organized navigation for easy reference.
3. **Features**: A detailed list of what the application can do, such as loading file paths, feature extraction, model training, and prediction.
4. **Prerequisites**: Necessary dependencies to run the project.
5. **Setup and Installation**: Instructions on how to clone the repository and install dependencies.
6. **Usage**: Detailed instructions for interacting with the GUI, including loading file paths, extracting features, training the model, and making predictions.
7. **Model Details**: Information about the ECAPA-TDNN model and how it is used for language classification in this project.
8. **License**: MIT license for the project.
9. **Acknowledgments**: Credits to libraries and tools used in the project.

Ensure that the repository URL, licensing, and any other project-specific information is properly updated before publishing the README.
