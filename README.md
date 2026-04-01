# Meme Recognition System

![Python Version](https://img.shields.io/badge/python-3.10-blue)

This project is a tool for analyzing political memes. It combines **Face Recognition**, **OCR**, and **LLM** to identify politicians, extract overlay text, and generate a contextual interpretation of the meme's meaning using Google Gemini.
## Data & Privacy Notice

Due to privacy constraints, this repository does not include original training/testing datasets, pre-trained models, cache files, or specific results. 

To use this project, please provide your own data and ensure they are placed in the following directories:
* `dataset/` - Place your raw images or data here.
* `encoding_caches/` - For stored feature vectors or temporary caches.
* `models/` - For saving or loading your trained model files (e.g., .h5, .pkl).
* `results/` - For output logs, accuracy reports, or prediction images.

## Key Features

*   **Face Recognition**: Identifies known political figures using `face_recognition` with detailed probability scores for each face.
*   **Optical Character Recognition (OCR)**: Extracts text using PaddleOCR with per-word confidence scores and caching.
*   **Context Analysis**: Uses Google Gemini to synthesize visual and textual data into a coherent explanation.
*   **Batch Processing**: Analyze multiple images at once to generate a detailed JSONL dataset and a self-contained HTML report (Base64 embedded images).
*   **Accuracy Evaluation**: Built-in tools to calculate Hit Rate and Top-1 accuracy for both Face Recognition and LLM predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Meme Analysis (Batch CLI)](#1-meme-analysis-batch-cli)
  - [2. Accuracy Calculation](#2-accuracy-calculation)
  - [3. Jupyter Notebook](#3-jupyter-notebook)
  - [4. Training Face Recognition Model](#4-training-face-recognition-model)
- [Project Structure](#project-structure)
- [Project Performance](#project-performance)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

## Project Structure

```text
Meme_recognition_NEW/
├── datasets/                     # Training data and Test images
     ├── train_dir_real
     ├── train_dir_non_real
     ├── test_dir_real
     ├── test_dir_non_real 
├── face_recognition/             # Face recognition scripts
│   ├── 1.0_train.py              # Script to train SVM models
│   ├── 2.0_predict.py            # Standalone prediction test script
│   └── encoding_caches/          # Face encoding caches
├── models/                       # Saved .pkl models
├── main/                         # Core analysis logic
│   ├── main.py                   # CLI entry point for batch analysis
│   ├── MemeAnalyzer.py           # Core analysis engine
│   ├── accuracy.py               # Accuracy calculation script
│   ├── testing_notebook.ipynb    # Interactive notebook
│   ├── results/                  # Batch analysis JSONL output
│   ├── reports/                  # Self-contained HTML reports
│   └── caches/                   # OCR and analysis caches
├── requirements.txt              # Python dependencies
└── README.md
```

## Prerequisites

* **Operating System**: macOS, Linux, or Windows.
* **Python**: Version **3.10**.
* **API Key**: A valid [Google Gemini API Key](https://aistudio.google.com/).

## Installation

### `macOS / Linux`
```bash
# 1.Install system compilation tools
brew install cmake

# 2. Clone the Repository
git clone https://github.com/joanhsieh17-gif/Meme-Recognition-Project.git
cd Meme-Recognition-Project

# 3. Set Up Virtual Environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Install Dependencies
pip install dlib
pip install -r requirements.txt

# 5.Configure API Key
export GEMINI_API_KEY="your_api_key_here"
```

#### `Windows`
```bash
# 1. Clone the Repository
git clone https://github.com/joanhsieh17-gif/Meme-Recognition-Project.git
cd Meme-Recognition-Project

# 2. Set Up Virtual Environment
py -3.10 -m venv venv
venv\Scripts\activate

# 3. Install Dependencies
pip install ./wheels/dlib-19.22.99-cp310-cp310-win_amd64.whl
pip install -r requirements.txt

# 4.Configure API Key
## cmd
set GEMINI_API_KEY="your_api_key_here"

## powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

## Usage

### 1. Put data in training and test dir
Ｗe can train different kinds of models like the model for recognize real images or cartoon images.

1.  Place training images in `datasets/train_dir_real/` (rename your folder)

    Please replace your images by the following structure, using subfolder to categorize different persons.
    
    Make sure to use `pypinyin` format to translate original names, if people's names you want to analyze are not in English. 
```text
datasets/
└── train_dir_real/
    ├── wang_xiao_ming/    # Must use pypinyin format      
    │   ├── 01_wang_xiao_ming.jpg # There is no specific format required for images.
    │   ├── 02_wang_xiao_ming.jpg
    │   └── 03_wang_xiao_ming.jpg
    └── person2/
        ├── 01_person2.jpg
        └── 02_person2.jpg
  ```

2. Place test images in `datasets/test_dir_real/` (rename your folder)
   
   Make sure to use `pypinyin` format to translate original names, if people's names you want to analyze are not in English.
   
   There is no subfolder in test dirs.
```text
datasets/
└── test_dir_real/    
    │   ├── 01_wang_xiao_ming.jpg # Must use pypinyin format
    │   ├── 02_wang_xiao_ming.jpg
    │   └── 03_wang_xiao_ming.jpg
    └── person2/
        ├── 01_person2.jpg
        └── 02_person2.jpg
  ```

### 2. Train the model
1. Remember to set the model and training data if you have different kinds of training data.
   `face_recognition/1.0_train.py`
```python
DEFAULT_CACHE_PATH = "encoding_caches/train_encodings_cache_real.pkl" # Path for saving face encodings (speeds up retraining).
DEFAULT_MODEL_PATH = "../models/trained_svm_model_real25.pkl"         # Where the final trained model will be saved.
DEFAULT_TRAIN_DIR = "../datasets/train_dir_real"                      # Choose the training data to train your model
DEFAULT_MAX_IMAGES_PER_PERSON = 25                                    # Limits the number of images processed per person folder. 
```

2.  Run the training script:
```bash
cd face_recognition
python 1.0_train.py
```
This will generate cached encodings in `encoding_caches/` and save the new model to `../models/`.

### 3. Meme Analysis (Batch CLI)

1. Set the model you have trained to analyze test data.
```python
def main():
    parser = argparse.ArgumentParser(description="Batch Meme Analysis Tool")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing images to analyze")
    # change here -> 'default="../models/trained_svm_model_real25.pkl"' 
    parser.add_argument("--model_path", type=str, default="../models/trained_svm_model_real25.pkl", help="Path to SVM model")
    parser.add_argument("--api_key", type=str, help="Google Gemini API Key (Optional if env var set)")
```


2.  Run the main script:

Set the test data path.

Change here -> ../datasets/test_dir_real/
```bash
cd main
python main.py --test_dir ../datasets/test_dir_real/
```
This will generate a JSONL result file and a visual HTML report.

  Results will be saved in:
- `main/results/`: JSONL raw data (includes `cand_details` and `ocr_details`).
- `main/reports/`: HTML reports with Base64 embedded images (can be viewed anywhere).

### 3. Accuracy Calculation

1. Set the path to save confusion matrices
```python
# --- update：build specific folder to restore confusion matrixs ---
    output_dir = "confusion_matrices/combine"  # set the path to save confusion matrices
    os.makedirs(output_dir, exist_ok=True)
```

2. After running the analysis, you can evaluate the accuracy of the predictions against the ground truth (parsed from filenames like `001_name.jpg`).

```bash
cd main
# Process the latest result file:
python accuracy.py

# Or specify a particular result file:
python accuracy.py --results_path results/2023-10-27_10-00-00.jsonl
```

### 4. Jupyter Notebook

For interactive analysis and experimentation:

```bash
jupyter notebook
# Open main/testing_notebook.ipynb
```



## Project Performance

We evaluate the system using six metrics to distinguish between raw detection capability and prediction precision:

### Metric Definitions
1.  **face_rec_hit**: Counted as correct if the ground truth person appears *anywhere* in the list of face candidates (all names above the threshold).
2.  **face_rec_top**: Counted as correct only if the ground truth person is the *highest probability* match for a detected face.
3.  **llm**: Counted as correct if the Gemini LLM's final prediction matches the ground truth.

**Filter None Settings:**
- **Filter None: False**: Accuracy is calculated across the **entire dataset**. If the model fails to detect anyone, it is counted as a 0% success for that image.
- **Filter None: True**: Accuracy is calculated only on images where the model **actually made a prediction**. This measures the "precision" or reliability of the model when it is confident enough to return a result.

### Test Results
We conducted a test on a total of 167 memes featuring 27 politicians. The results are as follows:
```
------------------------------------------------------------
Type: face_rec_hit    | Filter None: False | Acc: 134/167 = 0.8024
Type: face_rec_hit    | Filter None: True  | Acc: 134/150 = 0.8933
  (Filtered out 17 empty predictions)
------------------------------------------------------------
Type: face_rec_top    | Filter None: False | Acc: 120/167 = 0.7186
Type: face_rec_top    | Filter None: True  | Acc: 120/150 = 0.8000
  (Filtered out 17 empty predictions)
------------------------------------------------------------
Type: llm             | Filter None: False | Acc: 159/167 = 0.9521
Type: llm             | Filter None: True  | Acc: 159/166 = 0.9578
  (Filtered out 1 empty predictions)
------------------------------------------------------------
```
<br>
<p align="center">
  <img src="images_in_readme/Face%20Recognition%20Only%20v.s.%20New%20Design%20Pipeline(Face%20Rec%20+%20OCR%20+LLM).png" width="80%">
  <br>
  <em>Figure: Comparison between Face Recognition Only and New Design Pipeline</em>
</p>

<br>
<br>

<p align="center">
  <img src="images_in_readme/model%20optimization.png" width="80%">
  <br>
  <em>Figure: Model Optimization</em>
</p>

<br>

## Troubleshooting

- **`dlib` Installation**: Ensure `cmake` is installed.
- **PaddleOCR**: If running on Linux, ensure `libgl1` and `libgomp1` are installed.

## Acknowledgments

- **[face_recognition](https://github.com/ageitgey/face_recognition)**
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
- **[Google Gemini](https://ai.google.dev/)**

## Project Maintenance Note

* **Repository Update**: This repository has been refactored and migrated from a previous version to ensure a cleaner project structure and better privacy protection.
* **Commit History**: The original commit history has been reset during the migration process to remove legacy large files and sensitive data.
* **Refactored Version**: This is the current stable version of the project.
