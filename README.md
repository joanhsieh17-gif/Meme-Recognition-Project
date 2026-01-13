# Meme Recognition System

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

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

## Prerequisites

* **Operating System**: macOS, Linux, or Windows.
* **Python**: Version **3.10, 3.11, or 3.12**.
    * *Note*: Python 3.13 is **not yet supported** due to dependency compatibility.

* **API Key**: A valid [Google Gemini API Key](https://aistudio.google.com/).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/joanhsieh17-gif/Meme-Recognition-Project.git
cd Meme_recognition_NEW
```

### 2. Set Up Virtual Environment

#### `macOS / Linux`
```bash
python -m venv venv
source venv/bin/activate
```

#### `Windows`
We recommend using a specific Python version (e.g., 3.12) to create the environment.
```bash
# If you have the 'py' launcher installed (recommended):
py -3.12 -m venv venv

# Activate the environment:
venv\Scripts\activate
```

### 3. Install Dependencies

#### `macOS / Linux`
First, install `dlib` by following the instructions in [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf). You don't need to build the main dlib library.  
  
Then, install the remaining requirements:

```bash
brew install cmake
pip install -r requirements.txt
```

#### `Windows`
Critical for Windows: Installing dlib from source can be difficult. We highly recommend Option A.

*   Option A: The Easy Way (Pre-compiled Wheels)  Recommended This method avoids installing heavy C++ build tools.

    1. Check your Python version:
    ```bash
    python --version
    # Output example: Python 3.12.x
    ```
    
    2. Download the matching .whl file:
    
    Go to [Murtaza-Saeed/Dlib-Precompiled-Wheels.](https://github.com/Murtaza-Saeed/Dlib-Precompiled-Wheels-for-Python-on-Windows-x64-Easy-Installation)   
    *   If you use Python 3.12 -> Download dlib-19.24.99-cp312-cp312-win_amd64.whl
    *   If you use Python 3.11 -> Download the cp311 version.

    3. Install dlib:
    Place the downloaded .whl file in your project folder and run:

    ```bash
    # Replace with your actual filename
    pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
    ```
    4. Install remaining requirements & fix compatibility:
    ```bash
    # 1. Install standard requirements
    pip install -r requirements.txt
    
    # 2.  FIX: Downgrade Numpy to <2.0 to prevent "Unsupported image type" errors
    pip install "numpy<2"
    
    # 3.  FIX: Resolve OpenCV conflict
    pip install opencv-python-headless==4.10.0.84
    ```

*   Option B: The Hard Way (Build from Source) Only use this if Option A fails. You must compile dlib yourself.

    1. Install Visual Studio Build Tools:
    
        *   Download from Visual Studio Downloads.
    
        *   Run installer and select the "Desktop development with C++" workload.
    
        *   Ensure "MSVC" and "Windows 10/11 SDK" are checked.
    
    2. Install CMake:
    ```bash
    pip install cmake
    ```
    
    3. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    pip install "numpy<2"
    pip install opencv-python-headless==4.10.0.84
    ```


### 4. Configure API Key
#### `macOS / Linux`
```bash
export GEMINI_API_KEY="your_api_key_here"
```

#### `Windows (CMD)`
```bash
set GEMINI_API_KEY="your_api_key_here"
```
#### `Windows (PowerShell)`
```bash
$env:GEMINI_API_KEY="your_api_key_here"
```

## Usage

### 1. Meme Analysis (Batch CLI)

Use `main/main.py` to analyze all images in a directory. This will generate a JSONL result file and a visual HTML report.

```bash
cd main
python main.py --test_dir ../datasets/test_dir_real/
```

**Arguments:**
- `--test_dir`: Path to the directory containing images (Required).
- `--model_path`: Path to the face recognition SVM model (Default: `../models/trained_svm_model_real25.pkl`).
- `--api_key`: (Optional) Pass API key directly if not set in environment.

Results will be saved in:
- `main/results/`: JSONL raw data (includes `cand_details` and `ocr_details`).
- `main/reports/`: HTML reports with Base64 embedded images (can be viewed anywhere).

### 2. Accuracy Calculation

After running the analysis, you can evaluate the accuracy of the predictions against the ground truth (parsed from filenames like `001_name.jpg`).

```bash
cd main
# Process the latest result file:
python accuracy.py

# Or specify a particular result file:
python accuracy.py --results_path results/2023-10-27_10-00-00.jsonl
```

### 3. Jupyter Notebook

For interactive analysis and experimentation:

```bash
jupyter notebook
# Open main/testing_notebook.ipynb
```

### 4. Training Face Recognition Model

To retrain the face recognition model with new data:

1.  Place training images in `datasets/train_dir_real/` (organized by person folders).
```text
datasets/
└── train_dir_real/
    ├── person1/          
    │   ├── 01_person1.jpg
    │   ├── 02_person1.jpg
    │   └── 01_person1.jpg
    └── person2/
        ├── 01_person2.jpg
        └── 02_person2.jpg
  ```

2.  Run the training script:
    ```bash
    cd face_recognition
    python 1.0_train.py
    ```
    This will generate cached encodings in `encoding_caches/` and save the new model to `../models/`.

    Customizing Training Parameters:
    If you want to use different datasets or change where the model is saved, you can modify the following constants directly in `face_recognition/1.0_train.py`:
    
    | Parameter | Default Value | Description |
    | :--- | :--- | :--- |
    | `DEFAULT_TRAIN_DIR` | `../datasets/train_dir_real` | Path to your unzipped training images. |
    | `DEFAULT_MODEL_PATH` | `../models/trained_svm_model_real25.pkl` | Where the final trained model will be saved. |
    | `DEFAULT_CACHE_PATH` | `encoding_caches/train_encodings_cache_real.pkl` | Path for saving face encodings (speeds up retraining). |
    | `DEFAULT_MAX_IMAGES_PER_PERSON` | `25` | Limits the number of images processed per person folder. |
    
    **Example of changing the path in code:**
    ```python
    # 1.0_train.py
    DEFAULT_TRAIN_DIR = "../datasets/my_custom_dataset"
    DEFAULT_MODEL_PATH = "../models/my_new_model.pkl"

## Project Structure

```text
Meme_recognition_NEW/
├── datasets/                     # Training data and Test images
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
