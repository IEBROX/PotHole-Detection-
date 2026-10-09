# Pothole Detection Using YOLO11

A Python-based pothole detection application that uses computer vision and a trained YOLO object detection model to identify potholes in road images and video streams. The project provides a desktop graphical user interface for testing the model with uploaded images, video files, and a live camera feed.

## Overview

Potholes are a common road-safety problem that can damage vehicles and increase the risk of accidents. Manual road inspection is time-consuming and requires continuous human effort.

This project explores an automated approach to pothole detection using deep learning and image processing. A trained YOLO model analyzes visual input and identifies potholes by generating detection bounding boxes and confidence scores.

The application is developed in Python and uses PySide6 for its graphical user interface, OpenCV for image and video processing, and Ultralytics for loading and running the YOLO model.

## Features

- **Image Detection:** Upload a road image and detect potholes.
- **Video Detection:** Process a recorded video and identify potholes in its frames.
- **Live Camera Detection:** Use a connected camera for real-time pothole detection.
- **Detection Visualization:** Display detected potholes using bounding boxes and confidence scores.
- **Model Selection:** Select available YOLO model files from the application.
- **Add Custom Models:** Load additional compatible `.pt` model files through the interface.
- **Detection Results:** View detection information to help evaluate the model's output.
- **Training Results:** Store model training metrics, plots, and evaluation images for analysis.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11 / Ultralytics | Object detection |
| PySide6 | Desktop graphical user interface |
| OpenCV | Image, video, and camera processing |
| PyYAML | Reading YAML configuration files |
| NumPy | Numerical operations used by the image-processing stack |

## Project Structure

```text
PotHole-Detection-/
│
├── app/
│   ├── main.py
│   └── models/
│       ├── best.pt
│       └── best1.pt
│
├── models/
│   ├── best.pt
│   └── best1.pt
│
├── model/
│   ├── best.pt
│   ├── best1.pt
│   └── firstselftrained.pt
│
├── details/
│   ├── args.yaml
│   ├── results.csv
│   ├── results.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── labels.jpg
│   ├── train_batch*.jpg
│   ├── val_batch0_labels.jpg
│   ├── val_batch0_pred.jpg
│   └── weights/
│       ├── best.pt
│       └── last.pt
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Folder and File Description

**`app/`**

Contains the main application code and model files associated with the application.

- `main.py`: Main entry point for the desktop application. It handles the user interface, model selection, and detection workflow.
- `models/`: Contains model weights stored alongside the application code.

**`models/`**

Contains the model weights used by the application. The application searches this directory for `.pt` files and prefers `best.pt` as the default model when it is available.

- `best.pt`: Default model selected by the application.
- `best1.pt`: An additional model available for selection.

**`model/`**

Contains additional model weights and a self-trained model checkpoint. These files are retained separately from the application's default model directory.

**`details/`**

Contains training and evaluation outputs used to analyze model performance.

- `args.yaml`: Training configuration and arguments.
- `results.csv`: Training metrics recorded across epochs.
- `results.png`: Training-results visualization.
- `BoxF1_curve.png`: F1-score curve.
- `BoxPR_curve.png`: Precision-recall curve.
- `BoxP_curve.png`: Precision curve.
- `BoxR_curve.png`: Recall curve.
- `confusion_matrix.png`: Model classification results represented as a confusion matrix.
- `confusion_matrix_normalized.png`: Normalized confusion matrix.
- `labels.jpg`: Visualization of training labels and their distribution.
- `train_batch*.jpg`: Sample training batches and their annotations.
- `val_batch0_labels.jpg`: Ground-truth annotations for a validation batch.
- `val_batch0_pred.jpg`: Model predictions for a validation batch.
- `weights/`: Contains the `best.pt` and `last.pt` checkpoints from training.

**`.gitignore`**

Specifies files and directories that Git should exclude, such as virtual environments and Python cache files.

**`requirements.txt`**

Lists the main Python packages required to install and run the application.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/IEBROX/PotHole-Detection-.git
```

### 2. Navigate to the Project Directory

```bash
cd PotHole-Detection-
```

### 3. Create a Virtual Environment

Using Python's built-in virtual environment module:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** OpenCV camera access and graphical interface support depend on the operating system and available hardware. Linux users running a desktop environment may also need system-level libraries for Qt.

## Running the Application

From the root directory of the repository, execute:

```bash
python app/main.py
```

The application should open its desktop interface if the dependencies are installed correctly and the required model file is present.

### Using the Application

1. Launch the application.
2. Select an available YOLO model or add another compatible `.pt` model.
3. Upload a road image to test still-image detection.
4. Upload a video to process its frames for potholes.
5. Start the camera to test detection using a live video feed.
6. Examine the displayed bounding boxes, confidence scores, and detection results.

The available controls and displayed results depend on the functionality implemented in the current application version.

## Model Information

The application uses trained YOLO object detection weights in `.pt` format. The model files are stored in the repository so that the application can load them locally.

The model's class mapping and training configuration should be checked against the supplied weights before using a different model. A model trained to recognize potholes will not automatically detect unrelated road hazards.

The repository contains multiple checkpoints, including `best.pt`, `best1.pt`, and `firstselftrained.pt`. These files are retained as separate model artifacts; their performance should be compared using consistent test data before deciding which is best.

## Model Evaluation

The `details/` directory contains training and evaluation artifacts that can be used to study the model's behavior.

Important evaluation outputs include:

- **Precision:** The proportion of predicted potholes that are correct.
- **Recall:** The proportion of actual potholes detected by the model.
- **F1-score:** The harmonic mean of precision and recall.
- **Precision-recall curve:** Shows the relationship between precision and recall at different confidence thresholds.
- **Confusion matrix:** Summarizes classification outcomes.

These outputs help evaluate detection performance. Actual accuracy in different weather conditions, low-light environments, and on unseen roads must be established through appropriate testing rather than assumed from training plots alone.

## Hardware Deployment

The project is intended to support further testing on resource-constrained hardware, including Raspberry Pi.

A Raspberry Pi deployment would require checking the compatibility of the selected operating system, Python environment, OpenCV installation, camera, and Ultralytics/PyTorch dependencies. Model inference speed and resource consumption should also be measured on the target device.

**Raspberry Pi deployment and performance are not claimed as verified by this repository alone.** They should be documented after successful hardware testing.

## Future Improvements

- Improve detection performance in low-light conditions and different weather.
- Evaluate and compare the available model checkpoints.
- Optimize inference speed and resource usage.
- Test deployment on Raspberry Pi hardware.
- Add more diverse road images to evaluation.
- Record detection statistics and inference time during testing.

## Limitations

- Detection performance depends on the training data, image quality, lighting, and camera angle.
- False positives and missed potholes may occur.
- Real-time performance depends on the processing hardware and video resolution.
- Results on unseen roads and weather conditions require separate validation.

## Disclaimer

This project is intended for educational and research purposes. It is not a replacement for professional road inspections or a certified road-safety system.

## Author

**IEBROX**

GitHub: https://github.com/IEBROX/PotHole-Detection-
