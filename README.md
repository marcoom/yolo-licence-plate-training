
# Car Plate Detection using YOLOv8

An object detection project that uses Ultralytics' YOLOv8 model to detect car plates in images. This repository contains the annotated dataset, training scripts, and instructions to reproduce the results.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

## Introduction

The goal of this project is to develop a robust car plate detection system using the YOLOv8 object detection model. The dataset includes images of cars with annotated license plates. The model is trained to accurately detect and localize car plates in various conditions.

## Dataset

The dataset is organized into three subsets:

- **Training Set** (`train`): Used to train the model.
- **Validation Set** (`valid`): Used to validate the model during training.
- **Test Set** (`test`): Used to evaluate the final model performance.

Each subset contains:

- **Images**: Located in the `images` folder, in `.png` format.
- **Labels**: Corresponding annotations in YOLO format, located in the `labels` folder.

The dataset used to train the model is based on the [Large License Plate Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset)

### Data Annotation

The images were annotated using [LabelImg](https://github.com/HumanSignal/labelImg), an open-source graphical image annotation tool. For installation instructions, please refer to the [LabelImg website](https://github.com/HumanSignal/labelImg).

## Installation

### Prerequisites

- Python 3.8
- Conda package manager
- NVIDIA GPU with CUDA support (optional, but recommended for faster training)

### Steps

1. **Clone the Repository**

     ```bash
   git clone https://github.com/marcoom/yolo-licence-plate-training.git
   cd yolo-licence-plate-training
   ```

2. **Create a Virtual Environment**

     ```bash
   conda create -n yolo_license_plate_training python=3.8
   conda activate yolo_license_plate_training
   ```

3. **Install Dependencies**

     ```bash
   pip install -r requirements.txt
   ```

## Training

### Data Configuration

Ensure the `data.yaml` file is correctly set up. It should contain:

```yaml
path: <path_to_yolo-licence-plate-training>
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 1
names: ['car_plate']
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 30.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.0
  mosaic: 1.0
  mixup: 0.0
  erasing: 0.4
  crop_fraction: 1.0
```

- **path**: Path to the root directory of the dataset.
- **train**, **val**, **test**: Paths to the respective image directories.
- **nc**: Number of classes (in this case, 1 for car plates).
- **names**: List of class names.
- **augmentation**: Augmentation parameters for data augmentation. For an in-depth explanation of each parameter, check https://docs.ultralytics.com/guides/yolo-data-augmentation/

### Training Command

Train the model using the following command:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=300 imgsz=640 pretrained=True augment=True
```

- **Parameters**:
  - `task=detect`: Specifies the detection task.
  - `mode=train`: Sets the mode to training.
  - `model=yolov8n.pt`: Starts from the pre-trained YOLOv8 nano model.
  - `data=./data.yaml`: Uses the specified data configuration file.
  - `epochs=300`: Number of training epochs.
  - `imgsz=640`: Image size for training.
  - `augment=True`: Enables data augmentation.

  For more information about the training parameters, check https://docs.ultralytics.com/usage/cfg/#train-settings

To generate a .ncnn model after training is done, use the following command:

```bash
yolo export model=./runs/detect/train/weights/best.pt format=ncnn
```

The .ncnn model will be saved in the `runs/detect/train/weights/best_ncnn_model` directory.

### Training with Google Colab

You can also train the model using the `train_yolov8_jupyter.ipynb` notebook in Google Colab:

1. Upload the dataset and notebook to your Google Drive.
2. Open the notebook in Google Colab.
3. Adjust the paths in the notebook to point to your dataset.
4. Run all cells to start training.

## Results

Training artifacts and results are saved in the `runs/detect/train` directory.

### Key Outputs

- **Model Weights**:
  - `best.pt`: Best model based on validation metrics.
  - `last.pt`: Model from the final training epoch.
  - `best_ncnn_model`: Folder containing multiple files for the NCNN model.

- **Training Metrics**:
  - `results.png`: Overview of training metrics over epochs.
  - `F1_curve.png`: F1 score vs. confidence threshold.
  - `PR_curve.png`: Precision-Recall curve.
  - `confusion_matrix.png`: Confusion matrix of predictions.

- **Sample Images**:
  - `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`: Sample training images with annotations.
  - `val_batch0_pred.jpg`, `val_batch1_pred.jpg`: Validation images with model predictions.


## Usage

Use the trained model to detect car plates in new images:

```bash
yolo task=detect mode=predict model=./runs/detect/train/weights/best.pt source=./your_image_or_directory
```

- **source**: Path to an image file or a directory of images.

### Example

```bash
yolo task=detect mode=predict model=./runs/detect/train/weights/best.pt source=./test/images save=True
```

- **save=True**: Saves the prediction images with bounding boxes in the `runs/detect/predict` directory.

## Project Structure

```
.
├── data.yaml
├── README.md
├── requirements.txt
├── runs
│   └── detect
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           ├── PR_curve.png
│           ├── confusion_matrix.png
│           └── ... (other artifacts)
├── test
│   ├── classes.txt
│   ├── images
│   └── labels
├── train
│   ├── classes.txt
│   ├── images
│   └── labels
├── train_yolov8_jupyter.ipynb
├── valid
│   ├── classes.txt
│   ├── images
│   └── labels
├── yolo11n.pt
└── yolov8n.pt
```

## Acknowledgments

- **Ultralytics YOLOv8**: [GitHub Repository](https://github.com/ultralytics/ultralytics)
- **LabelImg**: [GitHub Repository](https://github.com/HumanSignal/labelImg)
