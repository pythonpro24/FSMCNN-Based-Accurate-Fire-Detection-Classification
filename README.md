# FSMCNN-Based-Accurate-Fire-Detection-Classification
FSMCNN-based fire detection system using a Self-Modulative Convolutional Neural Network with image preprocessing, classification, performance evaluation, and real-time fire prediction.
A deep learning-based fire detection system using a Self-Modulative Convolutional Neural Network (FSMCNN) for classifying Fire and Non-Fire images.

## Dataset

The project uses the Fire Detection Dataset from Kaggle.

Classes:

* Fire
* Neutral (Non-Fire)

## Workflow

1. Dataset Loading
2. Image Preprocessing

   * RGB Conversion
   * Resize (128×128)
   * Normalization
3. FSMCNN Model Training
4. Model Evaluation
5. Fire Prediction

## Model Architecture

* Conv2D (32 Filters)

* Batch Normalization

* MaxPooling2D

* Conv2D (64 Filters)

* Batch Normalization

* MaxPooling2D

* Conv2D (128 Filters)

* Batch Normalization

* MaxPooling2D

* Conv2D (256 Filters)

* Batch Normalization

* MaxPooling2D

* Global Average Pooling

* Dense Layer (256 Units)

* Dropout (0.5)

* Sigmoid Output Layer

## Requirements

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install kagglehub
pip install tqdm
```

## Training

Run the notebook or Python script to:

* Load dataset
* Preprocess images
* Train FSMCNN
* Save trained model

Model file:

```text
fire_detection_smcnn.keras
```

## Results

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 96.23% |
| Precision   | 95.30% |
| Recall      | 98.38% |
| F1-Score    | 96.82% |
| Specificity | 93.24% |
| ROC-AUC     | 99.41% |
| Error Rate  | 3.77%  |

## Prediction

The trained model can classify a new image as:

```text
Fire
```

or

```text
Neutral
```

## Applications

* Forest Fire Detection
* Smart Surveillance
* Industrial Safety
* Disaster Monitoring
* Early Warning Systems

## Author

FSMCNN-Based Accurate Fire Detection Using a Self-Modulative Convolutional Neural Network.

