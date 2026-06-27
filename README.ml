# Multi-Channel House Price Prediction using Keras Functional API

## Overview

This project demonstrates how to build a **multi-channel (multi-input)** deep learning model using the **Keras Functional API**.

Unlike traditional deep learning models that receive only a single type of input, many real-world problems involve multiple data modalities. In this project, the model learns from both:

* Structured (tabular) data
* House images

The independent feature extraction pipelines are built separately and then merged into a single network for final house price prediction.

---

## Why Multi-Channel Models?

For problems with a **single input**, the **Keras Sequential API** is usually sufficient because the data flows through a simple stack of layers.

However, many practical applications require multiple input sources. Examples include:

* Images + numerical features
* Text + metadata
* Audio + video
* Sensor data from multiple devices

In these situations, each input requires its own processing pipeline (or **channel**). The extracted features from each channel are then concatenated and passed through additional fully connected layers before producing the final prediction.

The Keras Functional API makes it straightforward to design these flexible architectures.

A simplified workflow is shown below:

```
Tabular Data ──► Dense Layers ──┐
                                ├──► Concatenate ─► Dense Layers ─► Output
Images ───────► CNN Layers ─────┘
```

---

# Dataset

This project uses the **House Prices Dataset** created by Eman Hamed.

Dataset repository:

https://github.com/emanhamed/houses-dataset

The dataset contains **535 house samples**.

Each sample consists of:

### Tabular Features

* Number of bedrooms
* Number of bathrooms
* Area
* Zipcode
* House price (target)

### Image Features

Each house contains **4 RGB images** captured from different viewpoints.

These images are used together as the visual representation of each property.

---

# Data Preprocessing

## Tabular Data

Standard preprocessing is applied to the structured features, including normalization and encoding where appropriate.

### Zipcode Processing

Some zipcodes contain only a very small number of samples.

Keeping these rare zipcodes as independent categories introduces unnecessary sparsity and provides little useful information to the model.

To address this issue, all rare zipcodes are merged into a single category represented by:

```
Zipcode = 0
```

This reduces the number of categories and improves the robustness of the model.

---

## Image Preprocessing

Image preprocessing includes the common computer vision preparation steps such as resizing and normalization.

Since each house has **four images**, all four images must be used simultaneously.

Two different approaches can be adopted:

### Method 1: Spatial Concatenation

Arrange the four images into a single larger image.

```
Output Shape:
(128, 128, 3)
```

This preserves the RGB channels while increasing the spatial dimensions.

---

### Method 2: Channel Concatenation

Concatenate the four RGB images along the channel dimension.

```
Output Shape:
(64, 64, 12)
```

Since each image has 3 channels:

```
4 × 3 = 12 channels
```

This approach keeps the original spatial resolution while allowing the CNN to learn jointly from all four views.

---

# Model Architecture

The model consists of two independent branches:

### Image Branch

* Convolutional Neural Network (CNN)
* Feature extraction from house images

### Tabular Branch

* Fully connected (Dense) layers
* Processing of structured numerical and categorical features

The outputs of both branches are concatenated and passed through additional Dense layers to predict the house price.

```
           Images
              │
           CNN Layers
              │
              ├──────────────┐
                             │
                             ▼
                        Concatenate
                             ▲
              ┌──────────────┘
              │
      Tabular Features
              │
         Dense Layers
              │
              ▼
        Fully Connected
              │
              ▼
        House Price
```

---

# Technologies Used

* Python
* TensorFlow
* Keras Functional API
* NumPy
* Pandas
* OpenCV
* Scikit-learn

---

# Project Goal

The objective of this project is to demonstrate how to design and train **multi-input neural networks** using the Keras Functional API by combining image data and structured tabular features for a regression task.
