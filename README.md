# Waste Classification Model API

## Overview

This project sets up a FastAPI application to serve a TensorFlow model for waste classification. The API provides an endpoint to classify waste items based on input features.

## Model Details

### Inputs
- **Input**: A list of features required by the TensorFlow model for classifying waste. The exact nature of the features will depend on how the model was trained (e.g., image data, text descriptions, etc.).

### Outputs
- **Output**: The classification result from the model, indicating the type of waste (Cardboard, Glass, Metal, Plastic, Paper).
