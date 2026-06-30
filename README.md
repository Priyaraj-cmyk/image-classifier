Image Classification Project (PyTorch)

Overview

This project implements an image classification model using PyTorch. It uses transfer learning with a pretrained convolutional neural network to classify images into predefined categories.

Features

* Transfer learning using pretrained models
* Data augmentation for improved generalization
* Training and validation performance tracking
* Model saving and loading for inference

Project Structure

```
project/
│
├── model.pth              # Trained model weights
├── requirements.txt       # Dependencies
├── CIFAR-10 Image Classification.ipynb         # Training notebook
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:

```
git clone
cd project
```

2. Install dependencies:

```
pip install -r requirements.txt
```

Training the Model

Run the notebook or training script:

```
jupyter notebook
```

Train the model and save weights:

```
torch.save(model.state_dict(), "model.pth")
```

## Inference

Load the trained model:

```python
import torch
from torchvision import models

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("model.pth"))
model.eval()
```

Use the model to predict:

```python
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
```

Requirements

All dependencies are listed in `requirements.txt`.

Notes

* Ensure the same model architecture is used when loading weights
* Normalize input images the same way as during training
* Use GPU if available for faster training

Author

Priyaraj
