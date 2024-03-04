# README

## Exoplanet-Detection_Machine_Learning

This repository contains code for time series classification using a transformer-based model for exoplanet detection. The project utilizes PyTorch for implementing the transformer model and training process.

## Overview

The repository includes several components:

- `dataset.py`: Defines the `KeplerDataset` class for loading and preprocessing the Kepler dataset for training and testing.
- `transformer.py`: Implements the transformer model architecture (`ViT`) for time series classification.
- `train.py`: Script for training the transformer model using distributed training with PyTorch `DistributedDataParallel`.
- `test.py`: Script for evaluating the trained model on the test dataset and generating performance metrics.
- `utils.py`: Contains utility functions used in the training and evaluation processes.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have the necessary dependencies installed, including PyTorch, NumPy, scikit-learn, and imbalanced-learn.
3. Prepare the dataset by downloading the Kepler dataset and placing it in the appropriate directory (`/big-data/BrainProject/ryan/models/ts_xformer/data/`).
4. Run the training script `train.py` to train the transformer model on the training dataset.
5. Run the evaluation script `test.py` to evaluate the trained model on the test dataset and generate performance metrics.
6. Customize the training parameters, model architecture, and dataset paths as needed for your specific application.

## Dataset

The dataset used in this project is the Kepler dataset, which contains time series data of stellar brightness measurements collected by the Kepler space telescope. The dataset is preprocessed to extract relevant features for exoplanet detection.

## Model Architecture

The transformer-based model (`ViT`) is used for time series classification. It consists of self-attention layers and feedforward layers to capture temporal dependencies in the input data.

## Distributed Training

The training script `train.py` utilizes distributed training with PyTorch `DistributedDataParallel` to accelerate training across multiple GPUs or nodes.

## Results

After training and evaluation, the model's performance metrics, including accuracy, precision, recall, and confusion matrix, are generated and visualized for analysis.

## Technical Skills

- **Programming Languages**: Python
- **Libraries/Frameworks**: PyTorch, NumPy, scikit-learn, imbalanced-learn
- **Distributed Computing**: PyTorch `DistributedDataParallel`
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Time series classification, Transformer models

## Author

Ankur Shah

## Contributions

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to submit a pull request.

## Contact

For any inquiries or feedback, please contact work.ankurshah@gmail.com.

