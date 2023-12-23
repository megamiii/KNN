# K-Nearest Neighbors (KNN) Project

This project includes a Python implementation of the k-Nearest Neighbors (KNN) algorithm, a simple yet powerful algorithm used for classification and regression tasks in machine learning. The provided script (`knn.py`) contains functions to load data and predict labels for data points based on their nearest neighbors.

## Features

- **Data Loading**: Functions to load testing and training data from CSV files.
- **KNN Classification**: Implementation of the KNN algorithm using Euclidean distance to determine the closest neighbors.
- **Tie Breaking**: Mechanism to handle ties between potential labels for a classification.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

### Installation

To use this KNN implementation, clone the repository to your local machine:

```bash
git clone https://github.com/megamiii/KNN.git
```

### Usage
The main script can be run in a Python environment with the necessary libraries installed. Ensure you have the dataset in the correct format as expected by the `load_knn_data` function.

## Detailed Results

For an in-depth analysis of the KNN algorithm's performance, including observations from output graphs and terminal results, please see the [RESULTS.md](RESULTS.md) file in this repository. It provides a comprehensive overview of the algorithm's accuracy, optimal parameters, and generalization capabilities based on the conducted tests.

## Files in the Repository
- `README.md`: Provides an overview of the project, including how to set up and run the KNN algorithm.
- `RESULTS.md`: Contains detailed analysis of the KNN algorithm's performance, the output graph, and the terminal output summary.
- `knn.py`: The main script with KNN algorithm implementation and data loading functions.
- `knn-dataset/`:
  - `train_inputs.csv`: Training data input features.
  - `train_labels.csv`: Training data labels.
  - `test_inputs.csv`: Test data input features.
  - `test_labels.csv`: Test data labels.
- `results/`:
  - `result.png`: The graph showing the relationship between the number of neighbors and accuracy.
  - `output.png`: The terminal output showcasing the results of the KNN algorithm.