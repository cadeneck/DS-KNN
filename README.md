
# Machine Learning Models: K-Nearest Neighbors, Decision Tree, and Scorer

This repository includes implementations of the K-Nearest Neighbors (KNN) algorithm, a basic Decision Tree classifier, and a Scorer class for evaluating decision tree splits. These models are foundational in machine learning for classification tasks and provide a good starting point for understanding how to implement and apply these algorithms in Python.

## K-Nearest Neighbors (KNN.py)

The `KNearestNeighbor` class implements the KNN algorithm, which classifies instances based on the majority vote of the `k` nearest neighbors in the feature space.

### Features

- `fit(self, X_train: np.ndarray, y_train: np.ndarray)`: Saves the training dataset.
- `predict(self, X_predict: np.ndarray)`: Predicts the class labels for the provided dataset.

## Decision Tree (dt.py)

The `DecisionTree` class provides a basic implementation of a decision tree for classification tasks. It supports creating trees using criteria like Gini index or information gain for splits.

### Components

- `Node`: Represents nodes in the tree, including decision nodes (splits) and leaves.
- `DecisionTree`: Manages the tree construction and prediction processes.

### Features

- `fit(self, X: ArrayLike, y: ArrayLike)`: Builds the decision tree from the training dataset.
- `predict(self, X: ArrayLike)`: Predicts class labels for a given dataset.

## Scorer (Scorer.py)

The `Scorer` class is designed to evaluate the quality of splits in the decision tree, using metrics like Gini index or entropy (information gain).

### Features

- `compute_class_probabilities(self, labels: ArrayLike)`: Computes the probabilities of each class in the dataset.
- `information_gain(self, data: ArrayLike, labels: ArrayLike, split_attribute: int)`: Calculates the information gain of a split.
- `gini_gain(self, data: ArrayLike, labels: ArrayLike, split_attribute: int)`: Calculates the Gini gain of a split.
