"""
Principal Component Analysis (PCA) is a dimensionality reduction technique 
used in machine learning. It transforms high-dimensional data into a lower-dimensional 
representation while retaining as much variance as possible.

This implementation includes:
- Standardizing the dataset.
- Computing principal components using Singular Value Decomposition (SVD).
- Returning transformed data and explained variance ratio.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from typing import Tuple


def collect_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Collects the dataset (Iris dataset) and returns feature matrix and target values.

    :return: Tuple containing feature matrix (X) and target labels (y)

    Example:
    >>> X, y = collect_dataset()
    >>> X.shape
    (150, 4)
    >>> y.shape
    (150,)
    """
    data = load_iris()
    return np.array(data.data), np.array(data.target)


def apply_pca(data_x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.

    :param data_x: Original dataset (features)
    :param n_components: Number of principal components to retain
    :return: Tuple containing transformed dataset and explained variance ratio

    Example:
    >>> X, _ = collect_dataset()
    >>> transformed_X, variance = apply_pca(X, 2)
    >>> transformed_X.shape
    (150, 2)
    >>> len(variance) == 2
    True
    """
    # Standardizing the dataset
    scaler = StandardScaler()
    data_x_scaled = scaler.fit_transform(data_x)

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_x_scaled)

    return principal_components, pca.explained_variance_ratio_


def main() -> None:
    """
    Driver function to execute PCA and display results.
    """
    data_x, data_y = collect_dataset()

    # Number of principal components to retain
    n_components = 2

    # Apply PCA
    transformed_data, variance_ratio = apply_pca(data_x, n_components)

    print("Transformed Dataset (First 5 rows):")
    print(transformed_data[:5])

    print("\nExplained Variance Ratio:")
    print(variance_ratio)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
