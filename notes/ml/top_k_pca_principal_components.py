import numpy as np

def find_optimal_components(explained_variance_ratio, threshold=0.95):
    """
    Find the optimal number of components that explain the desired amount of variance.
    
    Parameters:
        explained_variance_ratio (np.ndarray): Array of explained variance ratios
        threshold (float): Minimum cumulative variance to explain (default: 0.95 or 95%)
    
    Returns:
        int: Optimal number of components
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    return n_components

def pca(A, k=None, variance_threshold=0.95):
    """
    Perform PCA on matrix A and return the optimal number of principal components.
    
    Parameters:
        A (np.ndarray): The input data matrix where each row is a sample and each column is a feature.
        k (int, optional): Maximum number of components to return. If None, determined automatically.
        variance_threshold (float): Minimum cumulative variance to explain (default: 0.95 or 95%)
    
    Returns:
        principal_components (np.ndarray): A matrix whose columns are the principal components.
        explained_variance_ratio (np.ndarray): The proportion of variance explained by each component.
        n_components (int): Number of components actually used
    """
    # Step 1: Center the data by subtracting the mean of each feature
    A_centered = A - np.mean(A, axis=0)
    
    # Step 2: Compute the covariance matrix (features are in columns)
    covariance_matrix = np.cov(A_centered, rowvar=False)
    
    # Step 3: Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort the eigenvalues (and eigenvectors) in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Calculate explained variance ratio
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance
    
    # Find optimal number of components
    if k is None:
        k = min(find_optimal_components(explained_variance_ratio, variance_threshold), A.shape[1])
    else:
        k = min(k, A.shape[1])
    
    # Step 5: Select the optimal number of eigenvectors
    principal_components = sorted_eigenvectors[:, :k]
    explained_variance_ratio = explained_variance_ratio[:k]
    
    return principal_components, explained_variance_ratio, k

def analyze_dimensionality(A, variance_threshold=0.95):
    """
    Analyze the dimensionality of the dataset and explain why it has a certain number of principal components.
    
    Parameters:
        A (np.ndarray): The input data matrix
        variance_threshold (float): Minimum cumulative variance to explain
    """
    print(f"\nDimensionality Analysis for dataset shape {A.shape}:")
    print(f"Number of samples (rows): {A.shape[0]}")
    print(f"Number of features (columns/dimensions): {A.shape[1]}")
    
    # Get all principal components and their explained variance
    _, explained_variance_ratio, optimal_k = pca(A, variance_threshold=variance_threshold)
    
    print(f"\nUsing variance threshold of {variance_threshold*100}%")
    print("\nExplained Variance Ratio for each principal component:")
    cumulative_variance = 0
    for i, ratio in enumerate(explained_variance_ratio, 1):
        cumulative_variance += ratio
        print(f"PC{i}: {ratio:.6f} ({ratio*100:.2f}% of variance, cumulative: {cumulative_variance*100:.2f}%)")
    
    print(f"\nOptimal number of components needed: {optimal_k}")
    print(f"This explains {cumulative_variance*100:.2f}% of the total variance")
    
    print("\nWhy this number of components?")
    print("1. The maximum number of principal components possible is min(n_samples-1, n_features)")
    print(f"   In this case: min({A.shape[0]}-1, {A.shape[1]}) = {min(A.shape[0]-1, A.shape[1])}")
    if optimal_k < A.shape[1]:
        print(f"2. However, we only need {optimal_k} component(s) to explain {variance_threshold*100}% of the variance")
        print("   This means we can effectively reduce the dimensionality while preserving most of the information")

def test_pca_example():
    """Test function to demonstrate PCA with different datasets."""
    # Example 1: Original 2D dataset
    print("\n=== Example 1: Original 2D dataset ===")
    A1 = np.array([[2.5, 2.4],
                  [0.5, 0.7],
                  [2.2, 2.9],
                  [1.9, 2.2],
                  [3.1, 3.0],
                  [2.3, 2.7],
                  [2.0, 1.6],
                  [1.0, 1.1],
                  [1.5, 1.6],
                  [1.1, 0.9]])
    
    # Get optimal number of components
    pcs, var_ratio, n_components = pca(A1)
    print(f"\nOptimal number of principal components: {n_components}")
    print("Principal Components:")
    print(pcs[:, :n_components])
    print("\nVariance explained by each component:")
    print(var_ratio)
    
    # Analyze dimensionality
    analyze_dimensionality(A1)

    # Example 2: 3D dataset
    print("\n\n=== Example 2: 3D dataset ===")
    A2 = np.array([
        [1, 2, 0.5],
        [2, 4, 1],
        [3, 6, 1.5],
        [4, 8, 2],
        [1.1, 2.1, 3],
        [2.2, 4.2, 6],
        [3.3, 6.3, 9],
        [1, 1, 1],
        [2, 1, 3],
        [3, 1, 5],
        [0.5, 3, 2],
        [1.5, 5, 4],
    ])
    
    # Get optimal number of components
    pcs, var_ratio, n_components = pca(A2)
    print(f"\nOptimal number of principal components: {n_components}")
    print("Principal Components:")
    print(pcs[:, :n_components])
    print("\nVariance explained by each component:")
    print(var_ratio)
    
    # Analyze dimensionality
    analyze_dimensionality(A2)

if __name__ == "__main__":
    test_pca_example()
