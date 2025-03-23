"""
Singular Value Decomposition (SVD) for Matrix Compression

This module implements matrix compression using SVD. SVD decomposes a matrix A into three matrices:
A = U @ Σ @ V^T, where:
- U: Left singular vectors (orthogonal matrix)
- Σ: Diagonal matrix of singular values
- V^T: Right singular vectors (orthogonal matrix, transposed)

Key concepts:
1. SVD can be used for lossy matrix compression by keeping only the top k singular values
2. The singular values indicate the importance of each component
3. Compression ratio = original size / compressed size
4. Variance preserved indicates how much information is retained after compression

Real-world applications:
1. Image compression
2. Dimensionality reduction
3. Noise reduction
4. Recommendation systems
5. Latent semantic analysis in NLP
"""

import numpy as np

def find_optimal_singular_values(singular_values, threshold=0.95):
    """
    Find the optimal number of singular values that explain the desired amount of variance.
    
    Parameters:
        singular_values (np.ndarray): Array of singular values
        threshold (float): Minimum cumulative variance to explain (default: 0.95 or 95%)
    
    Returns:
        int: Optimal number of singular values
    """
    # Calculate the explained variance ratio
    explained_variance = singular_values ** 2
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    
    # Find number of components needed to explain threshold variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    k = np.argmax(cumulative_variance >= threshold) + 1
    return k

def svd(A, k=None, variance_threshold=0.95):
    """
    Perform SVD on matrix A and return the optimal low-rank decomposition.
    
    Parameters:
        A (np.ndarray): The input data matrix
        k (int, optional): Maximum number of singular values to use. If None, determined automatically.
        variance_threshold (float): Minimum cumulative variance to explain (default: 0.95 or 95%)
    
    Returns:
        U (np.ndarray): Left singular vectors
        s (np.ndarray): Singular values
        Vh (np.ndarray): Right singular vectors (transposed)
        k (int): Number of singular values actually used
        explained_variance_ratio (np.ndarray): Proportion of variance explained by each singular value
    """
    # Compute full SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Calculate explained variance ratio
    explained_variance = s ** 2
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    
    # Find optimal k if not specified
    if k is None:
        k = min(find_optimal_singular_values(s, variance_threshold), len(s))
    else:
        k = min(k, len(s))
    
    # Return truncated matrices
    return U[:, :k], s[:k], Vh[:k], k, explained_variance_ratio[:k]

def analyze_svd(A, variance_threshold=0.95):
    """
    Analyze the SVD decomposition of the matrix and explain its properties.
    
    Parameters:
        A (np.ndarray): The input data matrix
        variance_threshold (float): Minimum cumulative variance to explain
    """
    print(f"\nSVD Analysis for matrix shape {A.shape}:")
    print(f"Number of rows: {A.shape[0]}")
    print(f"Number of columns: {A.shape[1]}")
    
    # Get SVD decomposition
    U, s, Vh, k, explained_variance_ratio = svd(A, variance_threshold=variance_threshold)
    
    print(f"\nUsing variance threshold of {variance_threshold*100}%")
    print("\nExplained Variance Ratio for each singular value:")
    cumulative_variance = 0
    for i, ratio in enumerate(explained_variance_ratio, 1):
        cumulative_variance += ratio
        print(f"SV{i}: {ratio:.6f} ({ratio*100:.2f}% of variance, cumulative: {cumulative_variance*100:.2f}%)")
    
    print(f"\nOptimal rank needed: {k}")
    print(f"This explains {cumulative_variance*100:.2f}% of the total variance")
    
    print("\nResulting matrix shapes:")
    print(f"U: {U.shape} - Left singular vectors")
    print(f"s: {s.shape} - Singular values")
    print(f"Vh: {Vh.shape} - Right singular vectors (transposed)")
    
    # Verify reconstruction
    A_reconstructed = U @ np.diag(s) @ Vh
    reconstruction_error = np.linalg.norm(A - A_reconstructed, 'fro')
    print(f"\nFrobenius norm of reconstruction error: {reconstruction_error:.6f}")

def compute_compression_ratio(original_size, compressed_size):
    """
    Compute the compression ratio achieved by SVD.
    
    Parameters:
        original_size (int): Number of elements in original matrix
        compressed_size (int): Number of elements in compressed representation
    
    Returns:
        float: Compression ratio (original_size / compressed_size)
    """
    return original_size / compressed_size

def svd_compress(A, k=None, target_size_ratio=0.5):
    """
    Compress matrix A using SVD with automatic rank selection based on size reduction.
    
    The compression works by:
    1. Computing full SVD: A = U @ Σ @ V^T
    2. Selecting optimal rank k based on target compression ratio
    3. Keeping only top k singular values and corresponding vectors
    
    Storage analysis:
    - Original matrix: m × n elements
    - Compressed form: (m×k + k + k×n) elements, where:
      * U: m×k elements (left singular vectors)
      * Σ: k elements (singular values)
      * V^T: k×n elements (right singular vectors)
    
    Parameters:
        A (np.ndarray): The input data matrix
        k (int, optional): Specific rank to use. If None, determined by target_size_ratio
        target_size_ratio (float): Target compressed size as fraction of original (default: 0.5)
    
    Returns:
        U (np.ndarray): Left singular vectors
        s (np.ndarray): Singular values
        Vh (np.ndarray): Right singular vectors (transposed)
        compression_stats (dict): Statistics about the compression
    """
    # Compute full SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Calculate storage requirements for different ranks
    m, n = A.shape
    original_size = m * n
    
    if k is None:
        # For each possible rank, calculate compressed size
        compressed_sizes = []
        for r in range(1, min(m, n) + 1):
            # Size = U(m×r) + s(r) + Vh(r×n)
            compressed_size = m*r + r + r*n
            compressed_sizes.append(compressed_size)
            
            # If we've reached target compression ratio, use this rank
            if compressed_size <= original_size * target_size_ratio:
                k = r
                break
        
        # If no k found, use the rank that gives minimum size while preserving data
        if k is None:
            k = len(s)
    
    # Compute explained variance (proportional to squared singular values)
    explained_variance = s ** 2
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Calculate compression statistics
    compressed_size = m*k + k + k*n
    compression_ratio = compute_compression_ratio(original_size, compressed_size)
    
    compression_stats = {
        'rank': k,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'variance_explained': cumulative_variance[k-1]
    }
    
    # Return truncated matrices and stats
    return U[:, :k], s[:k], Vh[:k], compression_stats

def analyze_svd_compression(A, k=None, target_size_ratio=0.5):
    """
    Analyze the SVD compression of the matrix and its effectiveness.
    
    This function provides insights into:
    1. Compression efficiency (size reduction)
    2. Information preservation (variance explained)
    3. Reconstruction quality (error)
    4. Matrix shapes and singular value spectrum
    
    Parameters:
        A (np.ndarray): The input data matrix
        k (int, optional): Specific rank to use
        target_size_ratio (float): Target compressed size as fraction of original
    """
    print(f"\nSVD Compression Analysis for matrix shape {A.shape}:")
    print(f"Original matrix size: {A.shape[0]} × {A.shape[1]} = {A.shape[0] * A.shape[1]} elements")
    
    # Perform compression
    U, s, Vh, stats = svd_compress(A, k, target_size_ratio)
    
    # Print compression results
    print(f"\nCompression Results:")
    print(f"Rank used: {stats['rank']}")
    print(f"Original size: {stats['original_size']} elements")
    print(f"Compressed size: {stats['compressed_size']} elements")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Variance preserved: {stats['variance_explained']*100:.2f}%")
    
    # Reconstruct and compute error
    A_reconstructed = U @ np.diag(s) @ Vh
    relative_error = np.linalg.norm(A - A_reconstructed, 'fro') / np.linalg.norm(A, 'fro')
    print(f"Relative reconstruction error: {relative_error:.6f}")
    
    print("\nCompressed representation:")
    print(f"U: {U.shape} matrix")
    print(f"s: {s.shape} vector")
    print(f"Vh: {Vh.shape} matrix")
    
    # Show singular value spectrum
    print("\nSingular Value Spectrum:")
    for i, sigma in enumerate(s, 1):
        print(f"σ{i}: {sigma:.6f}")

def test_svd_example():
    """
    Test function to demonstrate SVD compression with different matrices.
    
    Examples demonstrate:
    1. Small matrix compression (may not be beneficial)
    2. Different compression ratios on medium-sized data
    3. Recovery of known low-rank structure with noise
    """
    # Example 1: Small 2D dataset (demonstrates limitations with small matrices)
    print("\n=== Example 1: Small 2D dataset ===")
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
    
    # Analyze compression (aiming for 50% size reduction)
    analyze_svd_compression(A1, target_size_ratio=0.5)

    # Example 2: 3D dataset (demonstrates different compression ratios)
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
    
    # Try different compression ratios
    print("\nTrying 70% size reduction:")
    analyze_svd_compression(A2, target_size_ratio=0.3)
    
    print("\nTrying 30% size reduction:")
    analyze_svd_compression(A2, target_size_ratio=0.7)

    # Example 3: Random matrix with known rank-2 structure
    print("\n\n=== Example 3: Random matrix with known rank-2 structure ===")
    # Create a rank-2 matrix by multiplying two rank-2 matrices
    np.random.seed(42)
    X = np.random.randn(20, 2)
    Y = np.random.randn(2, 15)
    A3 = X @ Y + np.random.randn(20, 15) * 0.1  # Add some noise
    
    # Try to recover the rank-2 structure
    analyze_svd_compression(A3, k=2)

if __name__ == "__main__":
    test_svd_example() 