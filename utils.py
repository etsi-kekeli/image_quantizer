import numpy as np

def assign_centroids(X : np.ndarray, centroids : np.ndarray) -> np.ndarray:
    """
    Assign each data point to the closest centroid
    Args:
        X: numpy array of shape (m, n) containing the data points
        centroids: numpy array of shape (K, n) containing the centroids
    Returns:
        idx: numpy array of shape (m, ) containing the index of the closest centroid
    """
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m)
    distances = np.zeros((m, K))
    
    for k in range(K):
        distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)

    idx = np.argmin(distances, axis=1)
        
                
    return idx

def update_centroids(X : np.ndarray, idx : np.ndarray, K : int) -> np.ndarray:
    """
    Compute the new centroids by taking the average of the data points assigned to each centroid
    Args:
        X: numpy array of shape (m, n) containing the data points
        idx: numpy array of shape (m, ) containing the index of the closest centroid
        K: number of centroids
    Returns:
        centroids: numpy array of shape (K, n) containing the new centroids
    """
    _, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        centroids[k] = np.mean(X[idx == k], axis=0)
        
    return centroids

def kmeans(X : np.ndarray, K : int, max_iter : int = 20, intial_centroids : np.ndarray = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform K-means on the data points and return the K centroids
    Args:
        X: numpy array of shape (m, n) containing the data points
        K: number of centroids
        max_iter: maximum number of iterations to perform
        initial_centroids: numpy array of shape (K, n) containing the initial centroids
    Returns:
        centroids: numpy array of shape (K, n) containing the centroids
        idx: numpy array of shape (m, ) containing the index of the closest centroid
        J : distorsion of the representation of data points by the centroids using MSE
    """
    
    m, n = X.shape
    if not intial_centroids:
        random_index = np.random.permutation(m)
        initial_centroids = X[random_index[:K]]
    
    centroids = initial_centroids
    prev_centroids = centroids.copy()
    idx = np.zeros(m)

    for i in range(max_iter):
        idx = assign_centroids(X, centroids)
        new_K = len(np.unique(idx))
        centroids = update_centroids(X, idx, new_K)
        if K == new_K and np.all(prev_centroids == centroids):
            break
        prev_centroids = centroids.copy()

    J = np.mean(np.linalg.norm(X - centroids[idx], axis=1)**2)

    return (centroids, idx, J)

