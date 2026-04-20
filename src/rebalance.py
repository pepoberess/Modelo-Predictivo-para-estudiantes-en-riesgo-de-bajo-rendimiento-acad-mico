import numpy as np

def undersample(X, y, random_state=42):
    """
    Randomly remove samples from the majority class until both classes
    have equal proportion.
 
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) — binary labels 0/1
 
    Returns
    -------
    X_resampled, y_resampled
    """
    np.random.seed(random_state)
 
    idx_minority = np.where(y == 0)[0]
    idx_majority = np.where(y == 1)[0]
 
    n_minority = len(idx_minority)
 
    # Randomly sample n_minority from majority
    idx_majority_downsampled = np.random.choice(idx_majority, n_minority, replace=False)
 
    idx_resampled = np.concatenate([idx_minority, idx_majority_downsampled])
    np.random.shuffle(idx_resampled)
 
    return X[idx_resampled], y[idx_resampled]
 
 
def oversample_duplicate(X, y, random_state=42):
    """
    Randomly duplicate samples from the minority class until both classes
    have equal proportion.
 
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) — binary labels 0/1
 
    Returns
    -------
    X_resampled, y_resampled
    """
    np.random.seed(random_state)
 
    idx_minority = np.where(y == 0)[0]
    idx_majority = np.where(y == 1)[0]

    n_majority = len(idx_majority)
    n_minority = len(idx_minority)
    n_to_add = n_majority - n_minority
 
    # Randomly sample with replacement from minority
    idx_minority_upsampled = np.random.choice(idx_minority, n_to_add, replace=True)
 
    idx_resampled = np.concatenate([np.arange(len(y)), idx_minority_upsampled])
    np.random.shuffle(idx_resampled)
 
    return X[idx_resampled], y[idx_resampled]
 
 
def oversample_smote(X, y, k=5, random_state=42):
    """
    SMOTE: generate synthetic samples for the minority class by interpolating
    between a sample and one of its K nearest neighbors.
 
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) — binary labels 0/1
    k : number of nearest neighbors to consider
 
    Returns
    -------
    X_resampled, y_resampled
    """
    np.random.seed(random_state)
 
    minority_class = 0
    majority_class = 1
 
    X_min = X[y == minority_class]
    X_maj = X[y == majority_class]
 
    n_majority = len(X_maj)
    n_minority = len(X_min)
    n_to_generate = n_majority - n_minority
 
    synthetic = []
 
    for _ in range(n_to_generate):
        # Pick a random minority sample
        idx = np.random.randint(0, n_minority)
        sample = X_min[idx]
 
        # Find K nearest neighbors among minority samples
        dists = np.linalg.norm(X_min - sample, axis=1)
        dists[idx] = np.inf  # To exclude itself
        neighbor_idxs = np.argsort(dists)[:k]
 
        # Pick one neighbor at random
        neighbor = X_min[np.random.choice(neighbor_idxs)]
 
        # Interpolate between sample and neighbor
        alpha = np.random.uniform(0, 1)
        new_sample = sample + alpha * (neighbor - sample)
        synthetic.append(new_sample)
 
    X_synthetic = np.array(synthetic)
    y_synthetic = np.zeros(n_to_generate)
 
    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.concatenate([y, y_synthetic])
 
    # Shuffle for synthetic samples to be mixed with the original ones
    idx_shuffle = np.random.permutation(len(y_resampled))
    return X_resampled[idx_shuffle], y_resampled[idx_shuffle]