import numpy as np

class BinaryLogisticRegression:
    def __init__(self, X, y, L1=0.0, L2=0.0):
        self.X = X
        self.y = y
        self.L1 = L1
        self.L2 = L2
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def probability(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def prediction(self, X):
        probs = self.probability(X)
        return (probs >= 0.5).astype(int)
    
    def cross_entropy(self, L1=False, L2=False, C=False):
        y_pred = self.probability(self.X)
        y = self.y
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if C:
            pi1 = np.mean(y == 0)  # minoritary
            pi2 = np.mean(y == 1)  # majoritary
            c_weight = pi2 / pi1
            loss = -np.mean(
                y * np.log(y_pred) +
                c_weight * (1 - y) * np.log(1 - y_pred)
            )
        else:
            loss = -np.mean(
                y * np.log(y_pred) +
                (1 - y) * np.log(1 - y_pred)
            )
        
        if L2:
            loss += self.L2 * np.sum(self.weights ** 2)
        if L1:
            loss += self.L1 * np.sum(np.abs(self.weights))
        
        return loss

    def gradients(self, L1=False, L2=False, C=False):
        y_pred = self.probability(self.X)
        y = self.y

        if C:
            pi1 = np.mean(y == 0)
            pi2 = np.mean(y == 1)
            c_weight = pi2 / pi1
            weights_per_sample = np.where(y == 0, c_weight, 1.0)
            error = (y_pred - y) * weights_per_sample
        else:
            error = y_pred - y

        dw = (1 / self.n_samples) * np.dot(self.X.T, error)
        db = (1 / self.n_samples) * np.sum(error)

        if L2:
            dw += 2 * self.L2 * self.weights
        
        return dw, db
    
    def fit(self, learning_rate=0.001, iters=10000, tol=1e-6, L1=False, L2=False, C=False, printing=False):
    
        prev_loss = float("inf")
        min_loss = float("inf")
        L2 = self.L2
        reachTOL = False
        
        for i in range(iters):
            
            dw, db = self.gradients(L1=L1, L2=L2, C=C)
            
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            current_loss = self.cross_entropy(L1=L1, L2=L2, C=C)
            
            if abs(prev_loss - current_loss) < tol:
                reachTOL = True
                min_loss = current_loss
                break
            
            prev_loss = current_loss

        if not reachTOL:
            min_loss = prev_loss

        if printing:
            print(f"Final Loss: {min_loss:.4f}")
            print(f"Weights: {self.weights}")
            print(f"Bias: {self.bias}")

class LDA:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = None
        self.priors = {}
        self.means = {}
        self.cov_inv = None
 
    def fit(self):
        X = self.X
        y = self.y

        self.classes = np.unique(y)
        n_samples, n_features = X.shape
 
        # Compute priors and means per class
        for k in self.classes:
            X_k = X[y == k]
            self.priors[k] = len(X_k) / n_samples
            self.means[k] = np.mean(X_k, axis=0)
 
        # Compute pooled within-class covariance matrix
        cov = np.zeros((n_features, n_features))
        for k in self.classes:
            X_k = X[y == k]
            diff = X_k - self.means[k]
            cov += diff.T @ diff
 
        cov /= (n_samples - len(self.classes))
 
        # Add small regularization for numerical stability
        cov += 1e-6 * np.eye(n_features)
 
        self.cov_inv = np.linalg.inv(cov)
 
    def _scores(self, X):
        """
        Compute the discriminant score for each class and each sample.
        score_k(x) = log(πₖ) - 0.5 * (x - μₖ)ᵀ Σ⁻¹ (x - μₖ)
        Returns array of shape (n_samples, n_classes)
        """
        scores = []
        for k in self.classes:
            diff = X - self.means[k]                          # (n_samples, n_features)
            mahal = np.sum(diff @ self.cov_inv * diff, axis=1) # mahalanobis distance
            score = np.log(self.priors[k]) - 0.5 * mahal
            scores.append(score)
        return np.column_stack(scores)  # (n_samples, n_classes)
 
    def probability(self, X):
        """
        Softmax over discriminant scores — returns class probabilities.
        Shape: (n_samples, n_classes)
        """
        scores = self._scores(X)
        # Numerically stable softmax as substracting a constant doesn't change the probabilities
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
    def prediction(self, X):
        """Returns predicted class label for each sample."""
        probs = self.probability(X)
        return self.classes[np.argmax(probs, axis=1)]
    
class MulticlassLogisticRegression:
    def __init__(self, X, y, L1=0.0, L2=0.0):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        self.L1 = L1
        self.L2 = L2
        
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
    
        self.weights = np.zeros((self.n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
 
    def one_hot(self):
        """Encode y as one-hot matrix of shape (n_samples, n_classes)."""
        y = self.y
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
 
    def softmax(self, z):
        """Numerically stable softmax. z shape: (n_samples, n_classes)"""
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)
 
    def probability(self, X):
        """Returns class probabilities. Shape: (n_samples, n_classes)"""
        z = X @ self.weights + self.bias
        return self.softmax(z)
 
    def prediction(self, X):
        """Returns predicted class label for each sample."""
        probs = self.probability(X)
        return self.classes[np.argmax(probs, axis=1)]
 
    def cross_entropy(self, L1=False, L2=False):
        y_oh = self.one_hot()
        probs = self.probability(self.X)
 
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
 
        loss = -np.mean(np.sum(y_oh * np.log(probs), axis=1))
 
        if L2:
            loss += self.L2 * np.sum(self.weights ** 2)
        if L1:
            loss += self.L1 * np.sum(np.abs(self.weights))
 
        return loss
 
    def gradients(self, L1=False, L2=False):
        y_oh = self.one_hot()
        probs = self.probability(self.X)
 
        error = probs - y_oh                                
        dw = (1 / self.n_samples) * self.X.T @ error        
        db = (1 / self.n_samples) * np.sum(error, axis=0)      

        if L2:
            dw += 2 * self.L2 * self.weights
        if L1:
            dw += self.L1 * np.sign(self.weights)
 
        return dw, db
 
    def fit(self, learning_rate=0.001, iters=10000, tol=1e-6, L1=False, L2=False, printing=True):
 
        prev_loss = float("inf")
        min_loss = float("inf")
        reachTOL = False
 
        for i in range(iters):
 
            dw, db = self.gradients(L1=L1, L2=L2)
 
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
 
            current_loss = self.cross_entropy(L1=L1, L2=L2)
 
            if abs(prev_loss - current_loss) < tol:
                reachTOL = True
                min_loss = current_loss
                if printing:
                    print(f"Converged at iteration {i}")
                break
 
            prev_loss = current_loss
 
        if not reachTOL:
            min_loss = prev_loss
            if printing:
                print(f"Reached maximum iterations without convergence.")
 
        if printing:
            print(f"Final Loss: {min_loss:.4f}")

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  
        self.threshold = threshold      
        self.left = left           
        self.right = right             
        self.value = value                         
 
    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  
        self.random_state = random_state
        self.root = None
 
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.root = self.grow(X, y, depth=0)
 
    def entropy(self, y):
        """H(S) = -sum(p_k * log2(p_k))"""
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-12))
 
    def information_gain(self, y, left_mask):
        """IG = H(parent) - weighted sum of H(children)"""
        n = len(y)
        n_left = left_mask.sum()
        n_right = n - n_left
 
        if n_left == 0 or n_right == 0:
            return 0.0
 
        h_parent = self.entropy(y)
        h_left = self.entropy(y[left_mask])
        h_right = self.entropy(y[~left_mask])
 
        return h_parent - (n_left / n) * h_left - (n_right / n) * h_right
 
    def best_split(self, X, y):
        """Find the best (feature, threshold) pair by maximizing information gain."""
        n_features = X.shape[1]
 
        # Random feature subsampling
        if self.max_features is not None:
            feature_idxs = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_idxs = np.arange(n_features)
 
        best_gain = -1
        best_feature = None
        best_threshold = None
 
        for f in feature_idxs:
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                # Skip if split doesn't respect min_samples_leaf
                if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
                    continue
                gain = self.information_gain(y, left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t
 
        return best_feature, best_threshold
 
    def grow(self, X, y, depth):
        """Recursively grow the tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
 
        # Stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth) or
            n_classes == 1 or
            n_samples < 2 * self.min_samples_leaf
        ):
            return Node(value=self.majority_class(y))
 
        # Find best split
        feature_idx, threshold = self.best_split(X, y)
 
        # If no valid split found, make leaf
        if feature_idx is None:
            return Node(value=self.majority_class(y))
 
        left_mask = X[:, feature_idx] <= threshold
        left = self.grow(X[left_mask], y[left_mask], depth + 1)
        right = self.grow(X[~left_mask], y[~left_mask], depth + 1)
 
        return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
 
    def majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
 
    def predict_one(self, x, node):
        """Traverse the tree for a single sample."""
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
 
    def prediction(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])
 
 
class RandomForest:
    def __init__(self, X, y, n_trees=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=42):
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
 
        n_features = X.shape[1]
        if max_features == "sqrt":
            self.max_features = int(np.sqrt(n_features))
        elif max_features == "log2":
            self.max_features = int(np.log2(n_features))
        elif isinstance(max_features, int):
            self.max_features = max_features
        else:
            self.max_features = n_features
 
    def fit(self):
        np.random.seed(self.random_state)
        X, y = self.X, self.y
        n_samples = X.shape[0]
        self.trees = []
 
        for i in range(self.n_trees):
            # Bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idxs], y[idxs]
 
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state + i  # different seed per tree
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
 
    def prediction(self, X):
        """Majority vote across all trees."""
        # Shape: (n_trees, n_samples)
        all_preds = np.array([tree.prediction(X) for tree in self.trees])
 
        # Majority vote per sample
        result = []
        for sample_preds in all_preds.T:
            classes, counts = np.unique(sample_preds, return_counts=True)
            result.append(classes[np.argmax(counts)])
        return np.array(result)
 
    def probability(self, X):
        """Class probabilities as vote proportions. Shape: (n_samples, n_classes)"""
        classes = np.unique(self.y)
        all_preds = np.array([tree.prediction(X) for tree in self.trees])
 
        probs = []
        for sample_preds in all_preds.T:
            counts = np.array([(sample_preds == c).sum() for c in classes])
            probs.append(counts / self.n_trees)
        return np.array(probs)
 
    def feature_importance(self):
        """
        Mean decrease in impurity across all trees and all splits.
        Returns array of shape (n_features,), normalized to sum to 1.
        """
        n_features = self.X.shape[1]
        importances = np.zeros(n_features)
 
        for tree in self.trees:
            self.accumulate_importance(tree.root, importances)
 
        importances /= importances.sum() if importances.sum() > 0 else 1
        return importances
 
    def accumulate_importance(self, node, importances):
        """Recursively accumulate information gain at each split node."""
        if node is None or node.is_leaf():
            return
        importances[node.feature_idx] += 1  # counts splits per feature
        self.accumulate_importance(node.left, importances)
        self.accumulate_importance(node.right, importances)
 