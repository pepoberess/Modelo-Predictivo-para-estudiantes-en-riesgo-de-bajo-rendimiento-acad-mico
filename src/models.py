import numpy as np

class LogisticRegression:
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
    
    def cross_entropy(self, L1=False, L2=False):
        y_pred = self.probability(self.X)
        y = self.y
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred)
        )
        
        if L2:
            loss += self.L2 * np.sum(self.weights ** 2)
        if L1:
            loss += self.L1 * np.sum(np.abs(self.weights))
        
        return loss
    
    def gradients(self, L1=False, L2=False):
        y_pred = self.probability(self.X)

        error = y_pred - self.y
        dw = (1 / self.n_samples) * np.dot(self.X.T, error)
        db = (1 / self.n_samples) * np.sum(error)

        if L2:
            dw += 2 * self.L2 * self.weights
        
        return dw, db
    
    def fit(self, learning_rate=0.001, iters=10000, tol=1e-6, L1=False, L2=False):
    
        prev_loss = float("inf")
        min_loss = float("inf")
        L2 = self.L2
        reachTOL = False
        
        for i in range(iters):
            
            dw, db = self.gradients(L1=L1, L2=L2)
            
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            current_loss = self.cross_entropy(L1=L1, L2=L2)
            
            if abs(prev_loss - current_loss) < tol:
                reachTOL = True
                min_loss = current_loss
                print(f"Converged at iteration {i}")
                break
            
            prev_loss = current_loss

        if not reachTOL:
            min_loss = prev_loss
            print(f"Reached maximum iterations without convergence.")

        print(f"Final Loss: {min_loss:.4f}")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")