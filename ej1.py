import numpy as np
import matplotlib.pyplot as plt

samples1 = np.random.normal(3, 5, 20)
samples2 = np.random.normal(1, 7, 20)

X = np.concatenate([samples1, samples2]).reshape(-1, 1)
y = np.array([0]*20 + [1]*20)

# Separar clases
X0 = X[y == 0]
X1 = X[y == 1]

# Medias
mu0 = X0.mean()
mu1 = X1.mean()

# Varianza compartida
sigma2 = np.var(X)

# Priors
pi0 = len(X0) / len(X)
pi1 = len(X1) / len(X)

# Frontera de decisión
boundary_lda = ( (mu0 + mu1)/2 
                - (sigma2/(mu1 - mu0)) * np.log(pi1/pi0) )

print("LDA boundary:", boundary_lda)


sigma0 = np.var(X0)
sigma1 = np.var(X1)

# Función discriminante
def delta(x, mu, sigma, pi):
    return -0.5*np.log(sigma) - ((x - mu)**2)/(2*sigma) + np.log(pi)

# Buscar frontera (numéricamente)
x_vals = np.linspace(min(X)[0], max(X)[0], 1000)

diff = []
for x in x_vals:
    d0 = delta(x, mu0, sigma0, pi0)
    d1 = delta(x, mu1, sigma1, pi1)
    diff.append(d0 - d1)

# Punto donde cambia signo
boundary_qda = x_vals[np.argmin(np.abs(diff))]

print("QDA boundary:", boundary_qda)
