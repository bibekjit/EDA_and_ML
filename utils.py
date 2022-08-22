import numpy as np


class LogScaler:
    def __init__(self):
        self.X_factor = {}
        self.y_factor = None

    def fit_transform(self, X, y):
        for i in range(X.shape[1]):
            if X[:, i].min() >= 0:
                X[:, i] = np.log(X[:, i] + 1)
            else:
                self.X_factor[i] = X[:, i].min()
                X[:, i] -= self.X_factor[i]
                X[:, i] = np.log(X[:, i] + 1)

        if y.min() >= 0:
            y = np.log(y+1)
        else:
            self.y_factor = y.min()
            y -= self.y_factor
            y = np.log(y+1)

        return X,y

    def transform(self,X):
        for i in range(X.shape[1]):
            if i not in self.X_factor:
                X[:,i] = np.log(X[:,i]+1)
            else:
                X[:, i] -= self.X_factor[i]
                if X[:,i].min() < self.X_factor[i]:
                    rows = X.shape[0]
                    for j in range(rows):
                        if X[j,i] < self.X_factor[i]:
                            X[j,i] = 0
                X[:, i] = np.log(X[:, i] + 1)

        return X








