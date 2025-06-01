import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.extraction import extract_embeddings

class KNRAnalyzer:
    def __init__(self, data_loader, k=5, metric='mae'):
        self.data_loader = data_loader
        self.k = k
        self.metric = metric

    def run(self, model, val_embeddings, val_labels, k=None, device='cpu', verbose=True):
        k = k or self.k

        # Extract training embeddings
        train_embeddings, train_labels = extract_embeddings(
            model, self.data_loader, device=device, verbose=False
        )

        # Fit and evaluate k-NN regressor
        knr = KNeighborsRegressor(n_neighbors=k)
        knr.fit(train_embeddings.numpy(), train_labels.numpy())
        preds = knr.predict(val_embeddings.numpy())

        # Choose regression metric
        if self.metric == 'mae':
            return mean_absolute_error(val_labels.numpy(), preds)
        elif self.metric == 'rmse':
            return np.sqrt(mean_squared_error(val_labels.numpy(), preds))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
