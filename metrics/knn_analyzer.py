import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from datasets.extraction import extract_embeddings

class KNNAnalyzer:
    def __init__(self, data_loader, k=5):
        """
        Parameters:
            data_loader (DataLoader): Loader for the training data (non-augmented)
            k (int): Default number of neighbors for k-NN
        """
        self.data_loader = data_loader
        self.k = k

    def run(self, model, val_embeddings, val_labels, k=None, device='cpu', verbose=True):
        """
        Computes k-NN accuracy between train and validation embeddings.

        Parameters:
            model (torch.nn.Module): The model used to extract training embeddings
            val_embeddings (Tensor): Precomputed embeddings of the validation set
            val_labels (Tensor): Ground-truth labels of the validation set
            k (int, optional): Overrides default k value
            device (str): Device on which to run the model
            verbose (bool): Whether to print progress

        Returns:
            float: k-NN classification accuracy
        """
        k = k or self.k

        # Extract training embeddings from clean data
        train_embeddings, train_labels = extract_embeddings(
            model, self.data_loader, device=device, verbose=False
        )

        # Fit and evaluate k-NN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embeddings.numpy(), train_labels.numpy())
        preds = knn.predict(val_embeddings.numpy())

        return accuracy_score(val_labels.numpy(), preds)
