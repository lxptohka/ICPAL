import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network, a neural network architecture for few-shot learning.

    The Prototypical Network computes a prototype (mean vector) for each class and predicts
    the class of a query sample based on its distance to these prototypes.
    In the contrastive variant, pseudo-prototypes are also introduced to enhance the representation
    by combining initial prototypes with pseudo-prototypes.
    """
    def __init__(self, input_size, embedding_size, num_classes):
        """
        Initialize the Prototypical Network.

        Args:
            input_size (int): Dimension of input features.
            embedding_size (int): Dimension of the embedding space.
            num_classes (int): Number of classes.
            H (int): Number of top-ranked query samples per class based on probability.
            fusion_coefficient (float): Coefficient used to fuse initial prototypes and pseudo-prototypes.
        """
        super(PrototypicalNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )

        self.num_classes = num_classes

    def forward(self, support_x, support_y, query_x, query_y, mode):
        """
        Forward pass.

        Args:
            support_x (torch.Tensor): Input data of the support set, shape (num_support_samples, input_size).
            support_y (torch.Tensor): Labels of the support set, shape (num_support_samples,).
            query_x (torch.Tensor): Input data of the query set, shape (num_query_samples, input_size).

        Returns:
            new_probabilities (torch.Tensor): Probability distribution over classes for query samples, shape (num_query_samples, num_classes).
            new_neg_distances (torch.Tensor): Negative distances between query samples and new prototypes, shape (num_query_samples, num_classes).
            new_prototypes (torch.Tensor): Fused prototype vectors, shape (num_classes, embedding_size).
            support_embeddings (torch.Tensor): Embeddings of the support set, shape (num_support_samples, embedding_size).
            query_embeddings (torch.Tensor): Embeddings of the query set, shape (num_query_samples, embedding_size).
        """
        if mode == 'train':
            # Embed the support and query sets
            support_embeddings = self.embedding(support_x)
            query_embeddings = self.embedding(query_x)

            # Compute class prototypes from all embeddings in the support set
            prototypes = []
            for label in range(self.num_classes):  # 遍历每个类别
                # Select embeddings corresponding to the current class
                class_vectors = support_embeddings[support_y == label]
                # Compute the mean vector of the current class
                prototype = class_vectors.mean(dim=0)
                # Store the result
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            # Compute the probability distribution for query samples
            distances = torch.cdist(query_embeddings, prototypes)   # Compute distances to prototypes
            neg_distances = -distances  # Convert to negative distances
            probabilities = F.softmax(neg_distances, dim=1)     # Apply softmax to get probabilities

            meta_loss = F.cross_entropy(neg_distances, query_y)

            return meta_loss, prototypes
        else:
            # Only compute embeddings
            support_embeddings = self.embedding(support_x)

            return support_embeddings