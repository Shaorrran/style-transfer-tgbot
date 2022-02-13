import torch

import numpy as np

class KMeans:
    def __init__(self, n_clusters, device="cpu", tol=1e-4, init="kmeans++"):
        self.n_clusters = n_clusters
        self.device = device
        self.tol = tol
        self.init = init
        self._labels = None
        self._cluster_centers = None

    def _initial_state(self, data):
        if self.init == "kmeans++":
            n, c = data.shape
            dis = torch.zeros((n, self.n_clusters), device=self.device)
            initial_state = torch.zeros((self.n_clusters, c), device=self.device)
            pr = np.repeat(1 / n, n)
            initial_state[0, :] = data[np.random.choice(np.arange(n), p=pr)]
            dis[:, 0] = torch.sum((data - initial_state[0, :]) ** 2, dim=1)

            for k in range(1, self.n_clusters):
                pr = torch.sum(dis, dim=1)/ torch.sum(dis)
                initial_state[k, :] = data[np.random.choice(np.arange(n), 1, p=pr.cpu().numpy())]
                dis[:, k] = torch.sum((data - initial_state[k, :]) ** 2, dim=1)
        else:
            n = data.shape[0]
            indices = np.random.choice(n, self.n_clusters)
            initial_state = data[indices]

        return initial_state

    @staticmethod
    def pairwise_distance(data1, data2=None):
        if data2 is None:
            data2 = data1

        a = data1.unsqueeze(dim=1)
        b = data2.unsqueeze(dim=0)

        dis = (a - b) ** 2.0
        dis = dis.sum(dim=-1).squeeze()
        return dis
    
    def fit(self, data):
        data = data.to(torch.float32)
        cluster_centers = self._initial_state(data)

        while True:
            dis = self.pairwise_distance(data, cluster_centers)
            labels = torch.argmin(dis, dim=1)
            cluster_centers_pre = cluster_centers.clone()

            for index in range(self.n_clusters):
                selected = (labels == index)
                if selected.any():
                    selected = data[labels == index]
                    cluster_centers[index] = selected.mean(dim=0)
                else:
                    cluster_centers[index] = torch.zeros_like(cluster_centers[0], device=self.device)
            
            center_shift = torch.sum(torch.sqrt(torch.sum((cluster_centers - cluster_centers_pre) ** 2, dim=1)))
            if center_shift ** 2 < self.tol:
                break
        
        self._labels = labels
        self._cluster_centers = cluster_centers

    @property
    def labels_(self):
        return self._labels
    
    @property
    def cluster_centers_(self):
        return self._cluster_centers