import torch

class AdaptiveInstanceNorm2d(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def _get_mean(self, features):
        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        return features_mean
    
    def _get_std(self, features):
        batch_size, c = features.size()[:2]
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + self.eps
        return features_std

    def forward(self, content, style):
        content_mean, content_std = self._get_mean(content), self._get_std(content)
        style_mean, style_std = self._get_mean(style), self._get_std(style)
        normalized = style_std * (content - content_mean) / content_std + style_mean
        return normalized