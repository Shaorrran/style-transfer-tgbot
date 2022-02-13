import collections
import pathlib

import PIL

import skimage
import skimage.color

import torch
import torchvision

import numpy as np

from architectures import kmeans

class StyleTransferCNN(torch.nn.Module):
    def __init__(self, alpha=1, pretrained=False):
        super().__init__()
        self.alpha = alpha
        torchhub_save_dir = (pathlib.Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
        torchhub_save_dir.mkdir(parents=True, exist_ok=True)
        if not (torchhub_save_dir / "vgg_normalized_conv5_1.pth").is_file():
                download_from_gdrive("1IAOFF5rDkVei035228Qp35hcTnliyMol", torchhub_save_dir / "vgg_normalized_conv5_1.pth")
        self.encoder = NormalizedVGG(pretrained_path=(torchhub_save_dir / "vgg_normalized_conv5_1.pth"))
        if pretrained:
            if not (torchhub_save_dir / "decoder_relu4_1.pth").is_file():
                download_from_gdrive("1kkoyNwRup9y5GT1mPbsZ_7WPQO9qB7ZZ", torchhub_save_dir / "decoder_relu4_1.pth")
            self.decoder = Decoder(level=4, pretrained_path="decoder_relu4_1.pth")
        else:
            self.decoder = Decoder(level=4)
    
    def _apply(self, fn):
        # redefine to move encoder and decoder to the same device the main model is on
        super()._apply(fn)
        self.encoder._apply(fn)
        self.decoder._apply(fn)
        return self

    def _calc_k(self, x, max_cluster=5, threshold_min=0.1, threshold_max=0.7):
        image = torchvision.transforms.ToPILImage()(x).convert("RGB")
        w, h = image.size
        w, h = self._calc_maxpool_size(w, h, 3)
        image = image.resize((w, h))
        image = skimage.color.rgb2lab(image).reshape(w * h, -1)
        k = 2
        k_means = kmeans.KMeans(k, device=x.device)
        image = torch.from_numpy(image).to(k_means.device)
        k_means.fit(image)
        labels = k_means.labels_
        prev_labels = k_means.labels_
        prev_cluster_centers = k_means.cluster_centers_
        while True:
            cnt = collections.Counter(labels.cpu().tolist())
            if k <= max_cluster and (cnt.most_common()[-1][1] / (w * h) > threshold_min or cnt.most_common()[0][1] / (
                w * h) > threshold_max):
                if cnt.most_common()[-2][1] / (w * h) < threshold_min:
                    labels = prev_labels
                    cluster_centers = prev_cluster_centers
                    k -= 1
                    break
                k += 1
            else:
                if k > max_cluster:
                    labels = prev_labels
                    cluster_centers = prev_cluster_centers
                    k -= 1
                else:
                    labels = k_means.labels_
                    cluster_centers = k_means.cluster_centers_
                break

            prev_labels = k_means.labels_
            prev_cluster_centers = k_means.cluster_centers_

            k_means = kmeans.KMeans(k, device=x.device)
            k_means.fit(image)
            labels = k_means.labels_
        
        new_clusters = cluster_centers.norm(dim=1).argsort(descending=False).tolist()
        new_clusters = [new_clusters.index(j) for j in range(k)]
        cluster_centers_norm, _ = torch.sort(cluster_centers.norm(dim=1))
        cluster_centers_norm -= cluster_centers_norm.min()
        cluster_centers_norm /= cluster_centers_norm.max()

        new_labels = torch.zeros_like(labels)
        for i in range(k):
            new_labels[labels == i] = new_clusters[i]
        
        label = new_labels.reshape(h, w)

        return label, cluster_centers_norm

    def _calc_maxpool_size(self, w, h, count=3):
        if count == 3:
            w = np.ceil(np.ceil(np.ceil(w / 2) / 2) / 2)
            h = np.ceil(np.ceil(np.ceil(h / 2) / 2) / 2)
        elif count == 2:
            w = np.ceil(np.ceil(w / 2) / 2)
            h = np.ceil(np.ceil(h / 2) / 2)
        elif count == 1:
            w = np.ceil(w / 2)
            h = np.ceil(h / 2)
        else:
            raise ValueError
        return int(w), int(h)

    def _cluster_matching(self, content_label, style_label, content_cluster_center_norm, style_cluster_center_norm, threshold=0.25):
        content_k = int(content_label.max().item() + 1)
        style_k = int(style_label.max().item() + 1)
        res = {}
        for i in range(content_k):
            res[i] = []
            match = False
            for j in range(style_k):
                if torch.abs(content_cluster_center_norm[i] - style_cluster_center_norm[j]) <= threshold:
                    match = True
                    res[i].append(j)
            if not match:
                res[i] = [jj for jj in range(style_k)]
        
        return res

    def _labeled_whiten_and_color(self, f_c, f_s, alpha, clabel):
        try:
            cc, ch, cw = f_c.shape
            cf = (f_c * clabel).reshape(cc, -1)
            num_nonzero = torch.sum(clabel).item() / cc
            c_mean = (torch.sum(cf, 1) / num_nonzero).reshape(cc, 1, 1) * clabel
            cf = (cf.reshape(cc, ch, cw) - c_mean).reshape(cc, -1)
            
            c_cov = torch.mm(cf, ct.t()) / (num_nonzero - 1)
            c_u, c_e, c_v = torch.svd(c_cov)
            c_d = c_e.pow(-0.5)

            w_step1 = torch.mm(c_v, torch.diag(c_d))
            w_step2 = torch.mm(w_step1, (c_v.t()))
            whitened = torch.mm(w_step2, cf)

            sf = f_s
            sc, shw = sf.shape
            s_mean = torch.mean(f_s, 1, keepdim=True)
            sf -= s_mean

            s_cov = torch.mm(sf, sf.t()) / (shw - 1)
            s_u, s_e, s_v = torch.svd(s_cov)
            s_d = s_e.pow(-0.5)

            c_step1 = torch.mm(s_v, torch.diag(s_d))
            c_step2 = torch.mm(c_step1, s_v.t())
            colored = torch.mm(c_step2, whitened).reshape(cc, ch, cw)

            colored += s_mean.reshape(sc, 1, 1) * clabel
            colored_feature = alpha * colored + (1 - alpha) * (f_c * clabel)
        except:
            colored_feature = f_c * clabel
        
        return colored_feature

    def generate(self, content, style):
        cs = []
        content_features = self.encoder(content, target="neck")
        style_features = self.encoder(style, target="neck")

        for c_image, s_image, c_feat, s_feat in zip(content, style, content_features, style_features):
            content_label, content_center_norm = self._calc_k(c_image)
            style_label, style_center_norm = self._calc_k(s_image)
            match = self._cluster_matching(content_label, style_label, content_center_norm, style_center_norm)

            cs_feature = torch.zeros_like(c_feat)
            for i, j in match.items():
                cl = (content_label == i).unsqueeze(dim=0).expand_as(c_feat).to(torch.float)
                sl = torch.zeros_like(s_feat)
                for jj in j:
                    sl += (style_label == jj).unsqueeze(dim=0).expand_as(s_feat).to(torch.float)
                sl = sl.to(torch.bool)
                sub_sf = s_feat[sl].reshape(s_feat.shape[0], -1)
                cs_feature += self._labeled_whiten_and_color(c_feat, sub_sf, self.alpha, cl)

            cs.append(cs_feature.unsqueeze(dim=0))
        cs = torch.cat(cs, dim=0)
        out = self.decoder(cs)
        return out

    def forward(self, content, style):
        out = self.generate(content, style)
        content_features = self.encoder(content)
        out_features = self.encoder(out, output_last_feature=True)
        out_middle_features = self.encoder(out, output_last_feature=False)
        style_middle_features = self.encoder(style, output_last_feature=False)
        return out, content_features, out_features, out_middle_features, style_middle_features