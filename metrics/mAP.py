import torch
import numpy as np
import faiss



class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features', 'target_labels']
        self.name     = 'mAP'

    def __call__(self, target_labels, features):
        gallery_features,query_features = features[0], features[1]
        labels, freqs = np.unique(target_labels, return_counts=True)
        R             = len(gallery_features)

        faiss_search_index  = faiss.IndexFlatL2(gallery_features.shape[-1])
        if isinstance(gallery_features, torch.Tensor):
            gallery_features = gallery_features.detach().cpu().numpy()
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)        
        faiss_search_index.add(gallery_features)
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.detach().cpu().numpy()
        nearest_neighbours  = faiss_search_index.search(query_features, int(R+1))[1][:,1:]

        target_labels = target_labels.reshape(-1)
        nn_labels = target_labels[nearest_neighbours]

        avg_r_precisions = []
        for label, freq in zip(labels, freqs):
            rows_with_label = np.where(target_labels==label)[0]
            for row in rows_with_label:
                n_recalled_samples           = np.arange(1,R+1)
                target_label_occ_in_row      = nn_labels[row,:]==label
                cumsum_target_label_freq_row = np.cumsum(target_label_occ_in_row)
                avg_r_pr_row = np.sum(cumsum_target_label_freq_row*target_label_occ_in_row/n_recalled_samples)/freq
                avg_r_precisions.append(avg_r_pr_row)

        return np.mean(avg_r_precisions)
